from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
from typing import Any, Optional
import re

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.control.session_controller import SessionController
from src.network.wifi_monitor import WifiMonitor
from src.state.runtime_state import RuntimeState
from src.telemetry.battery_listener import BatteryConfig, BatteryListener


_THERMAL_ZONE_CACHE: dict[str, Path | None] | None = None
_THERMAL_SOURCE_LOGGED = False
_THERMAL_FAILURE_LOGGED = False
_TEGRSTATS_SOURCE_LOGGED = False
_TEGRSTATS_FAILURE_LOGGED = False


class SessionStartRequest(BaseModel):
    session_name: Optional[str] = None


class BackgroundBatteryMonitor:
    def __init__(self, config: BatteryConfig, runtime_state: RuntimeState, logger: logging.Logger) -> None:
        self.config = config
        self.runtime_state = runtime_state
        self.logger = logger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="status-battery-monitor", daemon=True)
        self._thread.start()
        self.logger.info("Background battery monitor started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        listener: BatteryListener | None = None

        while not self._stop_event.is_set():
            try:
                if self.runtime_state.session_running:
                    if listener is not None:
                        listener.close()
                        listener = None
                    self._stop_event.wait(self.config.poll_interval)
                    continue

                if listener is None:
                    listener = BatteryListener(self.config, logger=self.logger)
                    listener.connect()

                record = listener.read_record(timeout=self.config.poll_interval)
                if record is None:
                    continue

                low_warning = (
                    record.remaining_percent is not None
                    and record.remaining_percent <= self.config.low_remaining_threshold
                )
                self.runtime_state.set_battery_state(
                    timestamp_iso=record.timestamp_iso,
                    unix_time=record.unix_time,
                    voltage_v=record.voltage_v,
                    current_a=record.current_a,
                    remaining_percent=record.remaining_percent,
                    battery_temp_c=record.battery_temp_c,
                    low_warning=low_warning,
                )
                self.runtime_state.update_component("battery", ready=True, running=False, ok=True, last_error=None)
            except Exception as exc:
                self.logger.warning("Background battery monitor read failed: %s", exc)
                self.runtime_state.update_component("battery", running=False, ok=False, last_error=str(exc))
                if listener is not None:
                    listener.close()
                    listener = None
                self._stop_event.wait(max(self.config.poll_interval, 1.0))

        if listener is not None:
            listener.close()


def find_latest_session_dir(project_root: Path) -> Path | None:
    sessions_dir = project_root / "data" / "sessions"
    if not sessions_dir.exists():
        return None

    session_dirs = [path for path in sessions_dir.iterdir() if path.is_dir()]
    if not session_dirs:
        return None

    return sorted(session_dirs, key=lambda path: path.name, reverse=True)[0]


def load_json_metadata(metadata_path: Path, logger: logging.Logger) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load metadata file %s: %s", metadata_path, exc)
        return None


def _as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_latest_battery_row(csv_path: Path, logger: logging.Logger) -> dict[str, Any]:
    if not csv_path.exists():
        return {"status": "no_file", "message": "battery log file not found"}

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except Exception as exc:
        logger.warning("Failed to read battery log %s: %s", csv_path, exc)
        return {"status": "error", "message": str(exc)}

    if not rows:
        return {"status": "no_data", "message": "battery log is empty"}

    latest = rows[-1]
    return {
        "voltage": _as_float(latest.get("voltage_v")),
        "current": _as_float(latest.get("current_a")),
        "percent": _as_float(latest.get("remaining_percent")),
        "battery_temp_c": _as_float(latest.get("battery_temp_c")),
        "last_updated": _as_float(latest.get("timestamp")),
    }


def _recent_battery_rows(csv_path: Path, logger: logging.Logger, sample_count: int = 5) -> list[dict[str, float | None]]:
    if not csv_path.exists():
        return []

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except Exception as exc:
        logger.warning("Failed to read battery history %s: %s", csv_path, exc)
        return []

    recent_rows = rows[-sample_count:]
    return [
        {
            "unix_time": _as_float(row.get("timestamp")),
            "voltage_v": _as_float(row.get("voltage_v")),
            "current_a": _as_float(row.get("current_a")),
            "remaining_percent": _as_float(row.get("remaining_percent")),
            "battery_temp_c": _as_float(row.get("battery_temp_c")),
        }
        for row in recent_rows
    ]


def _base_battery_state(voltage_v: float | None, battery_temp_c: float | None) -> str:
    if voltage_v is None and battery_temp_c is None:
        return "WARNING"
    if voltage_v is not None and voltage_v < 13.2:
        return "EMERGENCY"
    if battery_temp_c is not None and battery_temp_c >= 60.0:
        return "EMERGENCY"
    if voltage_v is not None and voltage_v < 14.0:
        return "CRITICAL"
    if battery_temp_c is not None and battery_temp_c >= 55.0:
        return "CRITICAL"
    if voltage_v is not None and voltage_v < 14.8:
        return "WARNING"
    if battery_temp_c is not None and battery_temp_c >= 45.0:
        return "WARNING"
    return "NORMAL"


def _escalate_battery_state(state: str) -> str:
    order = ["NORMAL", "WARNING", "CRITICAL", "EMERGENCY"]
    try:
        index = order.index(state)
    except ValueError:
        return state
    return order[min(index + 1, len(order) - 1)]


def classify_battery_status(
    voltage_v: float | None,
    current_a: float | None,
    battery_temp_c: float | None,
    recent_rows: list[dict[str, float | None]],
) -> dict[str, Any]:
    battery_state = _base_battery_state(voltage_v, battery_temp_c)
    sag_detected = False

    valid_rows = [row for row in recent_rows if row.get("unix_time") is not None and row.get("voltage_v") is not None]
    if len(valid_rows) >= 2:
        latest_row = valid_rows[-1]
        for previous_row in reversed(valid_rows[:-1]):
            if latest_row["unix_time"] is None or previous_row["unix_time"] is None:
                continue
            if latest_row["unix_time"] - previous_row["unix_time"] > 5.0:
                break
            voltage_drop = (previous_row["voltage_v"] or 0.0) - (latest_row["voltage_v"] or 0.0)
            if voltage_drop > 0.5:
                sag_detected = True
                battery_state = _escalate_battery_state(battery_state)
                break

    return {
        "battery_state": battery_state,
        "battery_voltage": voltage_v,
        "battery_current": current_a,
        "battery_temp": battery_temp_c,
        "voltage_sag_detected": sag_detected,
    }


def read_sonar_status(csv_path: Path, logger: logging.Logger, sample_count: int = 10) -> dict[str, Any]:
    if not csv_path.exists():
        return {
            "distance_mm": None,
            "distance_m": None,
            "confidence": None,
            "sample_count_used": 0,
            "variation_mm": None,
            "stable": False,
            "status": "no_data",
            "last_updated": None,
        }

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except Exception as exc:
        logger.warning("Failed to read sonar log %s: %s", csv_path, exc)
        return {
            "distance_mm": None,
            "distance_m": None,
            "confidence": None,
            "sample_count_used": 0,
            "variation_mm": None,
            "stable": False,
            "status": "no_data",
            "last_updated": None,
        }

    valid_rows = [
        row
        for row in rows
        if _as_float(row.get("timestamp")) is not None and _as_float(row.get("distance_mm")) is not None
    ]
    if not valid_rows:
        return {
            "distance_mm": None,
            "distance_m": None,
            "confidence": None,
            "sample_count_used": 0,
            "variation_mm": None,
            "stable": False,
            "status": "no_data",
            "last_updated": None,
        }

    recent_rows = valid_rows[-sample_count:]
    distances = [int(_as_float(row.get("distance_mm")) or 0) for row in recent_rows]
    confidences = [
        _as_float(row.get("confidence"))
        for row in recent_rows
        if _as_float(row.get("confidence")) is not None
    ]
    latest_row = recent_rows[-1]
    latest_distance_mm = distances[-1]
    latest_confidence = _as_float(latest_row.get("confidence"))
    average_confidence = statistics.mean(confidences) if confidences else None
    variation_mm = max(distances) - min(distances) if len(distances) >= 2 else 0.0
    status = "stable"
    stable = True

    if average_confidence is None or average_confidence < 70:
        status = "weak_signal"
        stable = False
    elif variation_mm > 80:
        status = "unstable"
        stable = False

    return {
        "distance_mm": latest_distance_mm,
        "distance_m": round(latest_distance_mm / 1000.0, 3),
        "confidence": latest_confidence,
        "sample_count_used": len(recent_rows),
        "variation_mm": round(float(variation_mm), 2),
        "stable": stable,
        "status": status,
        "last_updated": _as_float(latest_row.get("timestamp")),
    }


def _parse_tegrastats_metric(output: str, pattern: str) -> float | None:
    match = re.search(pattern, output, re.IGNORECASE)
    if not match:
        return None
    try:
        return round(float(match.group(1)), 2)
    except ValueError:
        return None


def _read_tegrastats_status(logger: logging.Logger) -> dict[str, Any] | None:
    global _TEGRSTATS_SOURCE_LOGGED
    global _TEGRSTATS_FAILURE_LOGGED

    try:
        result = subprocess.run(
            ["bash", "-lc", "tegrastats | head -n 1"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except Exception as exc:
        if not _TEGRSTATS_FAILURE_LOGGED:
            logger.warning("Failed to execute tegrastats: %s", exc)
            _TEGRSTATS_FAILURE_LOGGED = True
        return None

    output = (result.stdout or "").strip()
    if result.returncode != 0 or not output:
        if not _TEGRSTATS_FAILURE_LOGGED:
            logger.warning("tegrastats returned no usable output: %s", result.stderr.strip() or output)
            _TEGRSTATS_FAILURE_LOGGED = True
        return None

    cpu_temp_c = _parse_tegrastats_metric(output, r"CPU@([0-9]+(?:\.[0-9]+)?)C")
    gpu_temp_c = _parse_tegrastats_metric(output, r"GPU@([0-9]+(?:\.[0-9]+)?)C")
    tj_temp_c = _parse_tegrastats_metric(output, r"tj@([0-9]+(?:\.[0-9]+)?)C")
    board_temp_c = _parse_tegrastats_metric(output, r"(?:Tboard(?:_tegra)?|AO)@([0-9]+(?:\.[0-9]+)?)C")
    power_mw = _parse_tegrastats_metric(output, r"VDD_IN\s+([0-9]+(?:\.[0-9]+)?)mW")

    if all(value is None for value in (cpu_temp_c, gpu_temp_c, tj_temp_c, board_temp_c, power_mw)):
        if not _TEGRSTATS_FAILURE_LOGGED:
            logger.warning("tegrastats output did not match expected metrics: %s", output)
            _TEGRSTATS_FAILURE_LOGGED = True
        return None

    if not _TEGRSTATS_SOURCE_LOGGED:
        logger.info("System source detected: tegrastats")
        _TEGRSTATS_SOURCE_LOGGED = True
    _TEGRSTATS_FAILURE_LOGGED = False

    return {
        "cpu_temp_c": cpu_temp_c,
        "gpu_temp_c": gpu_temp_c,
        "tj_temp_c": tj_temp_c,
        "board_temp_c": board_temp_c,
        "power_in_w": round(power_mw / 1000.0, 3) if power_mw is not None else None,
        "source": "tegrastats",
    }


def _find_thermal_zone_cache(logger: logging.Logger) -> dict[str, Path | None]:
    global _THERMAL_ZONE_CACHE
    global _THERMAL_SOURCE_LOGGED

    if _THERMAL_ZONE_CACHE is not None:
        return _THERMAL_ZONE_CACHE

    zone_map: dict[str, Path | None] = {"cpu": None, "gpu": None, "board": None}
    thermal_root = Path("/sys/class/thermal")

    for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
        type_path = zone_dir / "type"
        temp_path = zone_dir / "temp"
        if not type_path.exists() or not temp_path.exists():
            continue

        try:
            zone_type = type_path.read_text(encoding="utf-8").strip().lower()
        except Exception:
            continue

        if zone_map["cpu"] is None and "cpu" in zone_type:
            zone_map["cpu"] = temp_path
        elif zone_map["gpu"] is None and "gpu" in zone_type:
            zone_map["gpu"] = temp_path
        elif zone_map["board"] is None and any(token in zone_type for token in ("board", "ao", "soc", "cv")):
            zone_map["board"] = temp_path

    _THERMAL_ZONE_CACHE = zone_map
    if not _THERMAL_SOURCE_LOGGED:
        logger.info(
            "System thermal source detected. cpu=%s gpu=%s board=%s",
            zone_map["cpu"],
            zone_map["gpu"],
            zone_map["board"],
        )
        _THERMAL_SOURCE_LOGGED = True
    return zone_map


def _read_temp_c(temp_path: Path | None) -> float | None:
    if temp_path is None or not temp_path.exists():
        return None

    raw_value = temp_path.read_text(encoding="utf-8").strip()
    value = float(raw_value)
    if value > 1000:
        value /= 1000.0
    return round(value, 2)


def read_system_status(logger: logging.Logger) -> dict[str, Any]:
    global _THERMAL_FAILURE_LOGGED

    try:
        tegrastats_status = _read_tegrastats_status(logger)
        if tegrastats_status is not None:
            available_temps = [
                temp
                for temp in (
                    tegrastats_status.get("cpu_temp_c"),
                    tegrastats_status.get("gpu_temp_c"),
                    tegrastats_status.get("tj_temp_c"),
                    tegrastats_status.get("board_temp_c"),
                )
                if temp is not None
            ]
            if available_temps:
                max_temp = max(available_temps)
                if max_temp < 70.0:
                    status = "normal"
                elif max_temp < 80.0:
                    status = "warning"
                else:
                    status = "hot"
            else:
                status = "unknown"

            return {
                "cpu_temp_c": tegrastats_status.get("cpu_temp_c"),
                "gpu_temp_c": tegrastats_status.get("gpu_temp_c"),
                "tj_temp_c": tegrastats_status.get("tj_temp_c"),
                "board_temp_c": tegrastats_status.get("board_temp_c"),
                "power_in_w": tegrastats_status.get("power_in_w"),
                "status": status,
                "source": tegrastats_status.get("source"),
            }

        zone_map = _find_thermal_zone_cache(logger)
        cpu_temp_c = _read_temp_c(zone_map["cpu"])
        gpu_temp_c = _read_temp_c(zone_map["gpu"])
        board_temp_c = _read_temp_c(zone_map["board"])
        available_temps = [temp for temp in (cpu_temp_c, gpu_temp_c, board_temp_c) if temp is not None]

        if not available_temps:
            return {
                "cpu_temp_c": None,
                "gpu_temp_c": None,
                "tj_temp_c": None,
                "board_temp_c": None,
                "power_in_w": None,
                "status": "unknown",
                "source": "unknown",
            }

        max_temp = max(available_temps)
        if max_temp < 70.0:
            status = "normal"
        elif max_temp < 80.0:
            status = "warning"
        else:
            status = "hot"

        _THERMAL_FAILURE_LOGGED = False
        return {
            "cpu_temp_c": cpu_temp_c,
            "gpu_temp_c": gpu_temp_c,
            "tj_temp_c": None,
            "board_temp_c": board_temp_c,
            "power_in_w": None,
            "status": status,
            "source": "thermal_zone",
        }
    except Exception as exc:
        if not _THERMAL_FAILURE_LOGGED:
            logger.warning("Failed to read Jetson thermal information: %s", exc)
            _THERMAL_FAILURE_LOGGED = True
        return {
            "cpu_temp_c": None,
            "gpu_temp_c": None,
            "tj_temp_c": None,
            "board_temp_c": None,
            "power_in_w": None,
            "status": "unknown",
            "source": "unknown",
        }


def resolve_last_error(snapshot: dict[str, Any]) -> str | None:
    for component_name in ("camera", "sonar", "battery", "network", "server"):
        component = snapshot.get(component_name, {})
        if component.get("last_error"):
            return str(component["last_error"])
    return None


def create_status_app(
    project_root: Path,
    runtime_state: RuntimeState,
    session_controller: SessionController,
    *,
    logger: logging.Logger,
    wifi_monitor: WifiMonitor | None = None,
    background_battery_monitor: BackgroundBatteryMonitor | None = None,
) -> FastAPI:
    app = FastAPI(title="Underwater Acquisition Kit Status Server")

    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Underwater Acquisition Kit</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7f8;
      --card: #ffffff;
      --text: #102027;
      --muted: #607d8b;
      --ok: #1b8f4d;
      --bad: #c0392b;
      --accent: #0b7285;
      --border: #d8e2e7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef5f7 0%, var(--bg) 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 680px;
      margin: 0 auto;
      padding: 20px 16px 32px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 14px;
      box-shadow: 0 8px 24px rgba(16, 32, 39, 0.06);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.5rem;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }
    button {
      border: none;
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 1rem;
      font-weight: 600;
      color: white;
      cursor: pointer;
    }
    button.start { background: var(--ok); }
    button.stop { background: var(--bad); }
    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
    }
    .status-row:last-child { border-bottom: none; }
    .label { color: var(--muted); }
    .value {
      text-align: right;
      font-weight: 600;
      word-break: break-word;
    }
    .pill {
      display: inline-block;
      min-width: 88px;
      text-align: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.9rem;
      color: white;
      background: var(--bad);
    }
    .pill.ok { background: var(--ok); }
    .pill.warn { background: #d97706; }
    .pill.unknown { background: #6b7280; }
    .message {
      margin-top: 12px;
      min-height: 24px;
      color: var(--accent);
      font-size: 0.95rem;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Underwater Acquisition Kit</h1>
      <p>Remote monitor and control panel</p>
      <div class="actions">
        <button class="start" onclick="startSession()">START</button>
        <button class="stop" onclick="stopSession()">STOP</button>
      </div>
      <div class="message" id="message">Ready.</div>
    </div>

    <div class="card">
      <div class="status-row"><span class="label">session_active</span><span class="value" id="session_active">-</span></div>
      <div class="status-row"><span class="label">active_session_id</span><span class="value" id="active_session_id">-</span></div>
      <div class="status-row"><span class="label">battery_voltage</span><span class="value" id="battery_voltage">-</span></div>
      <div class="status-row"><span class="label">battery_percent</span><span class="value" id="battery_percent">-</span></div>
      <div class="status-row"><span class="label">battery_temp</span><span class="value" id="battery_temp">-</span></div>
      <div class="status-row"><span class="label">battery_state</span><span class="value" id="battery_state">-</span></div>
      <div class="status-row"><span class="label">camera_running</span><span class="value" id="camera_running">-</span></div>
      <div class="status-row"><span class="label">sonar_running</span><span class="value" id="sonar_running">-</span></div>
    </div>

    <div class="card">
      <div class="status-row"><span class="label">sonar_distance</span><span class="value" id="sonar_distance">-</span></div>
      <div class="status-row"><span class="label">sonar_confidence</span><span class="value" id="sonar_confidence">-</span></div>
      <div class="status-row"><span class="label">sonar_variation</span><span class="value" id="sonar_variation">-</span></div>
      <div class="status-row"><span class="label">sonar_status</span><span class="value" id="sonar_status">-</span></div>
    </div>

    <div class="card">
      <div class="status-row"><span class="label">cpu_temp</span><span class="value" id="cpu_temp">-</span></div>
      <div class="status-row"><span class="label">gpu_temp</span><span class="value" id="gpu_temp">-</span></div>
      <div class="status-row"><span class="label">tj_temp</span><span class="value" id="tj_temp">-</span></div>
      <div class="status-row"><span class="label">board_temp</span><span class="value" id="board_temp">-</span></div>
      <div class="status-row"><span class="label">vdd_in_power</span><span class="value" id="vdd_in_power">-</span></div>
      <div class="status-row"><span class="label">thermal_status</span><span class="value" id="thermal_status">-</span></div>
    </div>
  </div>

  <script>
    function setText(id, value) {
      document.getElementById(id).textContent = value ?? "-";
    }

    function setBool(id, value) {
      setStatusPill(id, Boolean(value) ? 'true' : 'false', Boolean(value) ? 'ok' : 'bad');
    }

    function showMessage(text) {
      document.getElementById('message').textContent = text;
    }

    function setStatusPill(id, text, tone) {
      const element = document.getElementById(id);
      element.innerHTML = '<span class="pill ' + tone + '">' + text + '</span>';
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options);
      return await response.json();
    }

    async function refreshStatus() {
      try {
        const [health, battery, sonar, system] = await Promise.all([
          fetchJson('/health'),
          fetchJson('/battery'),
          fetchJson('/sonar'),
          fetchJson('/system'),
        ]);

        setBool('session_active', health.session_active);
        setText('active_session_id', health.active_session_id || '-');
        setBool('camera_running', health.camera_running);
        setBool('sonar_running', health.sonar_running);

        if (battery && battery.voltage != null) {
          setText('battery_voltage', Number(battery.voltage).toFixed(2) + ' V');
        } else {
          setText('battery_voltage', battery.status || '-');
        }

        if (battery && battery.percent != null) {
          setText('battery_percent', Number(battery.percent).toFixed(1) + ' %');
        } else {
          setText('battery_percent', battery.status || '-');
        }

        if (battery && battery.battery_temp_c != null) {
          setText('battery_temp', Number(battery.battery_temp_c).toFixed(1) + ' C');
        } else {
          setText('battery_temp', '-');
        }

        if (battery && battery.battery_state) {
          if (battery.battery_state === 'NORMAL') {
            setStatusPill('battery_state', battery.battery_state, 'ok');
          } else if (battery.battery_state === 'WARNING') {
            setStatusPill('battery_state', battery.battery_state, 'warn');
          } else {
            setStatusPill('battery_state', battery.battery_state, 'bad');
          }
        } else {
          setStatusPill('battery_state', 'unknown', 'unknown');
        }

        if (sonar && sonar.distance_m != null) {
          setText('sonar_distance', Number(sonar.distance_m).toFixed(3) + ' m');
        } else {
          setText('sonar_distance', sonar.status || '-');
        }

        if (sonar && sonar.confidence != null) {
          setText('sonar_confidence', Number(sonar.confidence).toFixed(1));
        } else {
          setText('sonar_confidence', '-');
        }

        if (sonar && sonar.variation_mm != null) {
          setText('sonar_variation', Number(sonar.variation_mm).toFixed(1) + ' mm');
        } else {
          setText('sonar_variation', '-');
        }

        if (sonar.status === 'stable') {
          setStatusPill('sonar_status', sonar.status, 'ok');
        } else if (sonar.status === 'no_data') {
          setStatusPill('sonar_status', sonar.status, 'unknown');
        } else {
          setStatusPill('sonar_status', sonar.status || 'unknown', 'bad');
        }

        setText('cpu_temp', system.cpu_temp_c != null ? Number(system.cpu_temp_c).toFixed(1) + ' C' : '-');
        setText('gpu_temp', system.gpu_temp_c != null ? Number(system.gpu_temp_c).toFixed(1) + ' C' : '-');
        setText('tj_temp', system.tj_temp_c != null ? Number(system.tj_temp_c).toFixed(1) + ' C' : '-');
        setText('board_temp', system.board_temp_c != null ? Number(system.board_temp_c).toFixed(1) + ' C' : '-');
        setText('vdd_in_power', system.power_in_w != null ? Number(system.power_in_w).toFixed(2) + ' W' : '-');

        if (system.status === 'normal') {
          setStatusPill('thermal_status', system.status, 'ok');
        } else if (system.status === 'warning') {
          setStatusPill('thermal_status', system.status, 'warn');
        } else if (system.status === 'hot') {
          setStatusPill('thermal_status', system.status, 'bad');
        } else {
          setStatusPill('thermal_status', system.status || 'unknown', 'unknown');
        }
      } catch (error) {
        showMessage('Status update failed: ' + error);
      }
    }

    async function startSession() {
      showMessage('Starting session...');
      try {
        const result = await fetchJson('/session/start');
        showMessage(result.message || 'session start requested');
        await refreshStatus();
      } catch (error) {
        showMessage('Start failed: ' + error);
      }
    }

    async function stopSession() {
      showMessage('Stopping session...');
      try {
        const result = await fetchJson('/session/stop');
        showMessage(result.message || 'session stop requested');
        await refreshStatus();
      } catch (error) {
        showMessage('Stop failed: ' + error);
      }
    }

    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""

    def start_session_response(session_name: str | None = None, request_method: str = "POST") -> dict[str, Any]:
        logger.info("Session start requested. method=%s session_name=%s", request_method, session_name)
        result = session_controller.start_session(session_name=session_name)
        if result.get("ok"):
            logger.info("Session started. session_id=%s", result.get("session_id"))
        else:
            logger.info("Duplicate start rejected. session_id=%s", result.get("session_id"))
        return {
            "success": bool(result.get("ok")),
            "session_id": result.get("session_id"),
            "message": result.get("message"),
            "started_at": runtime_state.snapshot().get("session_started_at"),
        }

    def stop_session_response(request_method: str = "POST") -> dict[str, Any]:
        logger.info("Session stop requested. method=%s", request_method)
        result = session_controller.stop_session()
        if not result.get("ok"):
            logger.info("Stop requested with no active session.")
            return {
                "success": False,
                "session_id": result.get("session_id"),
                "message": result.get("message"),
                "stopped_at": runtime_state.snapshot().get("last_updated_at"),
            }

        session_controller.wait(timeout=2.0)
        if session_controller.is_running():
            logger.info("Session stop still in progress. session_id=%s", result.get("session_id"))
            message = "session stop requested"
        else:
            logger.info("Session stopped. session_id=%s", result.get("session_id"))
            message = "session stopped"

        return {
            "success": True,
            "session_id": result.get("session_id"),
            "message": message,
            "stopped_at": runtime_state.snapshot().get("last_updated_at"),
        }

    @app.on_event("startup")
    def on_startup() -> None:
        logger.info("Status server started.")
        runtime_state.update_component("server", ready=True, running=True, ok=True, last_error=None)
        if wifi_monitor is not None:
            wifi_monitor.start()
        if background_battery_monitor is not None:
            background_battery_monitor.start()

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        logger.info("Status server shutting down.")
        if background_battery_monitor is not None:
            background_battery_monitor.stop()
        if wifi_monitor is not None:
            wifi_monitor.stop()
        runtime_state.update_component("server", running=False, ok=True, last_error=None)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        logger.info("Request received: GET /")
        return HTMLResponse(content=dashboard_html)

    @app.get("/status")
    def get_status() -> dict[str, Any]:
        logger.info("Request received: GET /status")
        snapshot = runtime_state.snapshot()
        latest_session = find_latest_session_dir(project_root)
        latest_session_id = latest_session.name if latest_session is not None else snapshot["active_session_id"]

        metadata: dict[str, Any]
        if latest_session is None:
            metadata = {"status": "no_session", "message": "no session directory found"}
        else:
            metadata = load_json_metadata(latest_session / "meta" / "session_metadata.json", logger)
            if metadata is None:
                metadata = {"status": "no_file", "message": "session metadata not found"}

        return {
            "session_active": snapshot["session_running"],
            "active_session_id": snapshot["active_session_id"],
            "latest_session_id": latest_session_id,
            "started_at": snapshot["session_started_at"],
            "metadata": metadata,
        }

    @app.get("/battery")
    def get_battery() -> dict[str, Any]:
        logger.info("Request received: GET /battery")
        latest_session = find_latest_session_dir(project_root)
        if latest_session is not None:
            battery_row = read_latest_battery_row(latest_session / "battery" / "battery_log.csv", logger)
            if battery_row.get("status") not in {"no_file", "no_data"}:
                classification = classify_battery_status(
                    battery_row.get("voltage"),
                    battery_row.get("current"),
                    battery_row.get("battery_temp_c"),
                    _recent_battery_rows(latest_session / "battery" / "battery_log.csv", logger),
                )
                return {**battery_row, **classification}

        latest_battery = runtime_state.snapshot().get("latest_battery", {})
        if latest_battery.get("unix_time") is None:
            return {"status": "no_data", "message": "no battery data available"}

        battery_payload = {
            "voltage": latest_battery.get("voltage_v"),
            "current": latest_battery.get("current_a"),
            "percent": latest_battery.get("remaining_percent"),
            "battery_temp_c": latest_battery.get("battery_temp_c"),
            "last_updated": latest_battery.get("unix_time"),
        }
        classification = classify_battery_status(
            battery_payload.get("voltage"),
            battery_payload.get("current"),
            battery_payload.get("battery_temp_c"),
            runtime_state.battery_history(),
        )
        return {**battery_payload, **classification}

    @app.get("/sonar")
    def get_sonar() -> dict[str, Any]:
        logger.info("Request received: GET /sonar")
        latest_session = find_latest_session_dir(project_root)
        if latest_session is None:
            return {
                "distance_mm": None,
                "distance_m": None,
                "confidence": None,
                "sample_count_used": 0,
                "variation_mm": None,
                "stable": False,
                "status": "no_data",
                "last_updated": None,
            }

        return read_sonar_status(latest_session / "sonar" / "sonar_log.csv", logger)

    @app.get("/system")
    def get_system() -> dict[str, Any]:
        logger.info("Request received: GET /system")
        return read_system_status(logger)

    @app.get("/health")
    def get_health() -> dict[str, Any]:
        logger.info("Request received: GET /health")
        snapshot = runtime_state.snapshot()
        disk_usage = shutil.disk_usage(runtime_state.data_root)
        wifi_status = None
        if snapshot["network_connected"] is not None or snapshot["network_ssid"] is not None:
            wifi_status = {
                "connected": snapshot["network_connected"],
                "label": snapshot["network_ssid"],
                "ok": snapshot["network"]["ok"],
            }

        return {
            "session_active": snapshot["session_running"],
            "active_session_id": snapshot["active_session_id"],
            "camera_running": snapshot["camera"]["running"],
            "sonar_running": snapshot["sonar"]["running"],
            "battery_running": snapshot["battery"]["running"],
            "disk_free_gb": round(disk_usage.free / (1024 ** 3), 2),
            "wifi_status": wifi_status,
            "last_error": resolve_last_error(snapshot),
            "power_warning": snapshot.get("power_warning"),
            "started_at": snapshot["session_started_at"],
        }

    @app.get("/session/current")
    def get_current_session() -> dict[str, Any]:
        logger.info("Request received: GET /session/current")
        snapshot = runtime_state.snapshot()
        return {
            "session_active": snapshot["session_running"],
            "active_session_id": snapshot["active_session_id"],
            "started_at": snapshot["session_started_at"],
            "stop_requested": snapshot["session_stop_requested"],
            "camera": snapshot["camera"],
            "sonar": snapshot["sonar"],
            "battery": snapshot["battery"],
        }

    @app.post("/session/start")
    def start_session(payload: SessionStartRequest | None = None) -> dict[str, Any]:
        logger.info("Request received: POST /session/start")
        session_name = payload.session_name if payload is not None else None
        return start_session_response(session_name=session_name, request_method="POST")

    @app.get("/session/start")
    def start_session_get(session_name: str | None = None) -> dict[str, Any]:
        logger.info("Request received: GET /session/start")
        return start_session_response(session_name=session_name, request_method="GET")

    @app.post("/session/stop")
    def stop_session() -> dict[str, Any]:
        logger.info("Request received: POST /session/stop")
        return stop_session_response(request_method="POST")

    @app.get("/session/stop")
    def stop_session_get() -> dict[str, Any]:
        logger.info("Request received: GET /session/stop")
        return stop_session_response(request_method="GET")

    return app
