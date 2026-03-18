from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.control.session_controller import SessionController
from src.network.wifi_monitor import WifiMonitor
from src.state.runtime_state import RuntimeState


class SessionStartRequest(BaseModel):
    session_name: Optional[str] = None


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
) -> FastAPI:
    app = FastAPI(title="Underwater Acquisition Kit Status Server")

    @app.on_event("startup")
    def on_startup() -> None:
        logger.info("Status server started.")
        runtime_state.update_component("server", ready=True, running=True, ok=True, last_error=None)
        if wifi_monitor is not None:
            wifi_monitor.start()

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        logger.info("Status server shutting down.")
        if wifi_monitor is not None:
            wifi_monitor.stop()
        runtime_state.update_component("server", running=False, ok=True, last_error=None)

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
        if latest_session is None:
            return {"status": "no_session", "message": "no session directory found"}

        return read_latest_battery_row(latest_session / "battery" / "battery_log.csv", logger)

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
        result = session_controller.start_session(session_name=session_name)
        if result.get("ok"):
            logger.info("Session started. session_id=%s", result.get("session_id"))
        else:
            logger.info("Duplicate session start rejected. session_id=%s", result.get("session_id"))
        return {
            "success": bool(result.get("ok")),
            "session_id": result.get("session_id"),
            "message": result.get("message"),
            "started_at": runtime_state.snapshot().get("session_started_at"),
        }

    @app.post("/session/stop")
    def stop_session() -> dict[str, Any]:
        logger.info("Request received: POST /session/stop")
        result = session_controller.stop_session()
        if result.get("ok"):
            logger.info("Session stop requested. session_id=%s", result.get("session_id"))
        else:
            logger.info("Session stop requested with no active session.")
        return {
            "success": bool(result.get("ok")),
            "session_id": result.get("session_id"),
            "message": result.get("message"),
            "stopped_at": runtime_state.snapshot().get("last_updated_at"),
        }

    return app
