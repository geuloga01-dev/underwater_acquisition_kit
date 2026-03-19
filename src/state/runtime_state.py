from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import shutil
import threading
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ComponentState:
    ready: bool = False
    running: bool = False
    ok: bool = False
    last_error: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class BatteryState:
    timestamp_iso: str | None = None
    unix_time: float | None = None
    voltage_v: float | None = None
    current_a: float | None = None
    remaining_percent: float | None = None
    battery_temp_c: float | None = None
    low_warning: bool = False


class RuntimeState:
    def __init__(self, data_root: Path) -> None:
        self._lock = threading.Lock()
        self.data_root = data_root
        self.active_session_id: str | None = None
        self.session_running = False
        self.session_stop_requested = False
        self.session_started_at: str | None = None
        self.camera = ComponentState()
        self.sonar = ComponentState()
        self.battery = ComponentState()
        self.network = ComponentState()
        self.server = ComponentState()
        self.latest_battery = BatteryState()
        self._battery_history: deque[dict[str, float | None]] = deque(maxlen=10)
        self.network_connected: bool | None = None
        self.network_ssid: str | None = None
        self.power_warning: str | None = None
        self.last_updated_at = _now_iso()

    def set_session(self, session_id: str | None, running: bool, stop_requested: bool = False) -> None:
        with self._lock:
            starting_new_session = running and session_id is not None and session_id != self.active_session_id
            self.active_session_id = session_id
            self.session_running = running
            self.session_stop_requested = stop_requested
            if starting_new_session:
                self.session_started_at = _now_iso()
            elif not running:
                self.session_started_at = None
            self.last_updated_at = _now_iso()

    def clear_session_runtime(self) -> None:
        with self._lock:
            self.active_session_id = None
            self.session_running = False
            self.session_stop_requested = False
            self.session_started_at = None

            for component in (self.camera, self.sonar, self.battery):
                component.running = False
                component.updated_at = _now_iso()

            self.last_updated_at = _now_iso()

    def set_power_warning(self, warning: str | None) -> None:
        with self._lock:
            self.power_warning = warning
            self.last_updated_at = _now_iso()

    def update_component(
        self,
        component_name: str,
        *,
        ready: bool | None = None,
        running: bool | None = None,
        ok: bool | None = None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            component = getattr(self, component_name)
            if ready is not None:
                component.ready = ready
            if running is not None:
                component.running = running
            if ok is not None:
                component.ok = ok
            if last_error is not None:
                component.last_error = last_error
            component.updated_at = _now_iso()
            self.last_updated_at = component.updated_at or _now_iso()

    def set_battery_state(
        self,
        *,
        timestamp_iso: str,
        unix_time: float,
        voltage_v: float | None,
        current_a: float | None,
        remaining_percent: float | None,
        battery_temp_c: float | None,
        low_warning: bool,
    ) -> None:
        with self._lock:
            self.latest_battery = BatteryState(
                timestamp_iso=timestamp_iso,
                unix_time=unix_time,
                voltage_v=voltage_v,
                current_a=current_a,
                remaining_percent=remaining_percent,
                battery_temp_c=battery_temp_c,
                low_warning=low_warning,
            )
            self._battery_history.append(
                {
                    "unix_time": unix_time,
                    "voltage_v": voltage_v,
                    "current_a": current_a,
                    "remaining_percent": remaining_percent,
                    "battery_temp_c": battery_temp_c,
                }
            )
            self.last_updated_at = _now_iso()

    def battery_history(self) -> list[dict[str, float | None]]:
        with self._lock:
            return list(self._battery_history)

    def set_network_status(self, connected: bool | None, ssid: str | None, last_error: str | None = None) -> None:
        with self._lock:
            self.network_connected = connected
            self.network_ssid = ssid
            self.network.ready = True
            self.network.running = True
            self.network.ok = bool(connected)
            self.network.last_error = last_error
            self.network.updated_at = _now_iso()
            self.last_updated_at = self.network.updated_at or _now_iso()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_session_id": self.active_session_id,
                "session_running": self.session_running,
                "session_stop_requested": self.session_stop_requested,
                "session_started_at": self.session_started_at,
                "camera": asdict(self.camera),
                "sonar": asdict(self.sonar),
                "battery": asdict(self.battery),
                "network": asdict(self.network),
                "server": asdict(self.server),
                "latest_battery": asdict(self.latest_battery),
                "network_connected": self.network_connected,
                "network_ssid": self.network_ssid,
                "power_warning": self.power_warning,
                "last_updated_at": self.last_updated_at,
            }

    def health_snapshot(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        disk_usage = shutil.disk_usage(self.data_root)
        return {
            "battery_link": "ok" if snapshot["battery"]["ok"] else "fail",
            "camera": snapshot["camera"],
            "sonar": snapshot["sonar"],
            "network": {
                "connected": snapshot["network_connected"],
                "ssid": snapshot["network_ssid"],
                "ok": snapshot["network"]["ok"],
            },
            "active_session_id": snapshot["active_session_id"],
            "disk_free_bytes": disk_usage.free,
        }
