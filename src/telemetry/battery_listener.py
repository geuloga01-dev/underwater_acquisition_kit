from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import logging
from pathlib import Path
import threading
from typing import Any

from src.state.runtime_state import RuntimeState


@dataclass(slots=True)
class BatteryConfig:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    poll_interval: float = 1.0
    csv_save: bool = True
    csv_path: str | None = None
    low_remaining_threshold: float = 20.0
    wait_heartbeat: bool = False
    heartbeat_timeout: float = 5.0


@dataclass(slots=True)
class BatteryRecord:
    timestamp_iso: str
    unix_time: float
    voltage_v: float | None
    current_a: float | None
    remaining_percent: float | None


def load_battery_config(raw_config: dict[str, Any]) -> BatteryConfig:
    section = raw_config.get("battery", {})
    return BatteryConfig(
        port=str(section.get("port", "/dev/ttyACM0")),
        baudrate=int(section.get("baudrate", 115200)),
        poll_interval=float(section.get("poll_interval", 1.0)),
        csv_save=_optional_bool(section.get("csv_save"), default=True),
        csv_path=section.get("csv_path"),
        low_remaining_threshold=float(section.get("low_remaining_threshold", 20.0)),
        wait_heartbeat=_optional_bool(section.get("wait_heartbeat"), default=False),
        heartbeat_timeout=float(section.get("heartbeat_timeout", 5.0)),
    )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class BatteryListener:
    def __init__(self, config: BatteryConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._connection = None

    def connect(self) -> None:
        try:
            from pymavlink import mavutil
        except ModuleNotFoundError as exc:
            raise RuntimeError("pymavlink is not installed. Install it on Jetson for battery logging.") from exc

        self._connection = mavutil.mavlink_connection(self.config.port, baud=self.config.baudrate)
        self.logger.info("Battery MAVLink connection opened. port=%s baudrate=%s", self.config.port, self.config.baudrate)

        if self.config.wait_heartbeat:
            self.logger.info("Waiting for MAVLink heartbeat. timeout=%.1fs", self.config.heartbeat_timeout)
            self._connection.wait_heartbeat(timeout=self.config.heartbeat_timeout)
            self.logger.info("MAVLink heartbeat received.")

    def read_record(self, timeout: float | None = None) -> BatteryRecord | None:
        if self._connection is None:
            raise RuntimeError("Battery listener has not been connected yet.")

        message = self._connection.recv_match(type="BATTERY_STATUS", blocking=True, timeout=timeout)
        if message is None:
            return None

        return normalize_battery_message(message)

    def close(self) -> None:
        if self._connection is not None:
            close_method = getattr(self._connection, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
            self._connection = None


def normalize_battery_message(message: Any) -> BatteryRecord:
    now = datetime.now(timezone.utc)
    voltages = [value for value in getattr(message, "voltages", []) if value and value > 0]
    voltage_v = sum(voltages) / 1000.0 if voltages else None

    current_raw = getattr(message, "current_battery", -1)
    current_a = current_raw / 100.0 if current_raw not in (-1, None) else None

    remaining_raw = getattr(message, "battery_remaining", -1)
    remaining_percent = float(remaining_raw) if remaining_raw not in (-1, None) else None

    return BatteryRecord(
        timestamp_iso=now.isoformat(),
        unix_time=now.timestamp(),
        voltage_v=voltage_v,
        current_a=current_a,
        remaining_percent=remaining_percent,
    )


def append_battery_csv(csv_path: Path, record: BatteryRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp_iso", "unix_time", "voltage_v", "current_a", "remaining_percent"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(record))


def run_battery_logging_loop(
    listener: BatteryListener,
    config: BatteryConfig,
    logger: logging.Logger,
    state: RuntimeState,
    csv_path: Path | None,
    stop_event: threading.Event,
) -> int:
    sample_count = 0
    state.update_component("battery", ready=True, running=True, ok=True, last_error=None)
    logger.info("Battery logging loop started.")

    try:
        while not stop_event.is_set():
            record = listener.read_record(timeout=config.poll_interval)
            if record is None:
                continue

            low_warning = (
                record.remaining_percent is not None
                and record.remaining_percent <= config.low_remaining_threshold
            )
            state.set_battery_state(
                timestamp_iso=record.timestamp_iso,
                unix_time=record.unix_time,
                voltage_v=record.voltage_v,
                current_a=record.current_a,
                remaining_percent=record.remaining_percent,
                low_warning=low_warning,
            )
            if csv_path is not None and config.csv_save:
                append_battery_csv(csv_path, record)

            logger.info(
                "Battery sample voltage_v=%s current_a=%s remaining_percent=%s",
                record.voltage_v,
                record.current_a,
                record.remaining_percent,
            )
            if low_warning:
                logger.warning(
                    "Battery remaining percentage is below threshold: %.1f <= %.1f",
                    record.remaining_percent,
                    config.low_remaining_threshold,
                )
            sample_count += 1
    except Exception as exc:
        state.update_component("battery", running=False, ok=False, last_error=str(exc))
        logger.exception("Battery logging loop failed: %s", exc)
        return sample_count

    state.update_component("battery", running=False, ok=True, last_error=None)
    return sample_count
