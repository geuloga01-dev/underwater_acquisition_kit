from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AttitudeConfig:
    csv_save: bool = True
    csv_path: str | None = None
    timeout_seconds: float = 0.5


@dataclass(slots=True)
class AttitudeRecord:
    timestamp: float
    roll: float | None
    pitch: float | None
    yaw: float | None

    @property
    def unix_time(self) -> float:
        return self.timestamp

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat()


def load_attitude_config(raw_config: dict[str, Any]) -> AttitudeConfig:
    section = raw_config.get("imu", {})
    return AttitudeConfig(
        csv_save=_optional_bool(section.get("csv_save"), default=True),
        csv_path=section.get("csv_path"),
        timeout_seconds=float(section.get("timeout_seconds", 0.5)),
    )


def normalize_attitude_message(message: Any, timestamp: float) -> AttitudeRecord:
    return AttitudeRecord(
        timestamp=timestamp,
        roll=_as_float(getattr(message, "roll", None)),
        pitch=_as_float(getattr(message, "pitch", None)),
        yaw=_as_float(getattr(message, "yaw", None)),
    )


def append_attitude_csv(csv_path: Path, record: AttitudeRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "roll", "pitch", "yaw"])
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": record.timestamp,
                "roll": record.roll,
                "pitch": record.pitch,
                "yaw": record.yaw,
            }
        )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
