from __future__ import annotations

import csv
import logging
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_app_logger


logger = get_app_logger("status_server", PROJECT_ROOT / "logs", level=logging.INFO)
app = FastAPI(title="Underwater Acquisition Kit Status Server")
logger.info("Status server initialized.")


def find_latest_session_dir() -> Path | None:
    sessions_dir = PROJECT_ROOT / "data" / "sessions"
    if not sessions_dir.exists():
        return None

    session_dirs = [path for path in sessions_dir.iterdir() if path.is_dir()]
    if not session_dirs:
        return None

    return sorted(session_dirs, key=lambda path: path.name, reverse=True)[0]


def load_json_metadata(metadata_path: Path) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None

    try:
        import json

        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load metadata file %s: %s", metadata_path, exc)
        return None


def read_latest_battery_row(csv_path: Path) -> dict[str, Any]:
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


def _as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


@app.get("/status")
def get_status() -> dict[str, Any]:
    logger.info("Request received: GET /status")
    latest_session = find_latest_session_dir()
    if latest_session is None:
        return {
            "session_id": None,
            "metadata": {"status": "no_session", "message": "no session directory found"},
        }

    metadata = load_json_metadata(latest_session / "meta" / "session_metadata.json")
    if metadata is None:
        metadata = {"status": "no_file", "message": "session metadata not found"}

    return {
        "session_id": latest_session.name,
        "metadata": metadata,
    }


@app.get("/battery")
def get_battery() -> dict[str, Any]:
    logger.info("Request received: GET /battery")
    latest_session = find_latest_session_dir()
    if latest_session is None:
        return {"status": "no_session", "message": "no session directory found"}

    return read_latest_battery_row(latest_session / "battery" / "battery_log.csv")
