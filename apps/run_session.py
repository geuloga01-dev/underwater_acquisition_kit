from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.camera.webcam import load_camera_config
from src.sonar.ping_logger import load_sonar_config
from src.utils.logger import get_app_logger
from src.utils.session import create_session_dirs, save_metadata, session_paths_to_dict


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(*configs: dict) -> int:
    for raw_config in configs:
        level_name = raw_config.get("logging", {}).get("level")
        if level_name:
            return getattr(logging, str(level_name).upper(), logging.INFO)
    return logging.INFO


def main() -> int:
    camera_config_path = PROJECT_ROOT / "configs" / "camera.yaml"
    sonar_config_path = PROJECT_ROOT / "configs" / "sonar.yaml"
    bootstrap_logger = get_app_logger("run_session_bootstrap", PROJECT_ROOT / "logs")

    try:
        camera_raw = load_yaml_config(camera_config_path)
        sonar_raw = load_yaml_config(sonar_config_path)
        session_paths = create_session_dirs(PROJECT_ROOT / "data", session_name="underwater_capture")
        logger = get_app_logger(
            f"run_session_{session_paths.session_id}",
            session_paths.logs,
            level=resolve_log_level(camera_raw, sonar_raw),
            log_filename="run_session.log",
        )

        camera_config = load_camera_config(camera_raw)
        sonar_config = load_sonar_config(sonar_raw)

        metadata = {
            "session": session_paths_to_dict(session_paths),
            "camera": camera_raw.get("camera", {}),
            "sonar": sonar_raw.get("sonar", {}),
            "notes": {
                "purpose": "Session scaffold for future camera recording and sonar logging integration.",
                "why_split": "Camera, sonar, and session helpers stay separate so hardware-specific code can evolve without rewriting the app entry points.",
            },
        }
        metadata_path = save_metadata(session_paths.meta / "session_metadata.json", metadata)

        logger.info("Session created: %s", session_paths.root)
        logger.info("Camera source prepared: %s", camera_config.source)
        logger.info("Sonar port prepared: %s", sonar_config.port)
        logger.info("Metadata saved: %s", metadata_path)
        logger.info("Session video dir: %s", session_paths.video)
        logger.info("Session sonar dir: %s", session_paths.sonar)
        logger.info("Session log dir: %s", session_paths.logs)
        logger.info("This MVP does not start full concurrent capture yet.")
        return 0
    except Exception as exc:
        bootstrap_logger.exception("Session setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
