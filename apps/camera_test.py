from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from src.camera.webcam import WebcamCapture, load_camera_config
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "camera.yaml"
    logger = get_app_logger("camera_test", PROJECT_ROOT / "logs")
    capture: WebcamCapture | None = None

    try:
        raw_config = load_yaml_config(config_path)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        camera_config = load_camera_config(raw_config)
        logger.info("Loaded camera config from %s", config_path)

        capture = WebcamCapture(camera_config, logger=logger)
        capture.open()

        if not camera_config.preview:
            logger.warning("camera.preview is false. camera_test expects preview, so preview will still open.")

        logger.info("Starting preview. Press 'q' to quit.")
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Failed to read a frame from the camera.")
                break

            cv2.imshow(camera_config.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user.")
                break

        return 0
    except FileNotFoundError:
        logger.exception("Camera config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception("Camera test failed: %s", exc)
        return 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        logger.info("Camera resources released.")


if __name__ == "__main__":
    raise SystemExit(main())
