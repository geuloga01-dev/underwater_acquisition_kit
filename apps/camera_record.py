from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from src.camera.recording import RecordingConfig, VideoRecorder, load_recording_config
from src.camera.webcam import WebcamCapture, load_camera_config
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def build_output_path(recording_config: RecordingConfig) -> Path:
    if recording_config.output_path:
        return Path(recording_config.output_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"camera_record_{timestamp}.{recording_config.container}"
    return PROJECT_ROOT / "data" / filename


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "camera.yaml"
    logger = get_app_logger("camera_record", PROJECT_ROOT / "logs")
    capture: WebcamCapture | None = None
    recorder: VideoRecorder | None = None

    try:
        raw_config = load_yaml_config(config_path)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        camera_config = load_camera_config(raw_config)
        recording_config = load_recording_config(raw_config)
        output_path = build_output_path(recording_config)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loaded camera config from %s", config_path)
        logger.info("Recording target: %s", output_path)

        capture = WebcamCapture(camera_config, logger=logger)
        capture.open()

        recorder = VideoRecorder(
            output_path=output_path,
            recording_config=recording_config,
            frame_size=(camera_config.width or 640, camera_config.height or 480),
            fps=float(camera_config.fps or 30),
            logger=logger,
        )

        start_time = time.monotonic()
        frame_count = 0

        logger.info(
            "Starting recording for %.1f seconds. Press 'q' to stop early.",
            recording_config.duration_seconds,
        )

        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Failed to read a frame from the camera.")
                break

            if recorder is None:
                raise RuntimeError("Recorder was not initialized.")

            recorder.write(frame)
            frame_count += 1

            if recording_config.preview:
                cv2.imshow(camera_config.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Recording stopped early by user.")
                    break

            elapsed = time.monotonic() - start_time
            if elapsed >= recording_config.duration_seconds:
                logger.info("Recording duration reached: %.2f seconds", elapsed)
                break

        elapsed = max(time.monotonic() - start_time, 0.001)
        logger.info(
            "Recording finished. frames=%d elapsed=%.2fs avg_fps=%.2f output=%s",
            frame_count,
            elapsed,
            frame_count / elapsed,
            output_path,
        )
        return 0
    except FileNotFoundError:
        logger.exception("Camera config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception("Camera recording failed: %s", exc)
        return 1
    finally:
        if recorder is not None:
            recorder.release()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        logger.info("Recording resources released.")


if __name__ == "__main__":
    raise SystemExit(main())
