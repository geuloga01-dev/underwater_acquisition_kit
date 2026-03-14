from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from src.camera.recording import VideoRecorder, load_recording_config
from src.camera.webcam import WebcamCapture, load_camera_config
from src.sonar.ping_logger import PingSonarClient, load_sonar_config, log_sonar_stream
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


def run_sonar_worker(
    sonar_raw: dict,
    csv_path: Path,
    logger: logging.Logger,
    stop_event: threading.Event,
    ready_event: threading.Event,
    startup_error: list[BaseException],
) -> None:
    client: PingSonarClient | None = None

    try:
        sonar_config = load_sonar_config(sonar_raw)
        client = PingSonarClient(sonar_config, logger=logger)
        client.connect()
        ready_event.set()
        sample_count = log_sonar_stream(
            client,
            sonar_config,
            logger,
            csv_path=csv_path,
            stop_event=stop_event,
        )
        logger.info("Sonar worker stopped. samples=%d output=%s", sample_count, csv_path)
    except Exception as exc:
        startup_error.append(exc)
        stop_event.set()
        logger.exception("Sonar worker failed: %s", exc)
    finally:
        if client is not None:
            client.close()


def perform_sonar_quick_check(sonar_raw: dict, logger: logging.Logger) -> None:
    logger.info("Initialization order: sonar quick check -> sonar worker -> camera open -> concurrent capture")
    logger.info("Running sonar-only quick check before camera initialization.")

    sonar_config = load_sonar_config(sonar_raw)
    client = PingSonarClient(sonar_config, logger=logger)
    try:
        client.connect()
        logger.info("Sonar quick check succeeded.")
    except Exception as exc:
        logger.exception("Sonar quick check failed before session start: %s", exc)
        raise RuntimeError(f"Sonar quick check failed: {exc}") from exc
    finally:
        client.close()


def run_camera_loop(
    camera_raw: dict,
    video_path: Path,
    logger: logging.Logger,
    stop_event: threading.Event,
) -> tuple[int, float]:
    capture: WebcamCapture | None = None
    recorder: VideoRecorder | None = None
    frame_count = 0
    start_time = time.monotonic()

    try:
        camera_config = load_camera_config(camera_raw)
        recording_config = load_recording_config(camera_raw)
        preview_enabled = recording_config.preview or camera_config.preview

        capture = WebcamCapture(camera_config, logger=logger)
        try:
            logger.info("Opening camera after sonar initialization. source=%s backend=%s", camera_config.source, camera_config.backend)
            capture.open()
        except Exception as exc:
            logger.exception("Camera initialization failed: %s", exc)
            raise RuntimeError(f"Camera initialization failed: {exc}") from exc

        recorder = VideoRecorder(
            output_path=video_path,
            recording_config=recording_config,
            frame_size=(camera_config.width or 640, camera_config.height or 480),
            fps=float(camera_config.fps or 30),
            logger=logger,
        )

        logger.info("Camera recording started. output=%s", video_path)

        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera.")

            recorder.write(frame)
            frame_count += 1

            if preview_enabled:
                cv2.imshow(camera_config.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Camera preview stop requested by user.")
                    stop_event.set()
                    break

            elapsed = time.monotonic() - start_time
            if recording_config.duration_seconds > 0 and elapsed >= recording_config.duration_seconds:
                logger.info("Camera recording duration reached: %.2f seconds", elapsed)
                stop_event.set()
                break

        elapsed = max(time.monotonic() - start_time, 0.001)
        return frame_count, elapsed
    finally:
        if recorder is not None:
            recorder.release()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


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
        recording_config = load_recording_config(camera_raw)
        video_path = session_paths.video / "camera_record.mp4"
        sonar_csv_path = session_paths.sonar / "sonar_log.csv"
        stop_event = threading.Event()
        sonar_ready = threading.Event()
        sonar_errors: list[BaseException] = []

        metadata = {
            "session": session_paths_to_dict(session_paths),
            "camera": camera_raw.get("camera", {}),
            "sonar": sonar_raw.get("sonar", {}),
            "recording": camera_raw.get("recording", {}),
            "notes": {
                "purpose": "Concurrent camera and sonar acquisition MVP for Jetson data collection.",
                "why_split": "Camera, sonar, and session helpers stay separate so hardware-specific code can evolve without rewriting the app entry points.",
            },
        }
        metadata_path = save_metadata(session_paths.meta / "session_metadata.json", metadata)

        logger.info("Session created: %s", session_paths.root)
        logger.info("Camera source prepared: %s", camera_config.source)
        logger.info("Sonar port prepared: %s", sonar_config.port)
        logger.info("Metadata saved: %s", metadata_path)
        logger.info("Session video output: %s", video_path)
        logger.info("Session sonar output: %s", sonar_csv_path)
        logger.info(
            "Session starting. preview=%s duration_seconds=%.1f",
            recording_config.preview or camera_config.preview,
            recording_config.duration_seconds,
        )
        logger.info("Preview is disabled by default unless camera.preview or recording.preview is set true.")

        perform_sonar_quick_check(sonar_raw, logger)
        logger.info("Waiting 1.0 second after sonar quick check before starting sonar worker.")
        time.sleep(1.0)

        sonar_thread = threading.Thread(
            target=run_sonar_worker,
            name="sonar-worker",
            args=(sonar_raw, sonar_csv_path, logger, stop_event, sonar_ready, sonar_errors),
            daemon=True,
        )
        sonar_thread.start()

        logger.info("Waiting for sonar worker initialization to complete before opening camera.")
        while not sonar_ready.is_set():
            if sonar_errors:
                logger.error("Sonar worker initialization failed: %s", sonar_errors[0])
                raise RuntimeError(f"Sonar initialization failed: {sonar_errors[0]}")
            if not sonar_thread.is_alive():
                raise RuntimeError("Sonar worker stopped before initialization completed.")
            time.sleep(0.1)

        logger.info("Sonar worker initialization confirmed. Waiting 1.0 second before camera open.")
        time.sleep(1.0)

        frame_count, elapsed = run_camera_loop(camera_raw, video_path, logger, stop_event)
        stop_event.set()
        sonar_thread.join(timeout=5.0)

        if sonar_errors:
            raise RuntimeError(f"Sonar logging failed during session: {sonar_errors[0]}")

        logger.info(
            "Session finished. frames=%d elapsed=%.2fs avg_fps=%.2f",
            frame_count,
            elapsed,
            frame_count / elapsed,
        )
        return 0
    except KeyboardInterrupt:
        bootstrap_logger.info("Session interrupted by user.")
        return 0
    except Exception as exc:
        bootstrap_logger.exception("Session setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
