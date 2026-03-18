from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.camera.recording import FrameTimestampWriter, VideoRecorder, load_recording_config
from src.camera.webcam import WebcamCapture, load_camera_config
from src.sonar.ping_logger import PingSonarClient, load_sonar_config, log_sonar_stream, prepare_sonar
from src.state.runtime_state import RuntimeState
from src.system.power_manager import JetsonPowerManager
from src.telemetry.battery_listener import BatteryListener, load_battery_config, run_battery_logging_loop
from src.utils.logger import get_app_logger
from src.utils.session import create_session_dirs, save_metadata, session_paths_to_dict


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(*configs: dict[str, Any]) -> int:
    for raw_config in configs:
        level_name = raw_config.get("logging", {}).get("level")
        if level_name:
            return getattr(logging, str(level_name).upper(), logging.INFO)
    return logging.INFO


def resolve_preview_setting(camera_raw: dict[str, Any], camera_config, recording_config) -> tuple[bool, str]:
    recording_section = camera_raw.get("recording", {})
    if "preview" in recording_section and recording_section.get("preview") not in (None, ""):
        return bool(recording_config.preview), "recording.preview"
    return bool(camera_config.preview), "camera.preview"


class SessionController:
    def __init__(
        self,
        project_root: Path,
        runtime_state: RuntimeState,
        power_manager: JetsonPowerManager | None = None,
    ) -> None:
        self.project_root = project_root
        self.runtime_state = runtime_state
        self.power_manager = power_manager
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._session_id: str | None = None

    def start_session(self, session_name: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "message": "session already running", "session_id": self._session_id}

            session_paths = create_session_dirs(self.project_root / "data", session_name or "underwater_capture")
            self._session_id = session_paths.session_id
            self._stop_event = threading.Event()
            self.runtime_state.set_session(session_paths.session_id, running=True, stop_requested=False)

            self._thread = threading.Thread(
                target=self._run_session,
                name=f"session-{session_paths.session_id}",
                args=(session_paths, self._stop_event),
                daemon=True,
            )
            self._thread.start()
            return {"ok": True, "message": "session started", "session_id": session_paths.session_id}

    def stop_session(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is None or not self._thread.is_alive() or self._stop_event is None:
                return {"ok": False, "message": "no active session", "session_id": self._session_id}

            self.runtime_state.set_session(self._session_id, running=True, stop_requested=True)
            self._stop_event.set()
            return {"ok": True, "message": "stop requested", "session_id": self._session_id}

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _run_session(self, session_paths, stop_event: threading.Event) -> None:
        logger = get_app_logger(
            f"run_session_{session_paths.session_id}",
            session_paths.logs,
            log_filename="run_session.log",
        )
        battery_thread: threading.Thread | None = None
        sonar_thread: threading.Thread | None = None
        battery_errors: list[BaseException] = []
        sonar_errors: list[BaseException] = []
        battery_ready = threading.Event()
        sonar_ready = threading.Event()

        try:
            camera_raw = load_yaml_config(self.project_root / "configs" / "camera.yaml")
            sonar_raw = load_yaml_config(self.project_root / "configs" / "sonar.yaml")
            battery_raw = load_yaml_config(self.project_root / "configs" / "battery.yaml")

            level = resolve_log_level(camera_raw, sonar_raw, battery_raw)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

            camera_config = load_camera_config(camera_raw)
            sonar_config = load_sonar_config(sonar_raw)
            battery_config = load_battery_config(battery_raw)
            recording_config = load_recording_config(camera_raw)
            preview_enabled, preview_source = resolve_preview_setting(camera_raw, camera_config, recording_config)

            if self.power_manager is not None:
                self.power_manager.set_mode("recording")

            self.runtime_state.update_component("camera", ready=False, running=False, ok=False, last_error=None)
            self.runtime_state.update_component("sonar", ready=False, running=False, ok=False, last_error=None)
            self.runtime_state.update_component("battery", ready=False, running=False, ok=False, last_error=None)

            video_path = session_paths.video / f"camera_record.{recording_config.container}"
            sonar_csv_path = session_paths.sonar / "sonar_log.csv"
            battery_csv_path = session_paths.battery / "battery_log.csv"
            session_start_time = time.time()
            active_camera = False
            active_sonar = False
            active_battery = False

            logger.info("Session created: %s", session_paths.root)
            logger.info("Preview resolved from %s: %s", preview_source, preview_enabled)
            logger.info("Acquisition is network-independent. Local recording continues without remote connectivity.")
            logger.info("Sonar port prepared: %s", sonar_config.port)
            logger.info("Battery port prepared: %s", battery_config.port)

            try:
                self._prepare_sonar(sonar_raw, logger)
                active_sonar = True
                self.runtime_state.update_component("sonar", ready=True, running=False, ok=True, last_error=None)
            except Exception as exc:
                logger.warning("Sonar unavailable for this session. Continuing without sonar: %s", exc)
                self.runtime_state.update_component("sonar", ready=False, running=False, ok=False, last_error="unavailable")
                active_sonar = False

            if active_sonar:
                logger.info("Waiting 1.0 second after sonar preparation before camera open.")
                time.sleep(1.0)

            battery_thread = threading.Thread(
                target=self._run_battery_worker,
                args=(battery_raw, battery_csv_path, logger, stop_event, battery_ready, battery_errors),
                name="battery-worker",
                daemon=True,
            )
            battery_thread.start()

            if active_sonar:
                sonar_thread = threading.Thread(
                    target=self._run_sonar_worker,
                    args=(sonar_raw, sonar_csv_path, logger, stop_event, sonar_ready, sonar_errors),
                    name="sonar-worker",
                    daemon=True,
                )
                sonar_thread.start()

                while not sonar_ready.is_set():
                    if sonar_errors:
                        logger.warning(
                            "Sonar worker failed during startup. Continuing without sonar: %s",
                            sonar_errors[0],
                        )
                        active_sonar = False
                        self.runtime_state.update_component(
                            "sonar",
                            ready=False,
                            running=False,
                            ok=False,
                            last_error="unavailable",
                        )
                        break
                    time.sleep(0.1)

            for _ in range(20):
                if battery_ready.is_set():
                    active_battery = True
                    break
                if battery_errors:
                    logger.warning(
                        "Battery listener unavailable for this session. Continuing without battery logging: %s",
                        battery_errors[0],
                    )
                    active_battery = False
                    self.runtime_state.update_component(
                        "battery",
                        ready=False,
                        running=False,
                        ok=False,
                        last_error="unavailable",
                    )
                    break
                if battery_thread is None or not battery_thread.is_alive():
                    break
                time.sleep(0.1)

            camera_result = self._run_camera_loop(camera_raw, video_path, logger, stop_event, preview_enabled)
            active_camera = bool(camera_result["opened"])

            active_sensors = ["camera"]
            if active_sonar:
                active_sensors.append("sonar")
            if active_battery:
                active_sensors.append("battery")
            metadata_path = save_metadata(
                session_paths.meta / "session_metadata.json",
                {
                    "session_id": session_paths.session_id,
                    "start_time": session_start_time,
                    "camera_fps": camera_config.fps,
                    "resolution": f"{camera_config.width}x{camera_config.height}",
                    "sensors": active_sensors,
                    "session": session_paths_to_dict(session_paths),
                    "camera": camera_raw.get("camera", {}),
                    "recording": camera_raw.get("recording", {}),
                    "sonar": sonar_raw.get("sonar", {}),
                    "battery": battery_raw.get("battery", {}),
                    "active_subsystems": {
                        "camera": active_camera,
                        "sonar": active_sonar,
                        "battery": active_battery,
                    },
                    "camera_runtime": camera_result,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            logger.info("Metadata saved: %s", metadata_path)

            stop_event.set()
            if sonar_thread is not None:
                sonar_thread.join(timeout=5.0)
            if battery_thread is not None:
                battery_thread.join(timeout=5.0)

            if battery_errors:
                logger.warning("Battery logging ended with error but acquisition continued: %s", battery_errors[0])
            if sonar_errors:
                logger.warning("Sonar logging ended with error but acquisition continued: %s", sonar_errors[0])
        except Exception as exc:
            logger.exception("Session failed: %s", exc)
            self.runtime_state.update_component("camera", ok=False, last_error=str(exc))
        finally:
            self.runtime_state.update_component("camera", running=False, ready=self.runtime_state.camera.ready, ok=self.runtime_state.camera.ok)
            self.runtime_state.update_component("sonar", running=False, ready=self.runtime_state.sonar.ready, ok=self.runtime_state.sonar.ok)
            self.runtime_state.update_component("battery", running=False, ready=self.runtime_state.battery.ready, ok=self.runtime_state.battery.ok)
            self.runtime_state.set_session(None, running=False, stop_requested=False)
            if self.power_manager is not None:
                self.power_manager.set_mode("idle")

    def _prepare_sonar(self, sonar_raw: dict[str, Any], logger: logging.Logger) -> None:
        logger.info("Starting sonar preparation before camera open.")
        sonar_config = load_sonar_config(sonar_raw)
        client = PingSonarClient(sonar_config, logger=logger)
        try:
            first_record = prepare_sonar(client)
            logger.info(
                "Sonar preparation succeeded. first_distance_mm=%s first_confidence=%s",
                first_record.distance_mm,
                first_record.confidence,
            )
        finally:
            client.close()

    def _run_sonar_worker(
        self,
        sonar_raw: dict[str, Any],
        csv_path: Path,
        logger: logging.Logger,
        stop_event: threading.Event,
        ready_event: threading.Event,
        error_list: list[BaseException],
    ) -> None:
        client: PingSonarClient | None = None
        try:
            sonar_config = load_sonar_config(sonar_raw)
            client = PingSonarClient(sonar_config, logger=logger)
            first_record = prepare_sonar(client)
            self.runtime_state.update_component("sonar", ready=True, running=True, ok=True, last_error=None)
            logger.info(
                "Sonar worker ready. first_distance_mm=%s first_confidence=%s",
                first_record.distance_mm,
                first_record.confidence,
            )
            ready_event.set()
            sample_count = log_sonar_stream(client, sonar_config, logger, csv_path=csv_path, stop_event=stop_event)
            logger.info("Sonar worker stopped. samples=%d output=%s", sample_count, csv_path)
        except Exception as exc:
            error_list.append(exc)
            self.runtime_state.update_component("sonar", running=False, ok=False, last_error=str(exc))
            logger.exception("Sonar worker failed: %s", exc)
        finally:
            if client is not None:
                client.close()

    def _run_battery_worker(
        self,
        battery_raw: dict[str, Any],
        csv_path: Path,
        logger: logging.Logger,
        stop_event: threading.Event,
        ready_event: threading.Event,
        error_list: list[BaseException],
    ) -> None:
        listener: BatteryListener | None = None
        try:
            battery_config = load_battery_config(battery_raw)
            listener = BatteryListener(battery_config, logger=logger)
            listener.connect()
            self.runtime_state.update_component("battery", ready=True, running=True, ok=True, last_error=None)
            ready_event.set()
            sample_count = run_battery_logging_loop(listener, battery_config, logger, self.runtime_state, csv_path, stop_event)
            logger.info("Battery worker stopped. samples=%d output=%s", sample_count, csv_path)
        except Exception as exc:
            error_list.append(exc)
            self.runtime_state.update_component("battery", running=False, ok=False, last_error=str(exc))
            logger.warning("Battery worker failed: %s", exc)
        finally:
            if listener is not None:
                listener.close()

    def _run_camera_loop(
        self,
        camera_raw: dict[str, Any],
        video_path: Path,
        logger: logging.Logger,
        stop_event: threading.Event,
        preview_enabled: bool,
    ) -> dict[str, Any]:
        capture: WebcamCapture | None = None
        recorder: VideoRecorder | None = None
        timestamp_writer: FrameTimestampWriter | None = None
        opened = False
        frame_count = 0
        recording_started = False
        try:
            camera_config = load_camera_config(camera_raw)
            recording_config = load_recording_config(camera_raw)
            capture = WebcamCapture(camera_config, logger=logger)

            logger.info("Camera open start. source=%s backend=%s", camera_config.source, camera_config.backend)
            capture.open()
            opened = True
            self.runtime_state.update_component("camera", ready=True, running=True, ok=True, last_error=None)
            logger.info("Camera open success.")

            recorder = VideoRecorder(
                output_path=video_path,
                recording_config=recording_config,
                frame_size=(camera_config.width or 640, camera_config.height or 480),
                fps=float(camera_config.fps or 30),
                logger=logger,
            )
            timestamp_writer = FrameTimestampWriter(video_path.parent.parent / "timestamps" / "frame_timestamps.csv")

            logger.info("Camera recording loop starting.")
            start_time = time.monotonic()
            if stop_event.is_set():
                logger.warning("Stop was requested before camera entered the recording loop.")
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the camera.")

                frame_timestamp = time.time()
                recorder.write(frame)
                if timestamp_writer is not None:
                    timestamp_writer.write(frame_count, frame_timestamp)
                logger.debug("Camera frame captured | frame_id=%d timestamp=%.6f", frame_count, frame_timestamp)
                frame_count += 1
                if not recording_started:
                    recording_started = True
                    logger.info("Camera recording started writing frames.")

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
            if frame_count == 0:
                logger.warning("Camera session ended before any frames were recorded.")
            logger.info("Camera worker finished. frames=%d elapsed=%.2fs avg_fps=%.2f", frame_count, elapsed, frame_count / elapsed)
            return {
                "opened": opened,
                "recording_started": recording_started,
                "frames_written": frame_count,
                "elapsed_seconds": elapsed,
            }
        except Exception as exc:
            self.runtime_state.update_component("camera", running=False, ok=False, last_error=str(exc))
            raise
        finally:
            if recorder is not None:
                recorder.release()
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()
