from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.camera.recording import FrameTimestampWriter, VideoRecorder, load_recording_config
from src.camera.webcam import WebcamCapture, load_camera_config
from src.sonar.ping_logger import PingSonarClient, SonarRecord, load_sonar_config, log_sonar_stream, prepare_sonar
from src.telemetry.gps_listener import GpsListener, load_gps_config, run_gps_logging_loop
from src.state.runtime_state import RuntimeState
from src.system.power_manager import JetsonPowerManager
from src.telemetry.battery_listener import append_battery_csv, load_battery_config, normalize_battery_message
from src.telemetry.attitude_listener import append_attitude_csv, load_attitude_config, normalize_attitude_message
from src.utils.logger import get_app_logger
from src.utils.session import create_session_dirs, save_metadata, session_paths_to_dict


_GRID_COLOR = (90, 180, 255)
_CENTER_COLOR = (0, 0, 255)
_ECHO_CENTER_COLOR = (0, 255, 255)
_ECHO_RASTER_COLOR = (80, 255, 80)
_ECHO_CENTER_REF_WIDTH = 1920.0
_ECHO_CENTER_REF_HEIGHT = 1080.0
_ECHO_CENTER_REF_X = 960.0
_ECHO_CENTER_REF_Y = 430.0


@dataclass(slots=True)
class PreviewSonarState:
    record: SonarRecord | None = None
    error: str | None = None
    last_update_monotonic: float | None = None


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


def draw_alignment_guides(frame):
    guided = frame.copy()
    height, width = guided.shape[:2]
    center_x = width // 2
    center_y = height // 2

    third_x = width // 3
    third_y = height // 3
    for x in (third_x, 2 * third_x):
        cv2.line(guided, (x, 0), (x, height), _GRID_COLOR, 1, cv2.LINE_AA)
    for y in (third_y, 2 * third_y):
        cv2.line(guided, (0, y), (width, y), _GRID_COLOR, 1, cv2.LINE_AA)

    cross_half = max(16, min(width, height) // 20)
    cv2.line(guided, (center_x - cross_half, center_y), (center_x + cross_half, center_y), _CENTER_COLOR, 2, cv2.LINE_AA)
    cv2.line(guided, (center_x, center_y - cross_half), (center_x, center_y + cross_half), _CENTER_COLOR, 2, cv2.LINE_AA)
    cv2.circle(guided, (center_x, center_y), 5, _CENTER_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        guided,
        "center",
        (center_x + 10, center_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _CENTER_COLOR,
        1,
        cv2.LINE_AA,
    )

    # Echo-center candidate estimated from seekingcenter session, scaled to current preview size.
    echo_x = int(round((_ECHO_CENTER_REF_X / _ECHO_CENTER_REF_WIDTH) * width))
    echo_y = int(round((_ECHO_CENTER_REF_Y / _ECHO_CENTER_REF_HEIGHT) * height))
    raster_step = max(22, min(width, height) // 24)

    for dx in (-raster_step, 0, raster_step):
        for dy in (-raster_step, 0, raster_step):
            px = echo_x + dx
            py = echo_y + dy
            color = _ECHO_CENTER_COLOR if (dx == 0 and dy == 0) else _ECHO_RASTER_COLOR
            radius = 7 if (dx == 0 and dy == 0) else 4
            thickness = 2 if (dx == 0 and dy == 0) else 1
            cv2.circle(guided, (px, py), radius, color, thickness, cv2.LINE_AA)

    cv2.line(guided, (echo_x - 14, echo_y), (echo_x + 14, echo_y), _ECHO_CENTER_COLOR, 2, cv2.LINE_AA)
    cv2.line(guided, (echo_x, echo_y - 14), (echo_x, echo_y + 14), _ECHO_CENTER_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        guided,
        "echo-center candidate",
        (echo_x + 12, echo_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _ECHO_CENTER_COLOR,
        1,
        cv2.LINE_AA,
    )
    return guided


def _build_profile_peaks(record: SonarRecord) -> list[dict[str, float]]:
    if not record.profile_data or record.scan_start_mm is None or record.scan_length_mm is None:
        return []
    values = [float(value) for value in record.profile_data]
    peaks: list[dict[str, float]] = []
    step_mm = float(record.scan_length_mm) / max(len(values), 1)
    for index in range(1, len(values) - 1):
        current = values[index]
        if current >= values[index - 1] and current >= values[index + 1]:
            peaks.append(
                {
                    "index": index,
                    "value": current,
                    "distance_mm": float(record.scan_start_mm) + (index + 0.5) * step_mm,
                }
            )
    peaks.sort(key=lambda item: item["value"], reverse=True)
    return peaks[:3]


def draw_profile_graph(canvas: Any, record: SonarRecord, origin_x: int, origin_y: int, width: int, height: int) -> None:
    cv2.rectangle(canvas, (origin_x, origin_y), (origin_x + width, origin_y + height), (55, 55, 55), 1)
    if not record.profile_data:
        cv2.putText(
            canvas,
            "profile unavailable",
            (origin_x + 8, origin_y + height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return

    values = [float(value) for value in record.profile_data]
    max_value = max(max(values), 1.0)
    step_x = width / max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = int(origin_x + index * step_x)
        y = int(origin_y + height - (value / max_value) * (height - 4)) - 2
        points.append((x, y))
    polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [polyline], False, (0, 220, 255), 1)

    for peak in _build_profile_peaks(record):
        peak_x = int(origin_x + peak["index"] * step_x)
        peak_y = int(origin_y + height - (peak["value"] / max_value) * (height - 4)) - 2
        cv2.circle(canvas, (peak_x, peak_y), 3, (0, 0, 255), -1)


def draw_sonar_overlay(frame: Any, state: PreviewSonarState) -> Any:
    panel_height = 190
    canvas = cv2.copyMakeBorder(frame, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    text_x = 16
    text_y = frame.shape[0] + 28
    line_height = 24

    if state.error is not None:
        cv2.putText(canvas, f"sonar error: {state.error}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        return canvas

    if state.record is None:
        cv2.putText(canvas, "waiting for sonar...", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    record = state.record
    age_text = "n/a" if state.last_update_monotonic is None else f"{time.monotonic() - state.last_update_monotonic:.2f}s"
    header_lines = [
        f"distance: {record.distance_mm} mm   confidence: {record.confidence}   ping: {record.ping_number}",
        f"scan_start: {record.scan_start_mm} mm   scan_length: {record.scan_length_mm} mm   gain: {record.gain_setting}",
        f"profile_len: {len(record.profile_data) if record.profile_data else 0}   last_update_age: {age_text}",
    ]
    for idx, line in enumerate(header_lines):
        cv2.putText(
            canvas,
            line,
            (text_x, text_y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    peaks = _build_profile_peaks(record)
    peak_lines = ["profile: no peaks"] if not peaks else [
        f"peak{idx}: {peak['distance_mm']:.0f} mm idx={int(peak['index'])} val={peak['value']:.0f}"
        for idx, peak in enumerate(peaks, start=1)
    ]
    for idx, line in enumerate(peak_lines, start=len(header_lines)):
        cv2.putText(
            canvas,
            line,
            (text_x, text_y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 120),
            1,
            cv2.LINE_AA,
        )

    graph_x = frame.shape[1] // 2
    graph_y = frame.shape[0] + 12
    graph_w = frame.shape[1] // 2 - 24
    graph_h = panel_height - 24
    draw_profile_graph(canvas, record, graph_x, graph_y, graph_w, graph_h)
    return canvas


def _estimate_effective_fps(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return None
    return (len(timestamps) - 1) / elapsed


def _resolve_writer_fps(
    *,
    requested_fps: int | None,
    reported_fps: float | None,
    measured_fps: float | None,
    logger: logging.Logger,
) -> float:
    fallback_fps = float(reported_fps or requested_fps or 30.0)
    if measured_fps is None or measured_fps < 1.0:
        return fallback_fps

    if reported_fps is None:
        logger.info(
            "Camera writer fps resolved from measured capture throughput. measured_fps=%.2f",
            measured_fps,
        )
        return measured_fps

    if abs(measured_fps - reported_fps) / max(reported_fps, 1.0) >= 0.10:
        logger.warning(
            "Camera writer fps adjusted to measured throughput. reported_fps=%.2f measured_fps=%.2f",
            reported_fps,
            measured_fps,
        )
        return measured_fps

    return fallback_fps


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
        self._preview_sonar_state = PreviewSonarState()
        self._preview_sonar_lock = threading.Lock()

    def _is_session_really_running_locked(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _cleanup_stale_session_locked(self) -> bool:
        if self._thread is None and self._stop_event is None and self._session_id is None:
            return False
        if self._is_session_really_running_locked():
            return False

        stale_session_id = self._session_id
        self._thread = None
        self._stop_event = None
        self._session_id = None
        self.runtime_state.clear_session_runtime()
        logging.getLogger(__name__).warning(
            "Stale session state detected and cleared. session_id=%s",
            stale_session_id,
        )
        return True

    def start_session(self, session_name: str | None = None) -> dict[str, Any]:
        with self._lock:
            stale_cleared = self._cleanup_stale_session_locked()
            if self._is_session_really_running_locked():
                return {"ok": False, "message": "session already running", "session_id": self._session_id}
            if stale_cleared:
                logging.getLogger(__name__).info("New session start allowed after stale cleanup.")

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
            stale_cleared = self._cleanup_stale_session_locked()
            if stale_cleared:
                logging.getLogger(__name__).info("Stop requested while session was already stale/stopped.")
            if not self._is_session_really_running_locked() or self._stop_event is None:
                self.runtime_state.clear_session_runtime()
                return {"ok": False, "message": "no active session", "session_id": self._session_id}

            self.runtime_state.set_session(self._session_id, running=True, stop_requested=True)
            self._stop_event.set()
            return {"ok": True, "message": "stop requested", "session_id": self._session_id}

    def is_running(self) -> bool:
        with self._lock:
            self._cleanup_stale_session_locked()
            return self._is_session_really_running_locked()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            self._cleanup_stale_session_locked()

    def _run_session(self, session_paths, stop_event: threading.Event) -> None:
        logger = get_app_logger(
            f"run_session_{session_paths.session_id}",
            session_paths.logs,
            log_filename="run_session.log",
        )
        gps_thread: threading.Thread | None = None
        pixhawk_thread: threading.Thread | None = None
        sonar_thread: threading.Thread | None = None
        gps_errors: list[BaseException] = []
        pixhawk_errors: list[BaseException] = []
        sonar_errors: list[BaseException] = []
        gps_ready = threading.Event()
        pixhawk_ready = threading.Event()
        sonar_ready = threading.Event()

        try:
            with self._preview_sonar_lock:
                self._preview_sonar_state = PreviewSonarState()
            camera_raw = load_yaml_config(self.project_root / "configs" / "camera.yaml")
            sonar_raw = load_yaml_config(self.project_root / "configs" / "sonar.yaml")
            gps_raw = load_yaml_config(self.project_root / "configs" / "gps.yaml")
            battery_raw = load_yaml_config(self.project_root / "configs" / "battery.yaml")
            imu_raw = load_yaml_config(self.project_root / "configs" / "imu.yaml")

            level = resolve_log_level(camera_raw, sonar_raw, gps_raw, battery_raw, imu_raw)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

            camera_config = load_camera_config(camera_raw)
            sonar_config = load_sonar_config(sonar_raw)
            gps_config = load_gps_config(gps_raw)
            battery_config = load_battery_config(battery_raw)
            attitude_config = load_attitude_config(imu_raw)
            recording_config = load_recording_config(camera_raw)
            preview_enabled, preview_source = resolve_preview_setting(camera_raw, camera_config, recording_config)

            if self.power_manager is not None:
                power_ok = self.power_manager.set_mode("recording")
                if not power_ok:
                    self.runtime_state.set_power_warning(self.power_manager.last_warning)
                    logger.warning("Power manager optimization skipped: %s", self.power_manager.last_warning)
                else:
                    self.runtime_state.set_power_warning(None)

            self.runtime_state.update_component("camera", ready=False, running=False, ok=False, last_error=None)
            self.runtime_state.update_component("sonar", ready=False, running=False, ok=False, last_error=None)
            self.runtime_state.update_component("gps", ready=False, running=False, ok=False, last_error=None)
            self.runtime_state.update_component("battery", ready=False, running=False, ok=False, last_error=None)
            self.runtime_state.update_component("imu", ready=False, running=False, ok=False, last_error=None)

            video_path = session_paths.video / f"camera_record.{recording_config.container}"
            sonar_csv_path = session_paths.sonar / "sonar_log.csv"
            sonar_profile_path = session_paths.sonar / "sonar_profile.jsonl"
            gps_csv_path = session_paths.gps / "gps_log.csv"
            battery_csv_path = session_paths.battery / "battery_log.csv"
            attitude_csv_path = session_paths.imu / "attitude_log.csv"
            session_start_time = time.time()
            active_camera = False
            active_sonar = False
            active_gps = False
            active_battery = False
            active_imu = False

            logger.info("Session created: %s", session_paths.root)
            logger.info("Preview resolved from %s: %s", preview_source, preview_enabled)
            logger.info("Acquisition is network-independent. Local recording continues without remote connectivity.")
            logger.info("Sonar port prepared: %s", sonar_config.port)
            logger.info("GPS prepared: enabled=%s port=%s", gps_config.enabled, gps_config.port)
            logger.info("Battery port prepared: %s", battery_config.port)
            logger.info("ATTITUDE logging prepared via shared Pixhawk MAVLink connection.")

            battery_imu_conflict = gps_config.enabled and battery_config.port == gps_config.port
            if battery_imu_conflict:
                logger.warning(
                    "Battery/IMU logging disabled for this session because it shares the GPS serial port: %s",
                    gps_config.port,
                )
                self.runtime_state.update_component(
                    "battery",
                    ready=False,
                    running=False,
                    ok=False,
                    last_error="disabled: port conflicts with gps",
                )
                self.runtime_state.update_component(
                    "imu",
                    ready=False,
                    running=False,
                    ok=False,
                    last_error="disabled: port conflicts with gps",
                )

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

            if gps_config.enabled:
                gps_thread = threading.Thread(
                    target=self._run_gps_worker,
                    args=(gps_raw, gps_csv_path, logger, stop_event, gps_ready, gps_errors),
                    name="gps-worker",
                    daemon=True,
                )
                gps_thread.start()

                for _ in range(20):
                    if gps_ready.is_set():
                        active_gps = True
                        break
                    if gps_errors:
                        logger.warning(
                            "GPS unavailable for this session. Continuing without gps logging: %s",
                            gps_errors[0],
                        )
                        self.runtime_state.update_component(
                            "gps",
                            ready=False,
                            running=False,
                            ok=False,
                            last_error="unavailable",
                        )
                        break
                    if gps_thread is None or not gps_thread.is_alive():
                        break
                    time.sleep(0.1)
            else:
                logger.info("GPS logging disabled in configs/gps.yaml.")

            if not battery_imu_conflict:
                pixhawk_thread = threading.Thread(
                    target=self._run_pixhawk_worker,
                    args=(battery_raw, imu_raw, battery_csv_path, attitude_csv_path, logger, stop_event, pixhawk_ready, pixhawk_errors),
                    name="pixhawk-worker",
                    daemon=True,
                )
                pixhawk_thread.start()
            else:
                logger.info("Skipping Pixhawk worker startup due to GPS port conflict.")

            if active_sonar:
                sonar_thread = threading.Thread(
                    target=self._run_sonar_worker,
                    args=(sonar_raw, sonar_csv_path, sonar_profile_path, logger, stop_event, sonar_ready, sonar_errors),
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

            if not battery_imu_conflict:
                for _ in range(20):
                    if pixhawk_ready.is_set():
                        active_battery = True
                        active_imu = True
                        break
                    if pixhawk_errors:
                        logger.warning(
                            "Pixhawk telemetry unavailable for this session. Continuing without battery/imu logging: %s",
                            pixhawk_errors[0],
                        )
                        active_battery = False
                        active_imu = False
                        self.runtime_state.update_component(
                            "battery",
                            ready=False,
                            running=False,
                            ok=False,
                            last_error="unavailable",
                        )
                        self.runtime_state.update_component(
                            "imu",
                            ready=False,
                            running=False,
                            ok=False,
                            last_error="unavailable",
                        )
                        break
                    if pixhawk_thread is None or not pixhawk_thread.is_alive():
                        break
                    time.sleep(0.1)

            camera_result = self._run_camera_loop(camera_raw, video_path, logger, stop_event, preview_enabled)
            active_camera = bool(camera_result["opened"])

            active_sensors = ["camera"]
            if active_sonar:
                active_sensors.append("sonar")
            if active_gps:
                active_sensors.append("gps")
            if active_battery:
                active_sensors.append("battery")
            if active_imu:
                active_sensors.append("imu")
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
                    "gps": gps_raw.get("gps", {}),
                    "battery": battery_raw.get("battery", {}),
                    "imu": imu_raw.get("imu", {}),
                    "camera_intrinsics": camera_raw.get("calibration", {}).get("intrinsics", {}),
                    "sonar_beam_angle_deg": sonar_raw.get("sonar", {}).get("beam_angle_deg"),
                    "camera_sonar_relative_pose": sonar_raw.get("sonar", {}).get("camera_sonar_relative_pose", {}),
                    "active_subsystems": {
                        "camera": active_camera,
                        "sonar": active_sonar,
                        "gps": active_gps,
                        "battery": active_battery,
                        "imu": active_imu,
                    },
                    "file_paths": {
                        "video": str(video_path),
                        "frame_timestamps": str(session_paths.timestamps / "frame_timestamps.csv"),
                        "sonar_csv": str(sonar_csv_path),
                        "sonar_profile_jsonl": str(sonar_profile_path),
                        "gps_csv": str(gps_csv_path),
                        "battery_csv": str(battery_csv_path),
                        "attitude_csv": str(attitude_csv_path),
                        "log": str(session_paths.logs / "run_session.log"),
                    },
                    "camera_runtime": camera_result,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            logger.info("Metadata saved: %s", metadata_path)

            stop_event.set()
            if gps_thread is not None:
                gps_thread.join(timeout=5.0)
            if sonar_thread is not None:
                sonar_thread.join(timeout=5.0)
            if pixhawk_thread is not None:
                pixhawk_thread.join(timeout=5.0)

            if gps_errors:
                logger.warning("GPS logging ended with error but acquisition continued: %s", gps_errors[0])
            if pixhawk_errors:
                logger.warning("Pixhawk telemetry logging ended with error but acquisition continued: %s", pixhawk_errors[0])
            if sonar_errors:
                logger.warning("Sonar logging ended with error but acquisition continued: %s", sonar_errors[0])
        except Exception as exc:
            logger.exception("Session failed: %s", exc)
            self.runtime_state.update_component("camera", ok=False, last_error=str(exc))
        finally:
            self.runtime_state.update_component("camera", running=False, ready=self.runtime_state.camera.ready, ok=self.runtime_state.camera.ok)
            self.runtime_state.update_component("sonar", running=False, ready=self.runtime_state.sonar.ready, ok=self.runtime_state.sonar.ok)
            self.runtime_state.update_component("gps", running=False, ready=self.runtime_state.gps.ready, ok=self.runtime_state.gps.ok)
            self.runtime_state.update_component("battery", running=False, ready=self.runtime_state.battery.ready, ok=self.runtime_state.battery.ok)
            self.runtime_state.update_component("imu", running=False, ready=self.runtime_state.imu.ready, ok=self.runtime_state.imu.ok)
            self.runtime_state.set_session(None, running=False, stop_requested=False)
            with self._lock:
                self._thread = None
                self._stop_event = None
                self._session_id = None
            logger.info("Runtime state cleared after stop.")
            if self.power_manager is not None:
                power_ok = self.power_manager.set_mode("idle")
                if not power_ok:
                    self.runtime_state.set_power_warning(self.power_manager.last_warning)
                    logger.warning("Power manager idle restore skipped: %s", self.power_manager.last_warning)

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
        profile_path: Path,
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
            self._update_preview_sonar_record(first_record)
            self.runtime_state.update_component("sonar", ready=True, running=True, ok=True, last_error=None)
            logger.info(
                "Sonar worker ready. first_distance_mm=%s first_confidence=%s profile_logging_enabled=%s profile_path=%s",
                first_record.distance_mm,
                first_record.confidence,
                bool(sonar_config.profile_save and sonar_config.profile_read_enabled),
                profile_path,
            )
            ready_event.set()
            sample_count = log_sonar_stream(
                client,
                sonar_config,
                logger,
                csv_path=csv_path,
                profile_path=profile_path,
                stop_event=stop_event,
                on_record=self._update_preview_sonar_record,
            )
            logger.info(
                "Sonar worker stopped. samples=%d csv_output=%s profile_output=%s",
                sample_count,
                csv_path,
                profile_path,
            )
        except Exception as exc:
            error_list.append(exc)
            self._set_preview_sonar_error(str(exc))
            self.runtime_state.update_component("sonar", running=False, ok=False, last_error=str(exc))
            logger.exception("Sonar worker failed: %s", exc)
        finally:
            if client is not None:
                client.close()

    def _run_gps_worker(
        self,
        gps_raw: dict[str, Any],
        csv_path: Path,
        logger: logging.Logger,
        stop_event: threading.Event,
        ready_event: threading.Event,
        error_list: list[BaseException],
    ) -> None:
        listener: GpsListener | None = None
        try:
            gps_config = load_gps_config(gps_raw)
            listener = GpsListener(gps_config, logger=logger)
            listener.connect()
            self.runtime_state.update_component("gps", ready=True, running=True, ok=True, last_error=None)
            ready_event.set()
            sample_count = run_gps_logging_loop(
                listener,
                gps_config,
                logger,
                self.runtime_state,
                csv_path if gps_config.csv_save else None,
                stop_event,
            )
            logger.info("GPS worker stopped. samples=%d csv_output=%s", sample_count, csv_path)
        except Exception as exc:
            error_list.append(exc)
            self.runtime_state.update_component("gps", running=False, ok=False, last_error=str(exc))
            logger.exception("GPS worker failed: %s", exc)
        finally:
            if listener is not None:
                listener.close()

    def _run_pixhawk_worker(
        self,
        battery_raw: dict[str, Any],
        imu_raw: dict[str, Any],
        battery_csv_path: Path,
        attitude_csv_path: Path,
        logger: logging.Logger,
        stop_event: threading.Event,
        ready_event: threading.Event,
        error_list: list[BaseException],
    ) -> None:
        connection = None
        battery_samples = 0
        attitude_samples = 0
        try:
            try:
                from pymavlink import mavutil
            except ModuleNotFoundError as exc:
                raise RuntimeError("pymavlink is not installed. Install it on Jetson for Pixhawk battery/imu logging.") from exc

            battery_config = load_battery_config(battery_raw)
            attitude_config = load_attitude_config(imu_raw)
            connection = mavutil.mavlink_connection(battery_config.port, baud=battery_config.baudrate)
            logger.info(
                "Pixhawk MAVLink connection opened for battery+imu. port=%s baudrate=%s",
                battery_config.port,
                battery_config.baudrate,
            )
            if battery_config.wait_heartbeat:
                logger.info("Waiting for MAVLink heartbeat. timeout=%.1fs", battery_config.heartbeat_timeout)
                connection.wait_heartbeat(timeout=battery_config.heartbeat_timeout)
                logger.info("MAVLink heartbeat received.")

            self.runtime_state.update_component("battery", ready=True, running=True, ok=True, last_error=None)
            self.runtime_state.update_component("imu", ready=True, running=True, ok=True, last_error=None)
            ready_event.set()

            timeout_seconds = max(battery_config.poll_interval, attitude_config.timeout_seconds)
            logger.info(
                "Pixhawk telemetry logging loop started. battery_csv=%s attitude_csv=%s",
                battery_csv_path,
                attitude_csv_path,
            )

            while not stop_event.is_set():
                message = connection.recv_match(type=["BATTERY_STATUS", "ATTITUDE"], blocking=True, timeout=timeout_seconds)
                if message is None:
                    continue

                message_type = message.get_type()
                if message_type == "BATTERY_STATUS":
                    record = normalize_battery_message(message)
                    low_warning = (
                        record.remaining_percent is not None
                        and record.remaining_percent <= battery_config.low_remaining_threshold
                    )
                    self.runtime_state.set_battery_state(
                        timestamp_iso=record.timestamp_iso,
                        unix_time=record.unix_time,
                        voltage_v=record.voltage_v,
                        current_a=record.current_a,
                        remaining_percent=record.remaining_percent,
                        battery_temp_c=record.battery_temp_c,
                        low_warning=low_warning,
                    )
                    if battery_config.csv_save:
                        try:
                            append_battery_csv(battery_csv_path, record)
                        except Exception as exc:
                            logger.warning("Battery CSV append failed but acquisition will continue: %s", exc)
                    battery_samples += 1
                elif message_type == "ATTITUDE":
                    record = normalize_attitude_message(message, time.time())
                    self.runtime_state.set_attitude_state(
                        timestamp_iso=record.timestamp_iso,
                        unix_time=record.unix_time,
                        roll=record.roll,
                        pitch=record.pitch,
                        yaw=record.yaw,
                    )
                    if attitude_config.csv_save:
                        try:
                            append_attitude_csv(attitude_csv_path, record)
                        except Exception as exc:
                            logger.warning("Attitude CSV append failed but acquisition will continue: %s", exc)
                    attitude_samples += 1

            logger.info(
                "Pixhawk telemetry worker stopped. battery_samples=%d attitude_samples=%d battery_output=%s attitude_output=%s",
                battery_samples,
                attitude_samples,
                battery_csv_path,
                attitude_csv_path,
            )
        except Exception as exc:
            error_list.append(exc)
            self.runtime_state.update_component("battery", running=False, ok=False, last_error=str(exc))
            self.runtime_state.update_component("imu", running=False, ok=False, last_error=str(exc))
            logger.warning("Pixhawk telemetry worker failed: %s", exc)
        finally:
            if connection is not None:
                close_method = getattr(connection, "close", None)
                if callable(close_method):
                    try:
                        close_method()
                    except Exception:
                        pass

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
        actual_settings: dict[str, Any] = {
            "width": None,
            "height": None,
            "fps": None,
            "pixel_format": None,
        }
        writer_fps: float | None = None
        bootstrap_measured_fps: float | None = None
        try:
            camera_config = load_camera_config(camera_raw)
            recording_config = load_recording_config(camera_raw)
            capture = WebcamCapture(camera_config, logger=logger)

            logger.info("Camera open start. source=%s backend=%s", camera_config.source, camera_config.backend)
            capture.open()
            opened = True
            self.runtime_state.update_component("camera", ready=True, running=True, ok=True, last_error=None)
            actual_settings = capture.describe_actual_settings()
            logger.info(
                "Camera open success. actual_width=%s actual_height=%s actual_fps=%s actual_pixel_format=%s",
                actual_settings["width"],
                actual_settings["height"],
                actual_settings["fps"],
                actual_settings["pixel_format"],
            )

            start_time = time.monotonic()
            bootstrap_frame_target = max(8, min(16, int(camera_config.fps or 30)))
            bootstrap_frames: list[Any] = []
            bootstrap_timestamps: list[float] = []
            logger.info(
                "Camera bootstrap sampling started. target_frames=%d",
                bootstrap_frame_target,
            )
            while len(bootstrap_frames) < bootstrap_frame_target and not stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the camera during bootstrap.")
                bootstrap_frames.append(frame)
                bootstrap_timestamps.append(time.time())

            bootstrap_measured_fps = _estimate_effective_fps(bootstrap_timestamps)
            writer_fps = _resolve_writer_fps(
                requested_fps=camera_config.fps,
                reported_fps=actual_settings["fps"],
                measured_fps=bootstrap_measured_fps,
                logger=logger,
            )
            logger.info(
                "Camera bootstrap sampling finished. sampled_frames=%d measured_fps=%s writer_fps=%.2f",
                len(bootstrap_frames),
                None if bootstrap_measured_fps is None else round(bootstrap_measured_fps, 2),
                writer_fps,
            )

            recorder = VideoRecorder(
                output_path=video_path,
                recording_config=recording_config,
                frame_size=(camera_config.width or 640, camera_config.height or 480),
                fps=writer_fps,
                logger=logger,
            )
            timestamp_writer = FrameTimestampWriter(video_path.parent.parent / "timestamps" / "frame_timestamps.csv")

            logger.info("Camera recording loop starting.")
            if stop_event.is_set():
                logger.warning("Stop was requested before camera entered the recording loop.")

            for frame, frame_timestamp in zip(bootstrap_frames, bootstrap_timestamps):
                recorder.write(frame)
                timestamp_writer.write(frame_count, frame_timestamp)
                logger.debug("Camera frame captured | frame_id=%d timestamp=%.6f", frame_count, frame_timestamp)
                frame_count += 1

            if frame_count > 0:
                recording_started = True
                logger.info("Camera recording started writing frames.")

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

                if preview_enabled:
                    preview_frame = draw_alignment_guides(frame)
                    preview_frame = draw_sonar_overlay(preview_frame, self._snapshot_preview_sonar_state())
                    cv2.imshow(camera_config.window_name, preview_frame)
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
            average_fps = frame_count / elapsed
            logger.info(
                "Camera worker finished. frames=%d elapsed=%.2fs avg_fps=%.2f writer_fps=%.2f",
                frame_count,
                elapsed,
                average_fps,
                writer_fps or 0.0,
            )
            if writer_fps is not None and abs(average_fps - writer_fps) / max(writer_fps, 1.0) >= 0.10:
                logger.warning(
                    "Camera recording speed mismatch detected. writer_fps=%.2f average_capture_fps=%.2f",
                    writer_fps,
                    average_fps,
                )
            return {
                "opened": opened,
                "recording_started": recording_started,
                "frames_written": frame_count,
                "elapsed_seconds": elapsed,
                "requested_width": camera_config.width,
                "requested_height": camera_config.height,
                "requested_fps": camera_config.fps,
                "requested_pixel_format": camera_config.pixel_format,
                "actual_width": actual_settings["width"],
                "actual_height": actual_settings["height"],
                "actual_fps": actual_settings["fps"],
                "actual_pixel_format": actual_settings["pixel_format"],
                "bootstrap_measured_fps": bootstrap_measured_fps,
                "writer_fps": writer_fps,
            }
        except Exception as exc:
            self.runtime_state.update_component("camera", running=False, ok=False, last_error=str(exc))
            raise
        finally:
            if timestamp_writer is not None:
                timestamp_writer.release()
            if recorder is not None:
                recorder.release()
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()

    def _update_preview_sonar_record(self, record: SonarRecord) -> None:
        with self._preview_sonar_lock:
            self._preview_sonar_state.record = record
            self._preview_sonar_state.error = None
            self._preview_sonar_state.last_update_monotonic = time.monotonic()

    def _set_preview_sonar_error(self, message: str) -> None:
        with self._preview_sonar_lock:
            self._preview_sonar_state.error = message

    def _snapshot_preview_sonar_state(self) -> PreviewSonarState:
        with self._preview_sonar_lock:
            return PreviewSonarState(
                record=self._preview_sonar_state.record,
                error=self._preview_sonar_state.error,
                last_update_monotonic=self._preview_sonar_state.last_update_monotonic,
            )
