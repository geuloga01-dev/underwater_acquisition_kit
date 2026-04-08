from __future__ import annotations

import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from src.analysis.sonar_profile_analysis import extract_peaks, ProfileSample
from src.camera.webcam import WebcamCapture, load_camera_config
from src.sonar.ping_logger import PingSonarClient, SonarRecord, load_sonar_config
from src.utils.logger import get_app_logger

_DISPLAY_MAX_WIDTH = 1280
_DISPLAY_MAX_HEIGHT = 900
_GRID_COLOR = (90, 180, 255)
_CENTER_COLOR = (0, 0, 255)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(*configs: dict[str, Any]) -> int:
    for raw_config in configs:
        level_name = raw_config.get("logging", {}).get("level")
        if level_name:
            return getattr(logging, str(level_name).upper(), logging.INFO)
    return logging.INFO


class SharedSonarState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.record: SonarRecord | None = None
        self.error: str | None = None
        self.last_update_monotonic: float | None = None

    def set_record(self, record: SonarRecord) -> None:
        with self._lock:
            self.record = record
            self.error = None
            self.last_update_monotonic = time.monotonic()

    def set_error(self, message: str) -> None:
        with self._lock:
            self.error = message

    def snapshot(self) -> tuple[SonarRecord | None, str | None, float | None]:
        with self._lock:
            return self.record, self.error, self.last_update_monotonic


def sonar_reader_loop(client: PingSonarClient, state: SharedSonarState, stop_event: threading.Event, logger: logging.Logger) -> None:
    try:
        first_record = client.connect_and_validate()
        state.set_record(first_record)
        logger.info(
            "Live sonar ready. first_distance_mm=%s first_confidence=%s profile=%s",
            first_record.distance_mm,
            first_record.confidence,
            "yes" if first_record.profile_data else "no",
        )
        while not stop_event.is_set():
            record = client.read_record()
            state.set_record(record)
            stop_event.wait(client.config.sample_interval)
    except Exception as exc:
        state.set_error(str(exc))
        logger.exception("Live sonar reader failed: %s", exc)
    finally:
        client.close()


def build_profile_sample(record: SonarRecord) -> ProfileSample | None:
    if not record.profile_data:
        return None
    return ProfileSample(
        line_index=-1,
        timestamp=record.timestamp,
        distance_mm=None if record.distance_mm is None else float(record.distance_mm),
        confidence=None if record.confidence is None else float(record.confidence),
        scan_start_mm=None if record.scan_start_mm is None else float(record.scan_start_mm),
        scan_length_mm=None if record.scan_length_mm is None else float(record.scan_length_mm),
        gain_setting=None if record.gain_setting is None else float(record.gain_setting),
        mode_auto=record.mode_auto,
        transmit_duration_us=None if record.transmit_duration_us is None else float(record.transmit_duration_us),
        ping_number=None if record.ping_number is None else float(record.ping_number),
        profile_data=[float(value) for value in record.profile_data],
    )


def format_peak_text(record: SonarRecord) -> list[str]:
    sample = build_profile_sample(record)
    if sample is None:
        return ["profile: no data"]

    peaks = sorted(extract_peaks(sample), key=lambda peak: peak.value, reverse=True)[:3]
    if not peaks:
        return ["profile: no peaks"]

    lines = []
    for idx, peak in enumerate(peaks, start=1):
        distance_mm = peak.distance_mm
        if distance_mm is None:
            lines.append(f"peak{idx}: idx={peak.index} val={peak.value:.0f}")
        else:
            lines.append(f"peak{idx}: {distance_mm:.0f} mm idx={peak.index} val={peak.value:.0f}")
    return lines


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

    values = record.profile_data
    max_value = max(max(values), 1)
    step_x = width / max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x = int(origin_x + index * step_x)
        y = int(origin_y + height - (value / max_value) * (height - 4)) - 2
        points.append((x, y))
    polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [polyline], False, (0, 220, 255), 1)

    sample = build_profile_sample(record)
    if sample is None:
        return

    peaks = sorted(extract_peaks(sample), key=lambda peak: peak.value, reverse=True)[:3]
    for peak in peaks:
        peak_x = int(origin_x + peak.index * step_x)
        peak_y = int(origin_y + height - (peak.value / max_value) * (height - 4)) - 2
        cv2.circle(canvas, (peak_x, peak_y), 3, (0, 0, 255), -1)


def draw_overlay(frame: Any, record: SonarRecord | None, error: str | None, last_update_monotonic: float | None) -> Any:
    panel_height = 190
    canvas = cv2.copyMakeBorder(frame, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    text_x = 16
    text_y = frame.shape[0] + 28
    line_height = 24

    if error is not None:
        cv2.putText(canvas, f"sonar error: {error}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        return canvas

    if record is None:
        cv2.putText(canvas, "waiting for sonar...", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    age_text = "n/a" if last_update_monotonic is None else f"{time.monotonic() - last_update_monotonic:.2f}s"
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

    for idx, line in enumerate(format_peak_text(record), start=len(header_lines)):
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


def resize_for_display(image: Any, max_width: int = _DISPLAY_MAX_WIDTH, max_height: int = _DISPLAY_MAX_HEIGHT) -> Any:
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return image
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def draw_alignment_guides(frame: Any) -> Any:
    guided = frame.copy()
    height, width = guided.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # 3x3 guide grid
    third_x = width // 3
    third_y = height // 3
    for x in (third_x, 2 * third_x):
        cv2.line(guided, (x, 0), (x, height), _GRID_COLOR, 1, cv2.LINE_AA)
    for y in (third_y, 2 * third_y):
        cv2.line(guided, (0, y), (width, y), _GRID_COLOR, 1, cv2.LINE_AA)

    # Center crosshair
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
    return guided


def main() -> int:
    logger = get_app_logger("live_camera_sonar_check", PROJECT_ROOT / "logs")
    capture: WebcamCapture | None = None
    sonar_client: PingSonarClient | None = None
    stop_event = threading.Event()
    sonar_thread: threading.Thread | None = None

    try:
        camera_raw = load_yaml_config(PROJECT_ROOT / "configs" / "camera.yaml")
        sonar_raw = load_yaml_config(PROJECT_ROOT / "configs" / "sonar.yaml")
        level = resolve_log_level(camera_raw, sonar_raw)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

        camera_config = load_camera_config(camera_raw)
        sonar_config = load_sonar_config(sonar_raw)
        if not camera_config.window_name:
            camera_config.window_name = "Live Camera Sonar Check"

        capture = WebcamCapture(camera_config, logger=logger)
        capture.open()

        state = SharedSonarState()
        sonar_client = PingSonarClient(sonar_config, logger=logger)
        sonar_thread = threading.Thread(
            target=sonar_reader_loop,
            args=(sonar_client, state, stop_event, logger),
            daemon=True,
            name="live-sonar-reader",
        )
        sonar_thread.start()

        logger.info("Starting live camera + sonar check. Press 'q' to quit.")
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("Failed to read a frame from the camera.")
                break

            frame = draw_alignment_guides(frame)
            record, error, last_update_monotonic = state.snapshot()
            display = draw_overlay(frame, record, error, last_update_monotonic)
            display = resize_for_display(display)
            cv2.imshow(camera_config.window_name or "Live Camera Sonar Check", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested by user.")
                break
        return 0
    except Exception as exc:
        logger.exception("Live camera sonar check failed: %s", exc)
        return 1
    finally:
        stop_event.set()
        if sonar_thread is not None:
            sonar_thread.join(timeout=3.0)
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        logger.info("Live camera sonar check finished.")


if __name__ == "__main__":
    raise SystemExit(main())
