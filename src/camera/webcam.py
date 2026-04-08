from __future__ import annotations

from dataclasses import dataclass
import logging
import platform
from typing import Any

import cv2


_BACKEND_MAP = {
    "any": cv2.CAP_ANY,
    "default": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
}


@dataclass(slots=True)
class CameraConfig:
    source: int | str = 0
    backend: str = "default"
    pixel_format: str | None = None
    width: int | None = 640
    height: int | None = 480
    fps: int | None = 30
    preview: bool = False
    autofocus: bool | None = True
    focus: int | None = None
    warmup_frames: int = 15
    window_name: str = "Camera Preview"


def load_camera_config(raw_config: dict[str, Any]) -> CameraConfig:
    camera_section = dict(raw_config.get("camera", {}))
    platform_overrides = raw_config.get("platform_overrides", {})
    platform_name = platform.system().lower()
    if isinstance(platform_overrides, dict):
        platform_section = platform_overrides.get(platform_name, {})
        if isinstance(platform_section, dict):
            camera_section.update(platform_section)

    return CameraConfig(
        source=camera_section.get("source", 0),
        backend=str(camera_section.get("backend", "default")).lower(),
        pixel_format=_optional_str(camera_section.get("pixel_format")),
        width=_optional_int(camera_section.get("width")),
        height=_optional_int(camera_section.get("height")),
        fps=_optional_int(camera_section.get("fps")),
        preview=_optional_bool(camera_section.get("preview"), default=False) or False,
        autofocus=_optional_bool(camera_section.get("autofocus"), default=True),
        focus=_optional_int(camera_section.get("focus")),
        warmup_frames=int(camera_section.get("warmup_frames", 15)),
        window_name=str(camera_section.get("window_name", "Camera Preview")),
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_bool(value: Any, default: bool | None = None) -> bool | None:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class WebcamCapture:
    def __init__(self, config: CameraConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.capture: cv2.VideoCapture | None = None
        self.actual_width: int | None = None
        self.actual_height: int | None = None
        self.actual_fps: float | None = None
        self.actual_pixel_format: str | None = None

    def open(self) -> None:
        backend = _BACKEND_MAP.get(self.config.backend, cv2.CAP_ANY)
        self.logger.info(
            "Opening camera source=%s backend=%s",
            self.config.source,
            self.config.backend,
        )

        self.capture = cv2.VideoCapture(self.config.source, backend)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera source '{self.config.source}' with backend '{self.config.backend}'."
            )

        self._apply_settings()

    def _apply_settings(self) -> None:
        if self.capture is None:
            return

        if self.config.pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*self.config.pixel_format[:4])
            self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        if self.config.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if self.config.autofocus is not None:
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.config.autofocus else 0)
        if self.config.focus is not None:
            self.capture.set(cv2.CAP_PROP_FOCUS, self.config.focus)

        self._warm_up_camera()
        self._capture_actual_settings()

    def _warm_up_camera(self) -> None:
        if self.capture is None:
            return

        # Give auto-exposure and autofocus a moment to settle before preview starts.
        for _ in range(max(0, self.config.warmup_frames)):
            self.capture.read()

    def read(self) -> tuple[bool, Any]:
        if self.capture is None:
            raise RuntimeError("Camera has not been opened yet.")
        return self.capture.read()

    def describe_actual_settings(self) -> dict[str, Any]:
        return {
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self.actual_fps,
            "pixel_format": self.actual_pixel_format,
        }

    def _capture_actual_settings(self) -> None:
        if self.capture is None:
            return

        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        self.actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        self.actual_pixel_format = _decode_fourcc(int(self.capture.get(cv2.CAP_PROP_FOURCC) or 0))

        self.logger.info(
            "Camera actual settings resolved. requested=%sx%s@%s format=%s actual=%sx%s@%s format=%s",
            self.config.width,
            self.config.height,
            self.config.fps,
            self.config.pixel_format,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
            self.actual_pixel_format,
        )
        if (
            self.config.width
            and self.config.height
            and self.actual_width
            and self.actual_height
            and (self.actual_width != self.config.width or self.actual_height != self.config.height)
        ):
            self.logger.warning(
                "Camera capture resolution fallback detected. requested=%sx%s actual=%sx%s",
                self.config.width,
                self.config.height,
                self.actual_width,
                self.actual_height,
            )

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def _decode_fourcc(value: int) -> str | None:
    if value <= 0:
        return None
    chars = [chr((value >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    decoded = "".join(chars).strip("\x00 ")
    return decoded or None
