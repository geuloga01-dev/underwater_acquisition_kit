from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2


@dataclass(slots=True)
class RecordingConfig:
    mode: str = "video"
    duration_seconds: float = 300.0
    preview: bool = False
    fourcc: str = "mp4v"
    container: str = "mp4"
    output_path: str | None = None
    image_extension: str = "jpg"
    image_quality: int = 98


def load_recording_config(raw_config: dict[str, Any]) -> RecordingConfig:
    section = raw_config.get("recording", {})
    return RecordingConfig(
        mode=str(section.get("mode", "video")).strip().lower(),
        duration_seconds=float(section.get("duration_seconds", 300)),
        preview=_optional_bool(section.get("preview"), default=False),
        fourcc=str(section.get("fourcc", "mp4v")),
        container=str(section.get("container", "mp4")),
        output_path=section.get("output_path"),
        image_extension=str(section.get("image_extension", "jpg")).strip().lower(),
        image_quality=int(section.get("image_quality", 98)),
    )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class VideoRecorder:
    def __init__(
        self,
        output_path: Path,
        recording_config: RecordingConfig,
        frame_size: tuple[int, int],
        fps: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_path = output_path
        self.recording_config = recording_config
        self.frame_size = frame_size
        self.fps = fps
        self.logger = logger or logging.getLogger(__name__)
        self.writer: cv2.VideoWriter | None = None
        self._start_timestamp: float | None = None
        self._frames_written = 0
        self._duplicate_frames = 0
        self._last_frame = None

    def _open_writer(self, frame) -> None:
        if self.writer is not None:
            return

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid frame size received from camera.")

        fourcc = cv2.VideoWriter_fourcc(*self.recording_config.fourcc)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for '{self.output_path}'.")

        self.logger.info(
            "Video writer opened. size=%sx%s fourcc=%s fps=%.2f output=%s",
            width,
            height,
            self.recording_config.fourcc,
            self.fps,
            self.output_path,
        )

    def write(self, frame, timestamp: float | None = None) -> None:
        if self.writer is None:
            self._open_writer(frame)
        if self.writer is None:
            raise RuntimeError("Video writer was not initialized.")

        if timestamp is None or self.fps <= 0:
            self.writer.write(frame)
            self._frames_written += 1
            self._last_frame = frame.copy()
            return

        if self._start_timestamp is None:
            self._start_timestamp = timestamp

        target_frame_count = max(
            self._frames_written + 1,
            int(round((timestamp - self._start_timestamp) * self.fps)) + 1,
        )

        while self._last_frame is not None and self._frames_written < target_frame_count - 1:
            self.writer.write(self._last_frame)
            self._frames_written += 1
            self._duplicate_frames += 1

        self.writer.write(frame)
        self._frames_written += 1
        self._last_frame = frame.copy()

    def release(self) -> None:
        if self.writer is not None:
            self.logger.info(
                "Video writer released. total_frames=%d duplicate_frames=%d",
                self._frames_written,
                self._duplicate_frames,
            )
            self.writer.release()
            self.writer = None


class ImageSequenceRecorder:
    def __init__(
        self,
        output_dir: Path,
        recording_config: RecordingConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.recording_config = recording_config
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_written = 0
        self._extension = self.recording_config.image_extension.lstrip(".") or "jpg"
        self._params = self._build_imwrite_params()
        self.logger.info(
            "Image sequence recorder opened. dir=%s extension=%s quality=%s",
            self.output_dir,
            self._extension,
            self.recording_config.image_quality,
        )

    def _build_imwrite_params(self) -> list[int]:
        ext = self._extension.lower()
        if ext in {"jpg", "jpeg"}:
            return [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, self.recording_config.image_quality))]
        if ext == "png":
            compression = 1 if self.recording_config.image_quality >= 95 else 3
            return [cv2.IMWRITE_PNG_COMPRESSION, compression]
        return []

    def write(self, frame, frame_id: int, timestamp: float | None = None) -> None:
        output_path = self.output_dir / f"frame_{frame_id:06d}.{self._extension}"
        ok = cv2.imwrite(str(output_path), frame, self._params)
        if not ok:
            raise RuntimeError(f"Could not write frame image '{output_path}'.")
        self._frames_written += 1

    def release(self) -> None:
        self.logger.info(
            "Image sequence recorder released. frames=%d dir=%s",
            self._frames_written,
            self.output_dir,
        )


class FrameTimestampWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame_id", "timestamp"])

    def write(self, frame_id: int, timestamp: float) -> None:
        self._writer.writerow([frame_id, timestamp])

    def release(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()
