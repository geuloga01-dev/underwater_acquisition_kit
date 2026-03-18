from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import logging
from pathlib import Path
import threading
import time
from typing import Any


_MAX_CONNECT_ATTEMPTS = 5
_RETRY_DELAY_SECONDS = 1.5
_SERIAL_SETTLE_SECONDS = 0.75


@dataclass(slots=True)
class SonarConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    sample_interval: float = 0.2
    csv_save: bool = True
    csv_path: str | None = None
    telemetry_enabled: bool = False


@dataclass(slots=True)
class SonarRecord:
    timestamp: float
    distance_mm: int | None
    confidence: int | None

    @property
    def unix_time(self) -> float:
        return self.timestamp

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat()


def load_sonar_config(raw_config: dict[str, Any]) -> SonarConfig:
    section = raw_config.get("sonar", {})
    return SonarConfig(
        port=str(section.get("port", "/dev/ttyUSB0")),
        baudrate=int(section.get("baudrate", 115200)),
        sample_interval=float(section.get("sample_interval", 0.2)),
        csv_save=_optional_bool(section.get("csv_save"), default=True),
        csv_path=section.get("csv_path"),
        telemetry_enabled=_optional_bool(section.get("telemetry_enabled"), default=False),
    )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class PingSonarClient:
    def __init__(self, config: SonarConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._device = None

    def connect(self) -> None:
        self.connect_and_validate()

    def connect_and_validate(self) -> SonarRecord:
        try:
            from brping import Ping1D
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "brping is not installed. Install it on Jetson or the Ubuntu environment before running sonar logging."
            ) from exc

        last_error: Exception | None = None
        previous_attempt_started_at: float | None = None

        for attempt in range(1, _MAX_CONNECT_ATTEMPTS + 1):
            attempt_started_at = time.monotonic()
            elapsed_since_previous = (
                None if previous_attempt_started_at is None else attempt_started_at - previous_attempt_started_at
            )
            previous_attempt_started_at = attempt_started_at
            serial_open_success = False
            initialize_called = False
            initialize_result: bool | None = None

            try:
                self.logger.info(
                    "Sonar init start. attempt=%d/%d port=%s baudrate=%s since_previous_attempt=%s",
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                    self.config.port,
                    self.config.baudrate,
                    "n/a" if elapsed_since_previous is None else f"{elapsed_since_previous:.2f}s",
                )

                self.close()
                self._device = Ping1D()
                self.logger.info("Sonar serial object create success. attempt=%d/%d", attempt, _MAX_CONNECT_ATTEMPTS)
                self._device.connect_serial(self.config.port, self.config.baudrate)
                serial_open_success = True
                self.logger.info(
                    "Sonar serial open success. attempt=%d/%d port=%s baudrate=%s",
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                    self.config.port,
                    self.config.baudrate,
                )
                self.logger.info(
                    "Waiting %.2fs before initialize to let serial settle. attempt=%d/%d",
                    _SERIAL_SETTLE_SECONDS,
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                )
                time.sleep(_SERIAL_SETTLE_SECONDS)

                initialize_called = True
                self.logger.info("Calling sonar initialize(). attempt=%d/%d", attempt, _MAX_CONNECT_ATTEMPTS)
                initialize_result = bool(self._device.initialize())
                self.logger.info(
                    "Sonar initialize returned: %s. attempt=%d/%d",
                    initialize_result,
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                )
                if not initialize_result:
                    raise RuntimeError("Ping1D initialize returned False.")

                self.logger.info("Running first sonar probe read. attempt=%d/%d", attempt, _MAX_CONNECT_ATTEMPTS)
                first_message = self._device.get_distance()
                if not first_message:
                    raise RuntimeError("First distance read returned no data.")
                first_record = normalize_record(first_message)
                self.logger.info(
                    "Sonar first probe read success. attempt=%d/%d distance_mm=%s confidence=%s",
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                    first_record.distance_mm,
                    first_record.confidence,
                )

                self.logger.info(
                    "Connected to Ping Sonar. port=%s baudrate=%s",
                    self.config.port,
                    self.config.baudrate,
                )
                return first_record
            except Exception as exc:
                last_error = exc
                self.close()
                self.logger.warning(
                    "Sonar init failed. attempt=%d/%d port=%s baudrate=%s serial_open_success=%s initialize_called=%s initialize_result=%s exception_class=%s exception=%s",
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                    self.config.port,
                    self.config.baudrate,
                    serial_open_success,
                    initialize_called,
                    initialize_result,
                    exc.__class__.__name__,
                    exc,
                )
                if attempt < _MAX_CONNECT_ATTEMPTS:
                    self.logger.info(
                        "Sleeping %.2fs before next sonar init retry. next_attempt=%d/%d",
                        _RETRY_DELAY_SECONDS,
                        attempt + 1,
                        _MAX_CONNECT_ATTEMPTS,
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"Could not initialize Ping Sonar device on {self.config.port} after {_MAX_CONNECT_ATTEMPTS} attempts."
        ) from last_error

    def read_record(self) -> SonarRecord:
        if self._device is None:
            raise RuntimeError("Sonar device has not been connected yet.")

        message = self._device.get_distance()
        if not message:
            raise RuntimeError("Failed to read distance data from Ping Sonar.")

        return normalize_record(message)

    def close(self) -> None:
        if self._device is None:
            return

        # Best-effort serial cleanup before retrying or exiting.
        serial_handle = getattr(self._device, "iodev", None)
        if serial_handle is not None:
            close_method = getattr(serial_handle, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
        self._device = None


def prepare_sonar(client: PingSonarClient) -> SonarRecord:
    return client.connect_and_validate()


def normalize_record(raw_message: dict[str, Any]) -> SonarRecord:
    timestamp = time.time()
    return SonarRecord(
        timestamp=timestamp,
        distance_mm=_as_int(raw_message.get("distance")),
        confidence=_as_int(raw_message.get("confidence")),
    )


def build_telemetry_packet(record: SonarRecord) -> dict[str, Any]:
    # Keep packet generation separate so local logging and future telemetry publish stay decoupled.
    return {
        "timestamp": record.timestamp,
        "distance_mm": record.distance_mm,
        "confidence": record.confidence,
    }


def append_csv_record(csv_path: Path, record: SonarRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timestamp", "distance_mm", "confidence"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": record.timestamp,
                "distance_mm": record.distance_mm,
                "confidence": record.confidence,
            }
        )


def log_sonar_stream(
    client: PingSonarClient,
    config: SonarConfig,
    logger: logging.Logger,
    csv_path: Path | None = None,
    max_samples: int | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    sample_count = 0
    logger.info("Starting sonar logging. Press Ctrl+C to stop.")

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            record = client.read_record()
            logger.info(
                "Sonar sample distance_mm=%s confidence=%s",
                record.distance_mm,
                record.confidence,
            )

            if config.csv_save and csv_path is not None:
                append_csv_record(csv_path, record)

            if config.telemetry_enabled:
                packet = build_telemetry_packet(record)
                logger.debug("Telemetry packet prepared: %s", packet)

            sample_count += 1
            if max_samples is not None and sample_count >= max_samples:
                break

            if stop_event is None:
                time.sleep(config.sample_interval)
            else:
                stop_event.wait(config.sample_interval)
    except KeyboardInterrupt:
        logger.info("Sonar logging interrupted by user.")

    return sample_count


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
