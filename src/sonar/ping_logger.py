from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any


_MAX_CONNECT_ATTEMPTS = 5
_RETRY_DELAY_SECONDS = 1.5
_SERIAL_SETTLE_SECONDS = 0.75
_PROFILE_WARN_LOG_LIMIT = 3
_PROFILE_DEFAULT_BATCH_SIZE = 16
_PROFILE_DEFAULT_QUEUE_SIZE = 128
_PROFILE_DEFAULT_FLUSH_INTERVAL_SECONDS = 1.0
_CSV_FIELDNAMES = [
    "timestamp",
    "distance_mm",
    "confidence",
    "valid",
    "scan_start_mm",
    "scan_length_mm",
    "gain_setting",
    "mode_auto",
    "transmit_duration_us",
    "ping_number",
]


@dataclass(slots=True)
class SonarConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    sample_interval: float = 0.2
    csv_save: bool = True
    csv_path: str | None = None
    telemetry_enabled: bool = False
    profile_save: bool = True
    profile_path: str | None = None
    profile_read_enabled: bool = True
    profile_batch_size: int = _PROFILE_DEFAULT_BATCH_SIZE
    profile_queue_size: int = _PROFILE_DEFAULT_QUEUE_SIZE
    profile_flush_interval_seconds: float = _PROFILE_DEFAULT_FLUSH_INTERVAL_SECONDS
    scan_start_mm: int | None = None
    scan_length_mm: int | None = None
    gain_setting: int | None = None
    mode_auto: bool | None = None
    transmit_duration_us: int | None = None


@dataclass(slots=True)
class SonarRecord:
    timestamp: float
    distance_mm: int | None
    confidence: int | None
    valid: bool | None = None
    scan_start_mm: int | None = None
    scan_length_mm: int | None = None
    gain_setting: int | None = None
    mode_auto: bool | None = None
    transmit_duration_us: int | None = None
    ping_number: int | None = None
    profile_data: list[int] | None = None

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
        profile_save=_optional_bool(section.get("profile_save"), default=True),
        profile_path=section.get("profile_path"),
        profile_read_enabled=_optional_bool(section.get("profile_read_enabled"), default=True),
        profile_batch_size=max(1, int(section.get("profile_batch_size", _PROFILE_DEFAULT_BATCH_SIZE))),
        profile_queue_size=max(1, int(section.get("profile_queue_size", _PROFILE_DEFAULT_QUEUE_SIZE))),
        profile_flush_interval_seconds=max(
            0.1,
            float(section.get("profile_flush_interval_seconds", _PROFILE_DEFAULT_FLUSH_INTERVAL_SECONDS)),
        ),
        scan_start_mm=_as_int(section.get("scan_start_mm")),
        scan_length_mm=_as_int(section.get("scan_length_mm")),
        gain_setting=_as_int(section.get("gain_setting")),
        mode_auto=_optional_bool(section.get("mode_auto"), default=None),
        transmit_duration_us=_as_int(section.get("transmit_duration_us")),
    )


def _optional_bool(value: Any, default: bool | None = False) -> bool | None:
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
        self._profile_supported: bool | None = None
        self._profile_warning_count = 0

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

                self._apply_fixed_settings()

                self.logger.info("Running first sonar probe read. attempt=%d/%d", attempt, _MAX_CONNECT_ATTEMPTS)
                first_record = self.read_record()
                self.logger.info(
                    "Sonar first probe read success. attempt=%d/%d distance_mm=%s confidence=%s profile_enabled=%s profile_supported=%s",
                    attempt,
                    _MAX_CONNECT_ATTEMPTS,
                    first_record.distance_mm,
                    first_record.confidence,
                    self.config.profile_read_enabled,
                    self._profile_supported,
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

        distance_message = self._device.get_distance()
        if not distance_message:
            raise RuntimeError("Failed to read distance data from Ping Sonar.")

        profile_message = None
        if self.config.profile_read_enabled:
            profile_message = self._try_get_profile_message()

        return normalize_record(distance_message, profile_message, self.config)

    def close(self) -> None:
        if self._device is None:
            return

        serial_handle = getattr(self._device, "iodev", None)
        if serial_handle is not None:
            close_method = getattr(serial_handle, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
        self._device = None

    def _apply_fixed_settings(self) -> None:
        if self._device is None:
            return

        settings = (
            ("set_mode_auto", self.config.mode_auto),
            ("set_gain_setting", self.config.gain_setting),
        )

        for method_name, value in settings:
            if value in (None, ""):
                continue
            method = getattr(self._device, method_name, None)
            if not callable(method):
                self.logger.debug("Sonar setting method not available: %s", method_name)
                continue
            try:
                method(value)
                self.logger.info("Applied sonar fixed setting %s=%s", method_name, value)
            except Exception as exc:
                self.logger.warning("Could not apply sonar setting %s=%s: %s", method_name, value, exc)

        if self.config.scan_start_mm is not None or self.config.scan_length_mm is not None:
            range_method = getattr(self._device, "set_range", None)
            if not callable(range_method):
                self.logger.debug("Sonar range setting method not available: set_range")
            elif self.config.scan_start_mm is None or self.config.scan_length_mm is None:
                self.logger.warning(
                    "Skipping set_range because both scan_start_mm and scan_length_mm are required. scan_start_mm=%s scan_length_mm=%s",
                    self.config.scan_start_mm,
                    self.config.scan_length_mm,
                )
            else:
                try:
                    range_method(self.config.scan_start_mm, self.config.scan_length_mm)
                    self.logger.info(
                        "Applied sonar fixed setting set_range(start_mm=%s, length_mm=%s)",
                        self.config.scan_start_mm,
                        self.config.scan_length_mm,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Could not apply sonar range setting start_mm=%s length_mm=%s: %s",
                        self.config.scan_start_mm,
                        self.config.scan_length_mm,
                        exc,
                    )

        if self.config.transmit_duration_us is not None:
            self.logger.info(
                "Configured transmit_duration_us=%s will be recorded when the device reports it. No compatible ping-python setter was found in the standard Ping1D API.",
                self.config.transmit_duration_us,
            )

    def _try_get_profile_message(self) -> dict[str, Any] | None:
        if self._device is None:
            return None

        getter = getattr(self._device, "get_profile", None)
        if not callable(getter):
            if self._profile_supported is None:
                self._profile_supported = False
                self.logger.info("Sonar profile API not available in current brping/device path. CSV logging will continue.")
            return None

        try:
            profile_message = getter()
            self._profile_supported = bool(profile_message)
            if self._profile_supported:
                return profile_message
            self._log_profile_warning_once("Sonar profile read returned no data. CSV logging will continue.")
        except Exception as exc:
            self._profile_supported = False
            self._log_profile_warning_once(f"Sonar profile read failed and will be skipped: {exc}")
        return None

    def _log_profile_warning_once(self, message: str) -> None:
        if self._profile_warning_count >= _PROFILE_WARN_LOG_LIMIT:
            return
        self._profile_warning_count += 1
        self.logger.warning(message)

    @property
    def profile_supported(self) -> bool | None:
        return self._profile_supported


class BufferedProfileWriter:
    def __init__(
        self,
        output_path: Path,
        logger: logging.Logger,
        batch_size: int,
        queue_size: int,
        flush_interval_seconds: float,
    ) -> None:
        self.output_path = output_path
        self.logger = logger
        self.batch_size = max(1, batch_size)
        self.flush_interval_seconds = max(0.1, flush_interval_seconds)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._drop_count = 0
        self._write_error_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="sonar-profile-writer", daemon=True)
        self._thread.start()
        self.logger.info(
            "Sonar profile writer started. path=%s batch_size=%s queue_size=%s flush_interval=%.2fs",
            self.output_path,
            self.batch_size,
            self._queue.maxsize,
            self.flush_interval_seconds,
        )

    def enqueue(self, record: SonarRecord) -> None:
        if record.profile_data is None:
            return

        payload = build_profile_payload(record)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._drop_count += 1
            if self._drop_count <= _PROFILE_WARN_LOG_LIMIT or self._drop_count % 25 == 0:
                self.logger.warning(
                    "Sonar profile queue overflow. dropped=%d queue_size=%d",
                    self._drop_count,
                    self._queue.maxsize,
                )

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.logger.info(
            "Sonar profile writer stopped. path=%s dropped=%d write_errors=%d",
            self.output_path,
            self._drop_count,
            self._write_error_count,
        )

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()

        while True:
            timeout = max(0.1, self.flush_interval_seconds - (time.monotonic() - last_flush))
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is not None:
                batch.append(item)

            should_flush = bool(batch) and (
                len(batch) >= self.batch_size
                or time.monotonic() - last_flush >= self.flush_interval_seconds
                or item is None
            )
            if should_flush:
                self._flush_batch(batch)
                batch = []
                last_flush = time.monotonic()

            if self._stop_event.is_set() and self._queue.empty():
                if batch:
                    self._flush_batch(batch)
                break

    def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        try:
            with self.output_path.open("a", encoding="utf-8") as file:
                for payload in batch:
                    file.write(json.dumps(payload, ensure_ascii=True))
                    file.write("\n")
        except Exception as exc:
            self._write_error_count += 1
            if self._write_error_count <= _PROFILE_WARN_LOG_LIMIT:
                self.logger.warning("Sonar profile JSONL write failed but logging will continue: %s", exc)


def prepare_sonar(client: PingSonarClient) -> SonarRecord:
    return client.connect_and_validate()


def normalize_record(
    distance_message: dict[str, Any],
    profile_message: dict[str, Any] | None,
    config: SonarConfig | None = None,
) -> SonarRecord:
    timestamp = time.time()
    return SonarRecord(
        timestamp=timestamp,
        distance_mm=_first_int("distance", distance_message, profile_message),
        confidence=_first_int("confidence", distance_message, profile_message),
        valid=_infer_valid(distance_message, profile_message),
        scan_start_mm=_first_int(
            "scan_start",
            distance_message,
            profile_message,
            fallback=getattr(config, "scan_start_mm", None),
        ),
        scan_length_mm=_first_int(
            "scan_length",
            distance_message,
            profile_message,
            fallback=getattr(config, "scan_length_mm", None),
        ),
        gain_setting=_first_int(
            "gain_setting",
            distance_message,
            profile_message,
            fallback=getattr(config, "gain_setting", None),
        ),
        mode_auto=_first_bool(
            "mode_auto",
            distance_message,
            profile_message,
            fallback=getattr(config, "mode_auto", None),
        ),
        transmit_duration_us=_first_int(
            "transmit_duration",
            distance_message,
            profile_message,
            fallback=getattr(config, "transmit_duration_us", None),
        ),
        ping_number=_first_int("ping_number", distance_message, profile_message),
        profile_data=_normalize_profile_data(profile_message.get("profile_data") if profile_message else None),
    )


def build_telemetry_packet(record: SonarRecord) -> dict[str, Any]:
    return {
        "timestamp": record.timestamp,
        "distance_mm": record.distance_mm,
        "confidence": record.confidence,
        "valid": record.valid,
        "scan_start_mm": record.scan_start_mm,
        "scan_length_mm": record.scan_length_mm,
        "gain_setting": record.gain_setting,
        "mode_auto": record.mode_auto,
        "transmit_duration_us": record.transmit_duration_us,
        "ping_number": record.ping_number,
    }


def append_csv_record(csv_path: Path, record: SonarRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": record.timestamp,
                "distance_mm": record.distance_mm,
                "confidence": record.confidence,
                "valid": record.valid,
                "scan_start_mm": record.scan_start_mm,
                "scan_length_mm": record.scan_length_mm,
                "gain_setting": record.gain_setting,
                "mode_auto": record.mode_auto,
                "transmit_duration_us": record.transmit_duration_us,
                "ping_number": record.ping_number,
            }
        )


def build_profile_payload(record: SonarRecord) -> dict[str, Any]:
    return {
        "timestamp": record.timestamp,
        "distance_mm": record.distance_mm,
        "confidence": record.confidence,
        "valid": record.valid,
        "scan_start_mm": record.scan_start_mm,
        "scan_length_mm": record.scan_length_mm,
        "gain_setting": record.gain_setting,
        "mode_auto": record.mode_auto,
        "transmit_duration_us": record.transmit_duration_us,
        "ping_number": record.ping_number,
        "profile_data": record.profile_data,
    }


def log_sonar_stream(
    client: PingSonarClient,
    config: SonarConfig,
    logger: logging.Logger,
    csv_path: Path | None = None,
    profile_path: Path | None = None,
    max_samples: int | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    sample_count = 0
    profile_logging_enabled = bool(config.profile_save and profile_path is not None and config.profile_read_enabled)
    profile_writer: BufferedProfileWriter | None = None
    logger.info(
        "Starting sonar logging. profile_logging_enabled=%s profile_path=%s profile_batch_size=%s profile_queue_size=%s",
        profile_logging_enabled,
        profile_path,
        config.profile_batch_size,
        config.profile_queue_size,
    )

    if profile_logging_enabled and profile_path is not None:
        profile_writer = BufferedProfileWriter(
            output_path=profile_path,
            logger=logger,
            batch_size=config.profile_batch_size,
            queue_size=config.profile_queue_size,
            flush_interval_seconds=config.profile_flush_interval_seconds,
        )
        profile_writer.start()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            record = client.read_record()
            logger.info(
                "Sonar sample distance_mm=%s confidence=%s ping_number=%s profile_data=%s",
                record.distance_mm,
                record.confidence,
                record.ping_number,
                "yes" if record.profile_data is not None else "no",
            )

            if config.csv_save and csv_path is not None:
                append_csv_record(csv_path, record)

            if profile_writer is not None:
                profile_writer.enqueue(record)

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
    finally:
        if profile_writer is not None:
            profile_writer.close()

    return sample_count


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normalize_profile_data(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    return None


def _first_int(field_name: str, *messages: dict[str, Any] | None, fallback: int | None = None) -> int | None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        value = message.get(field_name)
        if value in (None, ""):
            continue
        return int(value)
    return fallback


def _first_bool(field_name: str, *messages: dict[str, Any] | None, fallback: bool | None = None) -> bool | None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        value = message.get(field_name)
        if value in (None, ""):
            continue
        return bool(value)
    return fallback


def _infer_valid(*messages: dict[str, Any] | None) -> bool | None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if "valid" in message and message.get("valid") not in (None, ""):
            return bool(message.get("valid"))

    distance = _first_int("distance", *messages)
    confidence = _first_int("confidence", *messages)
    if distance is None:
        return False
    if confidence is None:
        return True
    return confidence > 0
