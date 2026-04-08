from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import logging
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any

from src.state.runtime_state import RuntimeState


@dataclass(slots=True)
class GpsConfig:
    enabled: bool = False
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout_seconds: float = 1.0
    csv_save: bool = True
    csv_path: str | None = None


@dataclass(slots=True)
class GpsRecord:
    timestamp: float
    sentence_type: str
    nmea_utc: str | None
    latitude_deg: float | None
    longitude_deg: float | None
    altitude_m: float | None
    fix_quality: int | None
    num_satellites: int | None
    hdop: float | None
    speed_knots: float | None
    track_deg: float | None

    @property
    def unix_time(self) -> float:
        return self.timestamp

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat()


def load_gps_config(raw_config: dict[str, Any]) -> GpsConfig:
    section = raw_config.get("gps", {})
    env_enabled = os.getenv("UAK_GPS_ENABLED")
    env_port = os.getenv("UAK_GPS_PORT")
    env_baudrate = os.getenv("UAK_GPS_BAUDRATE")
    return GpsConfig(
        enabled=_optional_bool(env_enabled if env_enabled not in (None, "") else section.get("enabled"), default=False),
        port=str(env_port if env_port not in (None, "") else section.get("port", "/dev/ttyUSB0")),
        baudrate=int(env_baudrate if env_baudrate not in (None, "") else section.get("baudrate", 115200)),
        timeout_seconds=float(section.get("timeout_seconds", 1.0)),
        csv_save=_optional_bool(section.get("csv_save"), default=True),
        csv_path=section.get("csv_path"),
    )


class GpsListener:
    def __init__(self, config: GpsConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._serial = None
        self._latest_fix: dict[str, Any] = {}

    def connect(self) -> None:
        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserial is not installed. Install it on Jetson for GPS logging.") from exc

        self._serial = serial.Serial(
            self.config.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout_seconds,
        )
        self.logger.info(
            "GPS serial connection opened. port=%s baudrate=%s timeout=%.2fs",
            self.config.port,
            self.config.baudrate,
            self.config.timeout_seconds,
        )

    def reconnect(self, delay_seconds: float = 0.2) -> None:
        self.close()
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        self.connect()

    def read_record(self, include_partial: bool = False) -> GpsRecord | None:
        if self._serial is None:
            raise RuntimeError("GPS listener has not been connected yet.")

        raw_line = self._serial.readline()
        if not raw_line:
            return None

        ubx_record = _extract_ubx_nav_pvt_record(raw_line)
        if ubx_record is not None:
            return ubx_record

        line = _extract_nmea_sentence(raw_line)
        if line is None:
            return None

        body = line[1:].split("*", 1)[0]
        fields = body.split(",")
        if not fields:
            return None

        sentence_type = fields[0]
        parsed: dict[str, Any] | None = None
        if sentence_type.endswith("GGA"):
            parsed = _parse_gga(fields)
        elif sentence_type.endswith("RMC"):
            parsed = _parse_rmc(fields)
        else:
            return None

        if parsed is None:
            return None

        for key, value in parsed.items():
            if value is not None:
                self._latest_fix[key] = value

        if (
            not include_partial
            and (self._latest_fix.get("latitude_deg") is None or self._latest_fix.get("longitude_deg") is None)
        ):
            return None

        return GpsRecord(
            timestamp=_resolve_record_timestamp(self._latest_fix),
            sentence_type=sentence_type,
            nmea_utc=_as_str(self._latest_fix.get("nmea_utc")),
            latitude_deg=_as_float(self._latest_fix.get("latitude_deg")),
            longitude_deg=_as_float(self._latest_fix.get("longitude_deg")),
            altitude_m=_as_float(self._latest_fix.get("altitude_m")),
            fix_quality=_as_int(self._latest_fix.get("fix_quality")),
            num_satellites=_as_int(self._latest_fix.get("num_satellites")),
            hdop=_as_float(self._latest_fix.get("hdop")),
            speed_knots=_as_float(self._latest_fix.get("speed_knots")),
            track_deg=_as_float(self._latest_fix.get("track_deg")),
        )

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


def append_gps_csv(csv_path: Path, record: GpsRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "sentence_type",
                "nmea_utc",
                "latitude_deg",
                "longitude_deg",
                "altitude_m",
                "fix_quality",
                "num_satellites",
                "hdop",
                "speed_knots",
                "track_deg",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": record.timestamp,
                "sentence_type": record.sentence_type,
                "nmea_utc": record.nmea_utc,
                "latitude_deg": record.latitude_deg,
                "longitude_deg": record.longitude_deg,
                "altitude_m": record.altitude_m,
                "fix_quality": record.fix_quality,
                "num_satellites": record.num_satellites,
                "hdop": record.hdop,
                "speed_knots": record.speed_knots,
                "track_deg": record.track_deg,
            }
        )


def run_gps_logging_loop(
    listener: GpsListener,
    config: GpsConfig,
    logger: logging.Logger,
    state: RuntimeState,
    csv_path: Path | None,
    stop_event: threading.Event,
) -> int:
    sample_count = 0
    state.update_component("gps", ready=True, running=True, ok=True, last_error=None)
    logger.info("GPS logging loop started.")

    try:
        while not stop_event.is_set():
            try:
                record = listener.read_record()
            except Exception as exc:
                if not _is_recoverable_serial_error(exc):
                    raise
                state.update_component("gps", running=True, ok=False, last_error=str(exc))
                logger.warning("GPS serial read hiccup, reconnecting: %s", exc)
                listener.reconnect()
                state.update_component("gps", running=True, ok=True, last_error=None)
                continue
            if record is None:
                continue

            state.set_gps_state(
                timestamp_iso=record.timestamp_iso,
                unix_time=record.unix_time,
                latitude_deg=record.latitude_deg,
                longitude_deg=record.longitude_deg,
                altitude_m=record.altitude_m,
                fix_quality=record.fix_quality,
                num_satellites=record.num_satellites,
                hdop=record.hdop,
                speed_knots=record.speed_knots,
                track_deg=record.track_deg,
            )

            if csv_path is not None and config.csv_save:
                append_gps_csv(csv_path, record)

            sample_count += 1
    except Exception as exc:
        state.update_component("gps", running=False, ok=False, last_error=str(exc))
        logger.exception("GPS logging loop failed: %s", exc)
        return sample_count

    state.update_component("gps", running=False, ok=True, last_error=None)
    return sample_count


def _parse_gga(fields: list[str]) -> dict[str, Any] | None:
    if len(fields) < 10:
        return None
    return {
        "nmea_utc": _optional_text(fields[1]),
        "latitude_deg": _parse_lat_lon(fields[2], fields[3]),
        "longitude_deg": _parse_lat_lon(fields[4], fields[5]),
        "fix_quality": _as_int(fields[6]),
        "num_satellites": _as_int(fields[7]),
        "hdop": _as_float(fields[8]),
        "altitude_m": _as_float(fields[9]),
    }


def _parse_rmc(fields: list[str]) -> dict[str, Any] | None:
    if len(fields) < 9:
        return None
    status = _optional_text(fields[2])
    latitude = _parse_lat_lon(fields[3], fields[4])
    longitude = _parse_lat_lon(fields[5], fields[6])
    return {
        "nmea_utc": _optional_text(fields[1]),
        "status": status,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "speed_knots": _as_float(fields[7]),
        "track_deg": _as_float(fields[8]),
    }


def _parse_lat_lon(raw_value: str, hemisphere: str) -> float | None:
    text = _optional_text(raw_value)
    hemi = _optional_text(hemisphere)
    if text is None or hemi is None:
        return None

    try:
        numeric = float(text)
    except ValueError:
        return None

    degrees = int(numeric // 100)
    minutes = numeric - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemi in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _checksum_matches(sentence: str) -> bool:
    if "*" not in sentence:
        return True
    body, checksum_text = sentence[1:].split("*", 1)
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    try:
        return checksum == int(checksum_text[:2], 16)
    except ValueError:
        return False


def _extract_nmea_sentence(raw_line: bytes) -> str | None:
    text = raw_line.decode("ascii", errors="ignore")
    start = text.find("$")
    if start < 0:
        return None

    candidate = text[start:]
    line_endings = [idx for idx in (candidate.find("\r"), candidate.find("\n")) if idx >= 0]
    if line_endings:
        candidate = candidate[: min(line_endings)]

    if "*" in candidate:
        star = candidate.find("*")
        if star + 3 <= len(candidate):
            candidate = candidate[: star + 3]

    candidate = candidate.strip()
    if not candidate.startswith("$"):
        return None
    if not _checksum_matches(candidate):
        return None
    return candidate


def _extract_ubx_nav_pvt_record(raw_line: bytes) -> GpsRecord | None:
    packet = _extract_ubx_packet(raw_line, msg_class=0x01, msg_id=0x07)
    if packet is None:
        return None

    payload = packet[6:-2]
    if len(payload) < 92:
        return None

    fix_type = payload[20]
    flags = payload[21]
    num_satellites = payload[23]
    longitude_deg = struct.unpack_from("<i", payload, 24)[0] / 1e7
    latitude_deg = struct.unpack_from("<i", payload, 28)[0] / 1e7
    h_msl_m = struct.unpack_from("<i", payload, 36)[0] / 1000.0
    ground_speed_knots = struct.unpack_from("<i", payload, 60)[0] / 1000.0 * 1.9438444924406048
    heading_deg = struct.unpack_from("<i", payload, 64)[0] / 1e5
    p_dop = struct.unpack_from("<H", payload, 76)[0] / 100.0

    year = struct.unpack_from("<H", payload, 4)[0]
    month = payload[6]
    day = payload[7]
    hour = payload[8]
    minute = payload[9]
    second = payload[10]
    nano = struct.unpack_from("<i", payload, 16)[0]
    nmea_utc = None
    if year and month and day:
        nmea_utc = f"{hour:02d}{minute:02d}{second:02d}"
    timestamp = _utc_timestamp_from_parts(year, month, day, hour, minute, second, nano)

    carr_soln = (flags >> 6) & 0x03
    fix_quality = _ubx_fix_quality(fix_type, carr_soln)

    return GpsRecord(
        timestamp=timestamp,
        sentence_type="UBX-NAV-PVT",
        nmea_utc=nmea_utc,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        altitude_m=h_msl_m,
        fix_quality=fix_quality,
        num_satellites=num_satellites,
        hdop=p_dop,
        speed_knots=ground_speed_knots,
        track_deg=heading_deg,
    )


def _extract_ubx_packet(raw_bytes: bytes, msg_class: int, msg_id: int) -> bytes | None:
    start = raw_bytes.find(b"\xb5\x62")
    if start < 0:
        return None

    data = raw_bytes[start:]
    if len(data) < 8:
        return None
    if data[2] != msg_class or data[3] != msg_id:
        return None

    payload_length = struct.unpack_from("<H", data, 4)[0]
    packet_length = 6 + payload_length + 2
    if len(data) < packet_length:
        return None

    packet = data[:packet_length]
    checksum = _ubx_checksum(packet[2:-2])
    if checksum != packet[-2:]:
        return None
    return packet


def _ubx_checksum(content: bytes) -> bytes:
    ck_a = 0
    ck_b = 0
    for byte in content:
        ck_a = (ck_a + byte) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return bytes((ck_a, ck_b))


def _ubx_fix_quality(fix_type: int, carr_soln: int) -> int | None:
    if carr_soln == 2:
        return 4
    if carr_soln == 1:
        return 5
    if fix_type >= 3:
        return 1
    if fix_type == 2:
        return 1
    return 0 if fix_type == 0 else None


def _resolve_record_timestamp(fields: dict[str, Any]) -> float:
    nmea_utc = _as_str(fields.get("nmea_utc"))
    if nmea_utc is None or len(nmea_utc) < 6:
        return time.time()

    now = datetime.now(timezone.utc)
    try:
        hour = int(nmea_utc[0:2])
        minute = int(nmea_utc[2:4])
        second_float = float(nmea_utc[4:])
        second = int(second_float)
        microsecond = int(round((second_float - second) * 1_000_000))
        candidate = now.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)
    except ValueError:
        return time.time()

    delta_seconds = candidate.timestamp() - now.timestamp()
    if delta_seconds > 12 * 3600:
        candidate -= timedelta(days=1)
    elif delta_seconds < -12 * 3600:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def _utc_timestamp_from_parts(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    nano: int,
) -> float:
    try:
        base = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return time.time()
    return base.timestamp() + nano / 1_000_000_000.0


def _is_recoverable_serial_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        fragment in text
        for fragment in (
            "returned no data",
            "device disconnected",
            "readiness to read",
            "input/output error",
            "resource temporarily unavailable",
        )
    )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
