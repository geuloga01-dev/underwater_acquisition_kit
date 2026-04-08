from __future__ import annotations

import argparse
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import socket
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


@dataclass(slots=True)
class RtkConfig:
    enabled: bool
    serial_port: str
    baudrate: int
    serial_timeout_seconds: float
    gga_interval_seconds: float
    ntrip_server: str
    ntrip_user: str
    ntrip_pass: str
    ntrip_mountpoint: str
    ntrip_version: str
    reconnect_delay_seconds: float
    write_rtcm_to_serial: bool
    log_nmea: bool


def load_rtk_config(raw_config: dict) -> RtkConfig:
    section = raw_config.get("rtk", {})
    return RtkConfig(
        enabled=_optional_bool(section.get("enabled"), default=False),
        serial_port=str(section.get("serial_port", "")),
        baudrate=int(section.get("baudrate", 115200)),
        serial_timeout_seconds=float(section.get("serial_timeout_seconds", 0.2)),
        gga_interval_seconds=float(section.get("gga_interval_seconds", 10.0)),
        ntrip_server=str(section.get("ntrip_server", "")),
        ntrip_user=str(section.get("ntrip_user", "")),
        ntrip_pass=str(section.get("ntrip_pass", "")),
        ntrip_mountpoint=str(section.get("ntrip_mountpoint", "")),
        ntrip_version=str(section.get("ntrip_version", "Ntrip/2.0")),
        reconnect_delay_seconds=float(section.get("reconnect_delay_seconds", 2.0)),
        write_rtcm_to_serial=_optional_bool(section.get("write_rtcm_to_serial"), default=True),
        log_nmea=_optional_bool(section.get("log_nmea"), default=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge NTRIP RTCM corrections into a u-blox GNSS receiver over serial.")
    parser.add_argument("--serial-port", default="", help="Override serial port.")
    parser.add_argument("--baudrate", type=int, default=0, help="Override serial baudrate.")
    parser.add_argument("--ntrip-server", default="", help="Override NTRIP server host:port.")
    parser.add_argument("--ntrip-user", default="", help="Override NTRIP username.")
    parser.add_argument("--ntrip-pass", default="", help="Override NTRIP password.")
    parser.add_argument("--mountpoint", default="", help="Override NTRIP mountpoint.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means until Ctrl+C.")
    return parser.parse_args()


def apply_overrides(raw_config: dict, args: argparse.Namespace) -> dict:
    section = raw_config.setdefault("rtk", {})
    if args.serial_port:
        section["serial_port"] = args.serial_port
    if args.baudrate:
        section["baudrate"] = args.baudrate
    if args.ntrip_server:
        section["ntrip_server"] = args.ntrip_server
    if args.ntrip_user:
        section["ntrip_user"] = args.ntrip_user
    if args.ntrip_pass:
        section["ntrip_pass"] = args.ntrip_pass
    if args.mountpoint:
        section["ntrip_mountpoint"] = args.mountpoint
    section["enabled"] = True
    return raw_config


class RtkBridge:
    def __init__(self, config: RtkConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.serial = None
        self.latest_gga: str | None = None
        self.latest_rmc: str | None = None
        self.last_gga_sent_monotonic = 0.0
        self.rtcm_messages = 0
        self.rtcm_bytes = 0

    def connect_serial(self) -> None:
        try:
            import serial
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserial is not installed. Install it with `pip install pyserial`.") from exc

        self.serial = serial.Serial(
            self.config.serial_port,
            baudrate=self.config.baudrate,
            timeout=self.config.serial_timeout_seconds,
        )
        self.logger.info(
            "Serial connection opened. port=%s baudrate=%s timeout=%.2fs",
            self.config.serial_port,
            self.config.baudrate,
            self.config.serial_timeout_seconds,
        )

    def close(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None

    def run(self, duration_seconds: float) -> None:
        deadline = time.monotonic() + duration_seconds if duration_seconds > 0 else None
        self.connect_serial()
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                self._prime_nmea()
                if self.latest_gga is None:
                    self.logger.info("Waiting for valid GGA sentence before opening NTRIP stream...")
                    time.sleep(0.5)
                    continue
                self._run_ntrip_session(deadline)
        finally:
            self.close()

    def _prime_nmea(self) -> None:
        start = time.monotonic()
        while self.latest_gga is None and time.monotonic() - start < 2.0:
            self._read_and_track_serial_line()

    def _run_ntrip_session(self, deadline: float | None) -> None:
        self.logger.info(
            "Connecting to NTRIP caster. server=%s mountpoint=%s",
            self.config.ntrip_server,
            self.config.ntrip_mountpoint,
        )
        sock, header = self._connect_ntrip_with_fallback()
        try:
            self.logger.info("NTRIP stream opened successfully. response=%r", header.strip())
            # Many casters expect the first GGA after they accept the stream.
            if self.latest_gga is not None:
                sock.sendall((self._gga_request_body()).encode("ascii"))
                self.last_gga_sent_monotonic = time.monotonic()
                self.logger.info("[NTRIP] Sent initial GGA to caster.")
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                self._pump_serial_nmea()
                self._maybe_send_periodic_gga(sock)
                message = self._read_rtcm_message(sock)
                if message is None:
                    raise RuntimeError("NTRIP stream ended or returned empty data.")
                if self.config.write_rtcm_to_serial and self.serial is not None:
                    self.serial.write(message)
                    self.serial.flush()
                self.rtcm_messages += 1
                self.rtcm_bytes += len(message)
                message_type = _rtcm_message_type(message)
                self.logger.info(
                    "[RTCM] count=%d bytes=%d type=%s total_bytes=%d",
                    self.rtcm_messages,
                    len(message),
                    message_type,
                    self.rtcm_bytes,
                )
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _connect_ntrip_with_fallback(self) -> tuple[socket.socket, str]:
        attempts = [
            ("ntrip_v2", self._build_request_v2()),
            ("ntrip_v1", self._build_request_v1()),
        ]
        errors: list[str] = []
        for label, request in attempts:
            sock = self._open_ntrip_socket()
            try:
                self.logger.info("[NTRIP] Trying request style=%s", label)
                sock.sendall(request.encode("ascii"))
                header = self._read_ntrip_header(sock)
                if "200 OK" in header or "ICY 200 OK" in header:
                    return sock, header
                errors.append(f"{label}: {header!r}")
                try:
                    sock.close()
                except Exception:
                    pass
            finally:
                pass
        raise RuntimeError("NTRIP request failed for all request styles. responses=\n- " + "\n- ".join(errors))

    def _pump_serial_nmea(self) -> None:
        for _ in range(4):
            self._read_and_track_serial_line()

    def _read_and_track_serial_line(self) -> None:
        if self.serial is None:
            return
        raw_line = self.serial.readline()
        if not raw_line:
            return
        line = raw_line.decode("ascii", errors="ignore").strip()
        if not line.startswith("$"):
            return
        if not _checksum_matches(line):
            return
        if line.startswith("$G") and ("GGA" in line or "RMC" in line):
            if "GGA" in line:
                self.latest_gga = line
                parsed = _parse_gga_fields(line)
                if self.config.log_nmea and parsed is not None:
                    self.logger.info(
                        "[NMEA] utc=%s lat=%s lon=%s fix=%s sats=%s hdop=%s alt_m=%s",
                        parsed.get("nmea_utc"),
                        parsed.get("latitude_deg"),
                        parsed.get("longitude_deg"),
                        parsed.get("fix_quality"),
                        parsed.get("num_satellites"),
                        parsed.get("hdop"),
                        parsed.get("altitude_m"),
                    )
            elif "RMC" in line:
                self.latest_rmc = line

    def _maybe_send_periodic_gga(self, sock: socket.socket) -> None:
        if self.latest_gga is None:
            return
        now = time.monotonic()
        if now - self.last_gga_sent_monotonic < self.config.gga_interval_seconds:
            return
        sock.sendall((self._gga_request_body()).encode("ascii"))
        self.last_gga_sent_monotonic = now
        self.logger.info("[NTRIP] Sent periodic GGA to caster.")

    def _gga_request_body(self) -> str:
        if self.latest_gga is None:
            return ""
        return self.latest_gga + "\r\n"

    def _read_rtcm_message(self, sock: socket.socket) -> bytes | None:
        header = self._safe_recv(sock, 1)
        if not header:
            return None
        while header and header[0] != 0xD3:
            header = self._safe_recv(sock, 1)
        if not header:
            return None

        length_bytes = self._safe_recv(sock, 2)
        if len(length_bytes) < 2:
            return None
        payload_length = ((length_bytes[0] & 0x03) << 8) | length_bytes[1]
        payload = self._safe_recv(sock, payload_length + 3)
        if len(payload) < payload_length + 3:
            return None
        return header + length_bytes + payload

    def _open_ntrip_socket(self) -> socket.socket:
        host, port = _split_host_port(self.config.ntrip_server)
        sock = socket.create_connection((host, port), timeout=10)
        sock.settimeout(10)
        return sock

    def _build_request_v2(self) -> str:
        host, port = _split_host_port(self.config.ntrip_server)
        auth = b64encode(f"{self.config.ntrip_user}:{self.config.ntrip_pass}".encode("utf-8")).decode("ascii")
        return (
            f"GET /{self.config.ntrip_mountpoint} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Ntrip-Version: {self.config.ntrip_version}\r\n"
            f"User-Agent: NTRIP underwater_acquisition_kit\r\n"
            f"Authorization: Basic {auth}\r\n"
            f"Accept: */*\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )

    def _build_request_v1(self) -> str:
        auth = b64encode(f"{self.config.ntrip_user}:{self.config.ntrip_pass}".encode("utf-8")).decode("ascii")
        return (
            f"GET /{self.config.ntrip_mountpoint} HTTP/1.0\r\n"
            f"User-Agent: NTRIP underwater_acquisition_kit\r\n"
            f"Ntrip-Version: {self.config.ntrip_version}\r\n"
            f"Authorization: Basic {auth}\r\n"
            f"Accept: */*\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )

    def _read_ntrip_header(self, sock: socket.socket) -> str:
        data = b""
        try:
            while b"\r\n\r\n" not in data and len(data) < 8192:
                chunk = sock.recv(1)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            pass
        return data.decode("latin1", errors="replace")

    @staticmethod
    def _safe_recv(sock: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)


def _optional_bool(value, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def _parse_gga_fields(sentence: str) -> dict[str, object] | None:
    body = sentence[1:].split("*", 1)[0]
    fields = body.split(",")
    if len(fields) < 10 or not fields[0].endswith("GGA"):
        return None
    return {
        "nmea_utc": fields[1] or None,
        "latitude_deg": _parse_lat_lon(fields[2], fields[3]),
        "longitude_deg": _parse_lat_lon(fields[4], fields[5]),
        "fix_quality": _as_int(fields[6]),
        "num_satellites": _as_int(fields[7]),
        "hdop": _as_float(fields[8]),
        "altitude_m": _as_float(fields[9]),
    }


def _parse_lat_lon(raw_value: str, hemisphere: str) -> float | None:
    if not raw_value or not hemisphere:
        return None
    try:
        numeric = float(raw_value)
    except ValueError:
        return None
    degrees = int(numeric // 100)
    minutes = numeric - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemisphere in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rtcm_message_type(message: bytes) -> int | None:
    if len(message) < 6:
        return None
    return ((message[3] << 8) | message[4]) >> 4


def _split_host_port(value: str) -> tuple[str, int]:
    if ":" not in value:
        return value, 2101
    host, port_text = value.rsplit(":", 1)
    return host, int(port_text)


def validate_config(config: RtkConfig) -> None:
    missing = []
    if not config.serial_port:
        missing.append("rtk.serial_port")
    if not config.ntrip_server:
        missing.append("rtk.ntrip_server")
    if not config.ntrip_user:
        missing.append("rtk.ntrip_user")
    if not config.ntrip_pass:
        missing.append("rtk.ntrip_pass")
    if not config.ntrip_mountpoint:
        missing.append("rtk.ntrip_mountpoint")
    if missing:
        raise RuntimeError("Missing required RTK config values:\n- " + "\n- ".join(missing))


def main() -> int:
    args = parse_args()
    logger = get_app_logger("rtk_ntrip_bridge", PROJECT_ROOT / "logs")
    config_path = PROJECT_ROOT / "configs" / "rtk.yaml"

    try:
        raw_config = apply_overrides(load_yaml_config(config_path), args)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        config = load_rtk_config(raw_config)
        validate_config(config)

        bridge = RtkBridge(config, logger)
        logger.info(
            "Starting RTK bridge. started_at=%s serial_port=%s ntrip_server=%s mountpoint=%s",
            datetime.now(timezone.utc).isoformat(),
            config.serial_port,
            config.ntrip_server,
            config.ntrip_mountpoint,
        )
        logger.info("RTK is active only when GGA fix quality later rises to 4 or 5.")
        bridge.run(args.duration)
        logger.info(
            "RTK bridge finished. rtcm_messages=%d total_rtcm_bytes=%d",
            bridge.rtcm_messages,
            bridge.rtcm_bytes,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("RTK bridge interrupted by user.")
        return 0
    except FileNotFoundError:
        logger.exception("RTK config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception("RTK bridge failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
