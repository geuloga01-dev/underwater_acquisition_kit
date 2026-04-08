from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.telemetry.gps_listener import GpsListener, GpsRecord, _is_recoverable_serial_error, load_gps_config
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live GPS NMEA sanity check.")
    parser.add_argument("--port", type=str, default="", help="Override GPS serial port.")
    parser.add_argument("--baudrate", type=int, default=0, help="Override GPS baudrate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means until Ctrl+C.")
    parser.add_argument("--show-raw", action="store_true", help="Also print sentence type and NMEA UTC.")
    return parser.parse_args()


def apply_overrides(raw_config: dict, args: argparse.Namespace) -> dict:
    gps_section = raw_config.setdefault("gps", {})
    if args.port:
        gps_section["port"] = args.port
    if args.baudrate:
        gps_section["baudrate"] = args.baudrate
    gps_section["enabled"] = True
    return raw_config


def format_record(record: GpsRecord, show_raw: bool) -> str:
    parts = [
        f"t={record.timestamp:.3f}",
        f"lat={record.latitude_deg}",
        f"lon={record.longitude_deg}",
        f"alt_m={record.altitude_m}",
        f"fix={record.fix_quality}",
        f"sats={record.num_satellites}",
        f"hdop={record.hdop}",
    ]
    if record.speed_knots is not None:
        parts.append(f"speed_knots={record.speed_knots}")
    if record.track_deg is not None:
        parts.append(f"track_deg={record.track_deg}")
    if show_raw:
        parts.append(f"sentence={record.sentence_type}")
        parts.append(f"nmea_utc={record.nmea_utc}")
    return " ".join(parts)


def inspect_serial_stream(listener: GpsListener) -> str:
    serial_handle = getattr(listener, "_serial", None)
    if serial_handle is None:
        return "serial not connected"

    waiting = getattr(serial_handle, "in_waiting", 0)
    if waiting <= 0:
        return "no bytes received yet"

    sample = serial_handle.read(min(waiting, 64))
    if not sample:
        return "bytes expected but none were read"

    if b"$G" in sample or b"$P" in sample:
        return "NMEA bytes are present but no full GGA/RMC sentence was parsed yet"

    printable = sum(32 <= byte <= 126 or byte in (9, 10, 13) for byte in sample)
    if printable < max(4, len(sample) // 3):
        return "non-NMEA binary bytes detected, likely UBX/RTCM output on this port"

    preview = sample.decode("ascii", errors="replace").strip().replace("\n", "\\n")
    if len(preview) > 48:
        preview = preview[:48] + "..."
    return f"unexpected ASCII bytes seen: {preview or '<empty>'}"


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / "gps.yaml"
    logger = get_app_logger("gps_live_check", PROJECT_ROOT / "logs")
    listener: GpsListener | None = None

    try:
        raw_config = apply_overrides(load_yaml_config(config_path), args)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        gps_config = load_gps_config(raw_config)
        listener = GpsListener(gps_config, logger=logger)
        listener.connect()

        logger.info(
            "Starting live GPS check. port=%s baudrate=%s duration_seconds=%s",
            gps_config.port,
            gps_config.baudrate,
            args.duration,
        )
        logger.info("Waiting for NMEA-derived fixes. Indoors this may stay fix=0 with no satellites.")

        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        sample_count = 0
        last_diagnostic_time = time.monotonic()
        last_diagnostic_message: str | None = None
        last_record_time: float | None = None
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break

            try:
                record = listener.read_record(include_partial=True)
            except Exception as exc:
                if not _is_recoverable_serial_error(exc):
                    raise
                logger.warning("GPS serial read hiccup, reconnecting: %s", exc)
                listener.reconnect()
                last_diagnostic_time = time.monotonic()
                last_diagnostic_message = None
                continue
            if record is None:
                now = time.monotonic()
                if last_record_time is not None and now - last_record_time < 10.0:
                    continue
                if now - last_diagnostic_time >= 10.0:
                    diagnostic = inspect_serial_stream(listener)
                    if diagnostic != last_diagnostic_message:
                        logger.warning("No parsed GPS record yet: %s", diagnostic)
                        last_diagnostic_message = diagnostic
                    last_diagnostic_time = now
                continue

            sample_count += 1
            last_record_time = time.monotonic()
            last_diagnostic_time = time.monotonic()
            last_diagnostic_message = None
            logger.info("%s", format_record(record, args.show_raw))

        logger.info("Live GPS check finished. samples=%d", sample_count)
        return 0
    except KeyboardInterrupt:
        logger.info("Live GPS check interrupted by user.")
        return 0
    except FileNotFoundError:
        logger.exception("GPS config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception("Live GPS check failed: %s", exc)
        return 1
    finally:
        if listener is not None:
            listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
