from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.sonar.ping_logger import PingSonarClient, load_sonar_config
from src.telemetry.attitude_listener import load_attitude_config, normalize_attitude_message
from src.telemetry.battery_listener import load_battery_config, normalize_battery_message
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(*configs: dict) -> int:
    for raw_config in configs:
        level_name = raw_config.get("logging", {}).get("level")
        if level_name:
            return getattr(logging, str(level_name).upper(), logging.INFO)
    return logging.INFO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Jetson sensor sanity check for sonar, battery, and ATTITUDE.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means until Ctrl+C.")
    parser.add_argument("--no-sonar", action="store_true", help="Skip sonar live check.")
    parser.add_argument("--no-pixhawk", action="store_true", help="Skip battery/ATTITUDE live check.")
    return parser.parse_args()


def run_sonar_loop(raw_config: dict, logger: logging.Logger, stop_event: threading.Event) -> None:
    client: PingSonarClient | None = None
    try:
        sonar_config = load_sonar_config(raw_config)
        client = PingSonarClient(sonar_config, logger=logger)
        first_record = client.connect_and_validate()
        logger.info(
            "[SONAR] ready distance_mm=%s confidence=%s valid=%s profile=%s",
            first_record.distance_mm,
            first_record.confidence,
            first_record.valid,
            "yes" if first_record.profile_data is not None else "no",
        )

        while not stop_event.is_set():
            record = client.read_record()
            logger.info(
                "[SONAR] t=%.3f distance_mm=%s confidence=%s valid=%s ping_number=%s profile=%s",
                record.timestamp,
                record.distance_mm,
                record.confidence,
                record.valid,
                record.ping_number,
                "yes" if record.profile_data is not None else "no",
            )
            stop_event.wait(sonar_config.sample_interval)
    except Exception as exc:
        logger.warning("[SONAR] live check failed: %s", exc)
    finally:
        if client is not None:
            client.close()


def run_pixhawk_loop(battery_raw: dict, imu_raw: dict, logger: logging.Logger, stop_event: threading.Event) -> None:
    connection = None
    try:
        from pymavlink import mavutil
    except ModuleNotFoundError as exc:
        logger.warning("[PIXHAWK] pymavlink is not installed: %s", exc)
        return

    try:
        battery_config = load_battery_config(battery_raw)
        attitude_config = load_attitude_config(imu_raw)
        connection = mavutil.mavlink_connection(battery_config.port, baud=battery_config.baudrate)
        logger.info("[PIXHAWK] connected port=%s baudrate=%s", battery_config.port, battery_config.baudrate)
        if battery_config.wait_heartbeat:
            connection.wait_heartbeat(timeout=battery_config.heartbeat_timeout)
            logger.info("[PIXHAWK] heartbeat received")

        timeout_seconds = max(0.1, battery_config.poll_interval, attitude_config.timeout_seconds)
        while not stop_event.is_set():
            message = connection.recv_match(type=["BATTERY_STATUS", "ATTITUDE"], blocking=True, timeout=timeout_seconds)
            if message is None:
                continue

            message_type = message.get_type()
            if message_type == "BATTERY_STATUS":
                record = normalize_battery_message(message)
                logger.info(
                    "[BATTERY] t=%.3f voltage_v=%s current_a=%s remaining_percent=%s temp_c=%s",
                    record.timestamp,
                    record.voltage_v,
                    record.current_a,
                    record.remaining_percent,
                    record.battery_temp_c,
                )
            elif message_type == "ATTITUDE":
                record = normalize_attitude_message(message, time.time())
                logger.info(
                    "[ATTITUDE] t=%.3f roll=%.4f pitch=%.4f yaw=%.4f",
                    record.timestamp,
                    record.roll if record.roll is not None else float("nan"),
                    record.pitch if record.pitch is not None else float("nan"),
                    record.yaw if record.yaw is not None else float("nan"),
                )
    except Exception as exc:
        logger.warning("[PIXHAWK] live check failed: %s", exc)
    finally:
        if connection is not None:
            close_method = getattr(connection, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass


def main() -> int:
    args = parse_args()
    logger = get_app_logger("live_sensor_check", PROJECT_ROOT / "logs")

    try:
        camera_raw = load_yaml_config(PROJECT_ROOT / "configs" / "camera.yaml")
        sonar_raw = load_yaml_config(PROJECT_ROOT / "configs" / "sonar.yaml")
        battery_raw = load_yaml_config(PROJECT_ROOT / "configs" / "battery.yaml")
        imu_raw = load_yaml_config(PROJECT_ROOT / "configs" / "imu.yaml")

        level = resolve_log_level(camera_raw, sonar_raw, battery_raw, imu_raw)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

        stop_event = threading.Event()
        threads: list[threading.Thread] = []

        if not args.no_sonar:
            threads.append(
                threading.Thread(
                    target=run_sonar_loop,
                    args=(sonar_raw, logger, stop_event),
                    name="live-sonar-check",
                    daemon=True,
                )
            )
        if not args.no_pixhawk:
            threads.append(
                threading.Thread(
                    target=run_pixhawk_loop,
                    args=(battery_raw, imu_raw, logger, stop_event),
                    name="live-pixhawk-check",
                    daemon=True,
                )
            )

        if not threads:
            logger.warning("Nothing selected to run. Remove --no-sonar or --no-pixhawk.")
            return 1

        logger.info(
            "Starting live sensor check. duration_seconds=%s sonar=%s pixhawk=%s",
            args.duration,
            not args.no_sonar,
            not args.no_pixhawk,
        )
        logger.info("Move the Pixhawk/Jetson gently to verify roll/pitch/yaw changes and move a target to verify sonar distance changes.")

        for thread in threads:
            thread.start()

        if args.duration > 0:
            stop_event.wait(args.duration)
            stop_event.set()
        else:
            while not stop_event.is_set():
                time.sleep(0.2)

        for thread in threads:
            thread.join(timeout=3.0)

        logger.info("Live sensor check finished.")
        return 0
    except KeyboardInterrupt:
        logger.info("Live sensor check interrupted by user.")
        return 0
    except Exception as exc:
        logger.exception("Live sensor check failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
