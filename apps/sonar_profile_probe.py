from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.sonar.ping_logger import PingSonarClient, load_sonar_config
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Ping Sonar profile_data availability.")
    parser.add_argument("--attempts", type=int, default=10, help="Number of profile read attempts.")
    parser.add_argument("--interval", type=float, default=0.25, help="Seconds between attempts.")
    parser.add_argument(
        "--save-first",
        action="store_true",
        help="Save the first successful profile payload to data/profile_probe/<timestamp>_profile.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / "sonar.yaml"
    logger = get_app_logger("sonar_profile_probe", PROJECT_ROOT / "logs")
    client: PingSonarClient | None = None

    try:
        raw_config = load_yaml_config(config_path)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        sonar_config = load_sonar_config(raw_config)
        client = PingSonarClient(sonar_config, logger=logger)
        first_record = client.connect_and_validate()
        device = client._device  # noqa: SLF001 - intentional probe access for debugging current brping path.

        has_get_profile = callable(getattr(device, "get_profile", None))
        logger.info(
            "Profile probe connected. first_distance_mm=%s first_confidence=%s has_get_profile=%s",
            first_record.distance_mm,
            first_record.confidence,
            has_get_profile,
        )
        print(f"has_get_profile: {has_get_profile}")

        first_saved = False
        profile_success_count = 0
        for attempt in range(1, args.attempts + 1):
            record = client.read_record()
            has_profile = record.profile_data is not None
            if has_profile:
                profile_success_count += 1

            print("-" * 60)
            print(f"attempt        : {attempt}/{args.attempts}")
            print(f"timestamp      : {record.timestamp}")
            print(f"distance_mm    : {record.distance_mm}")
            print(f"confidence     : {record.confidence}")
            print(f"valid          : {record.valid}")
            print(f"ping_number    : {record.ping_number}")
            print(f"profile?       : {has_profile}")
            print(f"profile_len    : {len(record.profile_data) if record.profile_data is not None else None}")
            print(f"profile_head   : {record.profile_data[:20] if record.profile_data is not None else None}")

            if has_profile and args.save_first and not first_saved:
                output_dir = PROJECT_ROOT / "data" / "profile_probe"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_profile.json"
                output_path.write_text(
                    json.dumps(
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
                            "profile_data": record.profile_data,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"saved_first_profile: {output_path}")
                first_saved = True

            if attempt < args.attempts:
                time.sleep(args.interval)

        print("=" * 60)
        print(f"profile_success_count: {profile_success_count}")
        if profile_success_count == 0:
            print("result: profile_data was not returned in this probe run")
        else:
            print("result: profile_data returned successfully")
        return 0
    except FileNotFoundError:
        logger.exception("Sonar config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception("Sonar profile probe failed: %s", exc)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
