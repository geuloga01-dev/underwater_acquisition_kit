from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.sonar.ping_logger import PingSonarClient, SonarRecord, load_sonar_config, prepare_sonar
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short sonar experiment data capture.")
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--pose", default="")
    parser.add_argument("--offset-cm", type=float, default=0.0)
    parser.add_argument("--true-distance-mm", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--sample-interval", type=float, default=None)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def build_output_path(label: str, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = slugify(label)
    return PROJECT_ROOT / "data" / "experiments" / f"{timestamp}_{safe_label}.csv"


def slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "experiment"


def write_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_idx",
                "timestamp_iso",
                "unix_time",
                "distance_mm",
                "confidence",
                "label",
                "pose",
                "offset_cm",
                "true_distance_mm",
            ]
        )


def append_row(
    csv_path: Path,
    sample_idx: int,
    record: SonarRecord,
    label: str,
    pose: str,
    offset_cm: float,
    true_distance_mm: float,
) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                sample_idx,
                record.timestamp_iso,
                record.unix_time,
                record.distance_mm,
                record.confidence,
                label,
                pose,
                offset_cm,
                true_distance_mm,
            ]
        )


def summarize_records(records: list[SonarRecord]) -> dict[str, float | int | None]:
    distances = [record.distance_mm for record in records if record.distance_mm is not None]
    confidences = [record.confidence for record in records if record.confidence is not None]
    unix_times = [record.unix_time for record in records]

    effective_hz: float | None = None
    if len(unix_times) >= 2:
        deltas = [curr - prev for prev, curr in zip(unix_times, unix_times[1:]) if curr > prev]
        if deltas:
            mean_delta = statistics.mean(deltas)
            if mean_delta > 0:
                effective_hz = 1.0 / mean_delta

    return {
        "count": len(records),
        "mean_distance_mm": statistics.mean(distances) if distances else None,
        "std_distance_mm": statistics.stdev(distances) if len(distances) >= 2 else 0.0 if distances else None,
        "min_distance_mm": min(distances) if distances else None,
        "max_distance_mm": max(distances) if distances else None,
        "mean_confidence": statistics.mean(confidences) if confidences else None,
        "effective_hz": effective_hz,
    }


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / "sonar.yaml"
    logger = get_app_logger("sonar_quick_test", PROJECT_ROOT / "logs")
    client: PingSonarClient | None = None

    try:
        raw_config = load_yaml_config(config_path)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        sonar_config = load_sonar_config(raw_config)
        if args.sample_interval is not None:
            sonar_config.sample_interval = args.sample_interval

        output_path = build_output_path(args.label, args.out)
        write_header(output_path)

        client = PingSonarClient(sonar_config, logger=logger)
        first_record = prepare_sonar(client)
        logger.info(
            "Sonar quick test ready. first_distance_mm=%s first_confidence=%s output=%s",
            first_record.distance_mm,
            first_record.confidence,
            output_path,
        )

        records: list[SonarRecord] = []
        for sample_idx in range(1, args.samples + 1):
            record = client.read_record()
            records.append(record)
            append_row(
                output_path,
                sample_idx,
                record,
                args.label,
                args.pose,
                args.offset_cm,
                args.true_distance_mm,
            )
            logger.info(
                "sample=%d/%d distance_mm=%s confidence=%s unix_time=%.6f",
                sample_idx,
                args.samples,
                record.distance_mm,
                record.confidence,
                record.unix_time,
            )

            if sonar_config.sample_interval > 0 and sample_idx < args.samples:
                time.sleep(sonar_config.sample_interval)

        summary = summarize_records(records)
        print(f"count: {summary['count']}")
        print(f"mean_distance_mm: {summary['mean_distance_mm']}")
        print(f"std_distance_mm: {summary['std_distance_mm']}")
        print(f"min_distance_mm: {summary['min_distance_mm']}")
        print(f"max_distance_mm: {summary['max_distance_mm']}")
        print(f"mean_confidence: {summary['mean_confidence']}")
        print(f"effective_hz: {summary['effective_hz']}")
        return 0
    except FileNotFoundError:
        logger.exception("Sonar config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception("Sonar quick test failed: %s", exc)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
