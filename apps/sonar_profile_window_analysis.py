from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sonar_profile_analysis import (  # noqa: E402
    ProfileSample,
    Peak,
    extract_peaks,
    index_to_distance_mm,
    load_profile_samples,
    rule_argmax,
    rule_front_valid_peak,
    rule_prominence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare profile peaks across selected timestamp windows in sonar_profile.jsonl."
    )
    parser.add_argument(
        "--profile-path",
        default=str(PROJECT_ROOT / "data" / "sonar_profile.jsonl"),
        help="Path to sonar_profile.jsonl. Defaults to data/sonar_profile.jsonl.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Window spec as label:start:end using Unix seconds. Repeat for multiple windows.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print machine-readable JSON after the human summary.",
    )
    return parser.parse_args()


def mean_sample(label: str, samples: list[ProfileSample]) -> dict[str, object]:
    profile_len = min(len(sample.profile_data) for sample in samples)
    mean_profile = [
        sum(sample.profile_data[index] for sample in samples) / len(samples)
        for index in range(profile_len)
    ]
    template = samples[0]
    aggregate = ProfileSample(
        line_index=-1,
        timestamp=sum(sample.timestamp for sample in samples) / len(samples),
        distance_mm=mean_or_none([sample.distance_mm for sample in samples]),
        confidence=mean_or_none([sample.confidence for sample in samples]),
        scan_start_mm=template.scan_start_mm,
        scan_length_mm=template.scan_length_mm,
        gain_setting=template.gain_setting,
        mode_auto=template.mode_auto,
        transmit_duration_us=template.transmit_duration_us,
        ping_number=None,
        profile_data=mean_profile,
    )

    strongest_index = max(range(profile_len), key=lambda idx: mean_profile[idx])
    strongest_peak = Peak(
        index=strongest_index,
        value=mean_profile[strongest_index],
        prominence=0.0,
        distance_mm=index_to_distance_mm(aggregate, strongest_index),
    )

    return {
        "label": label,
        "count": len(samples),
        "timestamp_start": min(sample.timestamp for sample in samples),
        "timestamp_end": max(sample.timestamp for sample in samples),
        "distance_mm_mean": aggregate.distance_mm,
        "distance_mm_min": min(sample.distance_mm for sample in samples if sample.distance_mm is not None),
        "distance_mm_max": max(sample.distance_mm for sample in samples if sample.distance_mm is not None),
        "confidence_mean": aggregate.confidence,
        "scan_start_mm": aggregate.scan_start_mm,
        "scan_length_mm": aggregate.scan_length_mm,
        "bin_size_mm": None
        if aggregate.scan_length_mm is None
        else aggregate.scan_length_mm / max(len(aggregate.profile_data), 1),
        "mean_profile_head": [round(value, 2) for value in mean_profile[:30]],
        "strongest_peak_index": strongest_peak.index,
        "strongest_peak_distance_mm": strongest_peak.distance_mm,
        "strongest_peak_value": round(strongest_peak.value, 2),
        "prominence_peak": serialize_rule(rule_prominence(aggregate)),
        "front_valid_peak": serialize_rule(rule_front_valid_peak(aggregate)),
        "argmax_exclude_0": serialize_rule(rule_argmax(aggregate, exclusion_mm=0.0)),
        "argmax_exclude_50": serialize_rule(rule_argmax(aggregate, exclusion_mm=50.0)),
        "top_peaks": serialize_peaks(extract_peaks(aggregate)[:8]),
    }


def serialize_rule(result: object) -> dict[str, object]:
    peak = getattr(result, "peak", None)
    return {
        "estimated_distance_mm": getattr(result, "estimated_distance_mm", None),
        "peak_index": None if peak is None else peak.index,
        "peak_value": None if peak is None else round(peak.value, 2),
        "peak_prominence": None if peak is None else round(peak.prominence, 2),
        "peak_distance_mm": None if peak is None else peak.distance_mm,
    }


def serialize_peaks(peaks: list[Peak]) -> list[dict[str, object]]:
    ordered = sorted(peaks, key=lambda peak: peak.value, reverse=True)
    return [
        {
            "index": peak.index,
            "value": round(peak.value, 2),
            "prominence": round(peak.prominence, 2),
            "distance_mm": peak.distance_mm,
        }
        for peak in ordered
    ]


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def parse_window_spec(spec: str) -> tuple[str, float, float]:
    label, start_raw, end_raw = spec.split(":", maxsplit=2)
    start = float(start_raw)
    end = float(end_raw)
    if end < start:
        raise ValueError(f"Window end must be >= start: {spec}")
    return label, start, end


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile_path)
    if not profile_path.is_absolute():
        profile_path = (PROJECT_ROOT / profile_path).resolve()

    samples = load_profile_samples(profile_path)
    if not samples:
        raise SystemExit(f"No usable profile samples found in {profile_path}")

    if not args.window:
        raise SystemExit("Pass at least one --window label:start:end argument.")

    summaries: list[dict[str, object]] = []
    for spec in args.window:
        label, start, end = parse_window_spec(spec)
        subset = [sample for sample in samples if start <= sample.timestamp <= end]
        if not subset:
            print(f"[{label}] no samples in window {start}..{end}")
            continue
        summary = mean_sample(label, subset)
        summaries.append(summary)
        print(f"[{label}] samples={summary['count']} distance_mean={summary['distance_mm_mean']:.2f} mm")
        print(
            f"  range={summary['distance_mm_min']}..{summary['distance_mm_max']} mm "
            f"bin_size={summary['bin_size_mm']:.2f} mm"
        )
        print(
            f"  strongest_peak=index {summary['strongest_peak_index']} "
            f"distance {summary['strongest_peak_distance_mm']:.2f} mm "
            f"value {summary['strongest_peak_value']}"
        )
        front_valid = summary["front_valid_peak"]
        print(
            f"  front_valid_peak=index {front_valid['peak_index']} "
            f"distance {front_valid['peak_distance_mm']:.2f} mm "
            f"value {front_valid['peak_value']}"
        )
        print(f"  mean_profile_head={summary['mean_profile_head']}")

    if args.dump_json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
