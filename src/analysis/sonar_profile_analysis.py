from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import statistics
from typing import Any


@dataclass(slots=True)
class ProfileSample:
    line_index: int
    timestamp: float
    distance_mm: float | None
    confidence: float | None
    scan_start_mm: float | None
    scan_length_mm: float | None
    gain_setting: float | None
    mode_auto: bool | None
    transmit_duration_us: float | None
    ping_number: float | None
    profile_data: list[float]


@dataclass(slots=True)
class Peak:
    index: int
    value: float
    prominence: float
    distance_mm: float | None


@dataclass(slots=True)
class RuleResult:
    estimated_distance_mm: float | None
    error_mm: float | None
    peak: Peak | None


def analyze_session(
    session_root: Path,
    output_dir: Path,
    logger: logging.Logger,
    exclusion_mm_values: list[float],
    representative_count: int = 5,
) -> dict[str, Any]:
    profile_path = session_root / "sonar" / "sonar_profile.jsonl"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_profile_samples(profile_path)
    if not samples:
        raise RuntimeError(f"No profile samples found in {profile_path}")

    rows: list[dict[str, Any]] = []
    exclusion_summaries: list[dict[str, Any]] = []
    best_exclusion = None
    best_mae = None

    for sample in samples:
        peaks = extract_peaks(sample)
        strongest_peak = max(peaks, key=lambda peak: peak.value, default=None)
        second_peak = sorted(peaks, key=lambda peak: peak.value, reverse=True)[1] if len(peaks) > 1 else None

        rule_a = rule_argmax(sample, exclusion_mm=0.0)
        rule_c = rule_prominence(sample)
        rule_d = rule_front_valid_peak(sample)

        row: dict[str, Any] = {
            "line_index": sample.line_index,
            "timestamp": sample.timestamp,
            "distance_mm": sample.distance_mm,
            "confidence": sample.confidence,
            "peak_count": len(peaks),
            "strongest_peak_distance_mm": strongest_peak.distance_mm if strongest_peak else None,
            "strongest_peak_value": strongest_peak.value if strongest_peak else None,
            "strongest_peak_prominence": strongest_peak.prominence if strongest_peak else None,
            "second_peak_distance_mm": second_peak.distance_mm if second_peak else None,
            "second_peak_value": second_peak.value if second_peak else None,
            "rule_a_distance_mm": rule_a.estimated_distance_mm,
            "rule_a_error_mm": rule_a.error_mm,
            "rule_c_distance_mm": rule_c.estimated_distance_mm,
            "rule_c_error_mm": rule_c.error_mm,
            "rule_d_distance_mm": rule_d.estimated_distance_mm,
            "rule_d_error_mm": rule_d.error_mm,
        }

        for exclusion_mm in exclusion_mm_values:
            result = rule_argmax(sample, exclusion_mm=exclusion_mm)
            key_prefix = exclusion_key(exclusion_mm)
            row[f"rule_b_{key_prefix}_distance_mm"] = result.estimated_distance_mm
            row[f"rule_b_{key_prefix}_error_mm"] = result.error_mm

        rows.append(row)

    for exclusion_mm in exclusion_mm_values:
        key_prefix = exclusion_key(exclusion_mm)
        errors = [abs(row[f"rule_b_{key_prefix}_error_mm"]) for row in rows if row[f"rule_b_{key_prefix}_error_mm"] is not None]
        mae = statistics.mean(errors) if errors else None
        summary = {
            "exclusion_mm": exclusion_mm,
            "count": len(errors),
            "mae_mm": mae,
        }
        exclusion_summaries.append(summary)
        if mae is not None and (best_mae is None or mae < best_mae):
            best_mae = mae
            best_exclusion = exclusion_mm

    summary = build_summary(rows, exclusion_summaries, best_exclusion)
    save_csv(output_dir / "rule_comparison.csv", rows)
    create_figures(output_dir, rows, samples, exclusion_summaries, best_exclusion)
    create_report(output_dir / "report.md", session_root, summary, rows, exclusion_summaries, representative_count)

    logger.info("Sonar profile analysis finished. samples=%d output_dir=%s", len(samples), output_dir)
    return summary


def load_profile_samples(profile_path: Path) -> list[ProfileSample]:
    samples: list[ProfileSample] = []
    with profile_path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            profile_data_raw = payload.get("profile_data") or []
            if not isinstance(profile_data_raw, list) or not profile_data_raw:
                continue
            samples.append(
                ProfileSample(
                    line_index=line_index,
                    timestamp=float(payload.get("timestamp")),
                    distance_mm=_as_float(payload.get("distance_mm")),
                    confidence=_as_float(payload.get("confidence")),
                    scan_start_mm=_as_float(payload.get("scan_start_mm")),
                    scan_length_mm=_as_float(payload.get("scan_length_mm")),
                    gain_setting=_as_float(payload.get("gain_setting")),
                    mode_auto=_as_bool(payload.get("mode_auto")),
                    transmit_duration_us=_as_float(payload.get("transmit_duration_us")),
                    ping_number=_as_float(payload.get("ping_number")),
                    profile_data=[float(value) for value in profile_data_raw],
                )
            )
    return samples


def extract_peaks(sample: ProfileSample) -> list[Peak]:
    values = sample.profile_data
    peaks: list[Peak] = []
    for index in range(1, len(values) - 1):
        current = values[index]
        if current >= values[index - 1] and current > values[index + 1]:
            prominence = compute_prominence(values, index)
            peaks.append(
                Peak(
                    index=index,
                    value=current,
                    prominence=prominence,
                    distance_mm=index_to_distance_mm(sample, index),
                )
            )
    if len(values) >= 1 and not peaks:
        best_index = max(range(len(values)), key=lambda idx: values[idx])
        peaks.append(
            Peak(
                index=best_index,
                value=values[best_index],
                prominence=0.0,
                distance_mm=index_to_distance_mm(sample, best_index),
            )
        )
    return peaks


def compute_prominence(values: list[float], index: int) -> float:
    current = values[index]
    left_min = min(values[: index + 1]) if index >= 0 else current
    right_min = min(values[index:]) if index < len(values) else current
    baseline = max(left_min, right_min)
    return current - baseline


def index_to_distance_mm(sample: ProfileSample, index: int) -> float | None:
    if sample.scan_start_mm is None or sample.scan_length_mm is None or not sample.profile_data:
        return None
    step = sample.scan_length_mm / max(len(sample.profile_data), 1)
    return sample.scan_start_mm + (index + 0.5) * step


def rule_argmax(sample: ProfileSample, exclusion_mm: float) -> RuleResult:
    start_index = exclusion_index(sample, exclusion_mm)
    values = sample.profile_data[start_index:]
    if not values:
        return RuleResult(None, None, None)
    best_offset = max(range(len(values)), key=lambda idx: values[idx])
    peak_index = start_index + best_offset
    peak = Peak(
        index=peak_index,
        value=sample.profile_data[peak_index],
        prominence=compute_prominence(sample.profile_data, peak_index),
        distance_mm=index_to_distance_mm(sample, peak_index),
    )
    return RuleResult(peak.distance_mm, compute_error_mm(peak.distance_mm, sample.distance_mm), peak)


def rule_prominence(sample: ProfileSample) -> RuleResult:
    peaks = extract_peaks(sample)
    if not peaks:
        return RuleResult(None, None, None)
    peak = max(peaks, key=lambda item: item.prominence)
    return RuleResult(peak.distance_mm, compute_error_mm(peak.distance_mm, sample.distance_mm), peak)


def rule_front_valid_peak(sample: ProfileSample) -> RuleResult:
    peaks = extract_peaks(sample)
    if not peaks:
        return RuleResult(None, None, None)
    max_value = max(peak.value for peak in peaks)
    threshold = max_value * 0.55
    for peak in sorted(peaks, key=lambda item: item.index):
        if peak.value >= threshold and peak.prominence >= 0.2 * max_value:
            return RuleResult(peak.distance_mm, compute_error_mm(peak.distance_mm, sample.distance_mm), peak)
    peak = min(peaks, key=lambda item: item.index)
    return RuleResult(peak.distance_mm, compute_error_mm(peak.distance_mm, sample.distance_mm), peak)


def exclusion_index(sample: ProfileSample, exclusion_mm: float) -> int:
    if exclusion_mm <= 0 or sample.scan_start_mm is None or sample.scan_length_mm is None or not sample.profile_data:
        return 0
    step = sample.scan_length_mm / max(len(sample.profile_data), 1)
    if step <= 0:
        return 0
    exclude_until = sample.scan_start_mm + exclusion_mm
    index = int(max(0, math.floor((exclude_until - sample.scan_start_mm) / step)))
    return min(index, max(0, len(sample.profile_data) - 1))


def exclusion_key(exclusion_mm: float) -> str:
    return str(int(exclusion_mm)).replace("-", "neg")


def compute_error_mm(estimated_distance_mm: float | None, actual_distance_mm: float | None) -> float | None:
    if estimated_distance_mm is None or actual_distance_mm is None:
        return None
    return estimated_distance_mm - actual_distance_mm


def build_summary(rows: list[dict[str, Any]], exclusion_summaries: list[dict[str, Any]], best_exclusion: float | None) -> dict[str, Any]:
    def mae(field: str) -> float | None:
        errors = [abs(row[field]) for row in rows if row.get(field) is not None]
        return statistics.mean(errors) if errors else None

    return {
        "sample_count": len(rows),
        "rule_a_mae_mm": mae("rule_a_error_mm"),
        "rule_c_mae_mm": mae("rule_c_error_mm"),
        "rule_d_mae_mm": mae("rule_d_error_mm"),
        "best_rule_b_exclusion_mm": best_exclusion,
        "best_rule_b_mae_mm": next((item["mae_mm"] for item in exclusion_summaries if item["exclusion_mm"] == best_exclusion), None),
    }


def save_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    samples: list[ProfileSample],
    exclusion_summaries: list[dict[str, Any]],
    best_exclusion: float | None,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    actual = [row["distance_mm"] for row in rows if row["distance_mm"] is not None and row["strongest_peak_distance_mm"] is not None]
    strongest = [row["strongest_peak_distance_mm"] for row in rows if row["distance_mm"] is not None and row["strongest_peak_distance_mm"] is not None]
    if actual and strongest:
        plt.figure(figsize=(6, 6))
        plt.scatter(actual, strongest, s=12, alpha=0.6)
        min_v = min(actual + strongest)
        max_v = max(actual + strongest)
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
        plt.xlabel("Actual distance_mm")
        plt.ylabel("Strongest peak distance_mm")
        plt.title("Strongest Peak vs Actual Distance")
        plt.tight_layout()
        plt.savefig(output_dir / "scatter_strongest_vs_actual.png", dpi=150)
        plt.close()

    strongest_errors = [row["strongest_peak_distance_mm"] - row["distance_mm"] for row in rows if row["distance_mm"] is not None and row["strongest_peak_distance_mm"] is not None]
    if strongest_errors:
        plt.figure(figsize=(7, 4))
        plt.hist(strongest_errors, bins=40)
        plt.xlabel("Error mm")
        plt.ylabel("Count")
        plt.title("Strongest Peak Error Histogram")
        plt.tight_layout()
        plt.savefig(output_dir / "error_histogram_strongest_peak.png", dpi=150)
        plt.close()

    ex_values = [item["exclusion_mm"] for item in exclusion_summaries if item["mae_mm"] is not None]
    ex_mae = [item["mae_mm"] for item in exclusion_summaries if item["mae_mm"] is not None]
    if ex_values and ex_mae:
        plt.figure(figsize=(7, 4))
        plt.plot(ex_values, ex_mae, marker="o")
        plt.xlabel("Near-field exclusion mm")
        plt.ylabel("MAE mm")
        plt.title("Exclusion vs MAE (Rule B)")
        plt.tight_layout()
        plt.savefig(output_dir / "exclusion_vs_mae.png", dpi=150)
        plt.close()

    conf = [row["confidence"] for row in rows if row["confidence"] is not None and row["rule_a_error_mm"] is not None]
    err = [abs(row["rule_a_error_mm"]) for row in rows if row["confidence"] is not None and row["rule_a_error_mm"] is not None]
    if conf and err:
        plt.figure(figsize=(7, 4))
        plt.scatter(conf, err, s=12, alpha=0.6)
        plt.xlabel("Confidence")
        plt.ylabel("|Error| mm")
        plt.title("Confidence vs Error (Rule A)")
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_vs_error.png", dpi=150)
        plt.close()

    heatmap = [sample.profile_data for sample in samples[: min(200, len(samples))]]
    if heatmap:
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Profile intensity")
        plt.xlabel("Profile bin")
        plt.ylabel("Sample index")
        title = "Profile Heatmap"
        if best_exclusion is not None:
            title += f" (best_exclusion={best_exclusion:.0f} mm)"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / "profile_heatmap.png", dpi=150)
        plt.close()


def create_report(
    output_path: Path,
    session_root: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    exclusion_summaries: list[dict[str, Any]],
    representative_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success_cases = sorted(
        [row for row in rows if row.get("rule_a_error_mm") is not None],
        key=lambda row: abs(row["rule_a_error_mm"]),
    )[:representative_count]
    failure_cases = sorted(
        [row for row in rows if row.get("rule_a_error_mm") is not None],
        key=lambda row: abs(row["rule_a_error_mm"]),
        reverse=True,
    )[:representative_count]

    lines: list[str] = []
    lines.append(f"# Sonar Profile Analysis Report\n")
    lines.append(f"- Session root: `{session_root}`")
    lines.append(f"- Sample count: {summary['sample_count']}")
    lines.append(f"- Rule A MAE: {format_metric(summary['rule_a_mae_mm'])} mm")
    lines.append(f"- Rule C MAE: {format_metric(summary['rule_c_mae_mm'])} mm")
    lines.append(f"- Rule D MAE: {format_metric(summary['rule_d_mae_mm'])} mm")
    lines.append(f"- Best Rule B exclusion: {summary['best_rule_b_exclusion_mm']}")
    lines.append(f"- Best Rule B MAE: {format_metric(summary['best_rule_b_mae_mm'])} mm")
    lines.append("")
    lines.append("## Strongest Peak Hypothesis")
    lines.append(strongest_peak_interpretation(summary))
    lines.append("")
    lines.append("## Near-Field Clutter Interpretation")
    lines.append(near_field_interpretation(exclusion_summaries))
    lines.append("")
    lines.append("## Generalization and Limits")
    lines.append(
        "These results are tied to the current scan range, gain, environment, and the available `profile_data` quality. "
        "Rule-based estimators can highlight whether strongest-peak selection is plausible, but they do not replace a full sonar signal model. "
        "Generalization is limited when bottom type, angle, clutter, or target geometry changes."
    )
    lines.append("")
    lines.append("## Representative Success Cases")
    lines.extend(format_case_lines(success_cases))
    lines.append("")
    lines.append("## Representative Failure Cases")
    lines.extend(format_case_lines(failure_cases))
    lines.append("")
    lines.append("## Figure Files")
    lines.append("- `scatter_strongest_vs_actual.png`")
    lines.append("- `error_histogram_strongest_peak.png`")
    lines.append("- `exclusion_vs_mae.png`")
    lines.append("- `confidence_vs_error.png`")
    lines.append("- `profile_heatmap.png`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strongest_peak_interpretation(summary: dict[str, Any]) -> str:
    rule_a_mae = summary.get("rule_a_mae_mm")
    best_rule_b_mae = summary.get("best_rule_b_mae_mm")
    if rule_a_mae is None:
        return "No valid comparison could be computed because the dataset does not contain usable profile samples."
    if best_rule_b_mae is not None and best_rule_b_mae < rule_a_mae:
        return (
            f"Rule B outperformed global argmax by reducing MAE from {rule_a_mae:.2f} mm to {best_rule_b_mae:.2f} mm. "
            "That suggests the strongest peak hypothesis is weakened by near-field clutter in at least part of the dataset."
        )
    return (
        f"Rule A achieved MAE {rule_a_mae:.2f} mm, and no exclusion setting produced a lower MAE. "
        "That supports the strongest peak hypothesis for this dataset."
    )


def near_field_interpretation(exclusion_summaries: list[dict[str, Any]]) -> str:
    if not exclusion_summaries:
        return "No exclusion sweep results were available."
    best = min((item for item in exclusion_summaries if item["mae_mm"] is not None), key=lambda item: item["mae_mm"], default=None)
    if best is None:
        return "Exclusion sweep did not yield usable metrics."
    if best["exclusion_mm"] <= 0:
        return "The best result occurred without excluding the near field, so clutter close to the transducer was not the main error source in this dataset."
    return (
        f"The best exclusion setting was {best['exclusion_mm']:.0f} mm with MAE {best['mae_mm']:.2f} mm. "
        "That indicates near-field clutter likely influences the naive strongest-peak estimate."
    )


def format_case_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- none"]
    lines = []
    for row in rows:
        lines.append(
            "- line={line_index}, distance_mm={distance_mm}, strongest_peak_distance_mm={strongest_peak_distance_mm}, "
            "confidence={confidence}, rule_a_error_mm={rule_a_error_mm}".format(**row)
        )
    return lines


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
