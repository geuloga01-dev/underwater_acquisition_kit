from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sonar_profile_analysis import analyze_session
from src.utils.logger import get_app_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sonar_profile.jsonl and compare rule-based distance estimators.")
    parser.add_argument("--session", required=True, help="Path to a session root, e.g. data/sessions/<session_id>")
    parser.add_argument(
        "--exclusion-mm",
        default="0,50,100,150,200,300",
        help="Comma-separated near-field exclusion values in mm for Rule B.",
    )
    parser.add_argument("--representative-count", type=int, default=5, help="Number of success/failure examples to include.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_app_logger("sonar_profile_analysis", PROJECT_ROOT / "logs")

    try:
        session_root = Path(args.session)
        if not session_root.is_absolute():
            session_root = (PROJECT_ROOT / session_root).resolve()

        exclusion_mm_values = [float(value.strip()) for value in args.exclusion_mm.split(",") if value.strip()]
        output_dir = session_root / "analysis" / "sonar_profile"
        summary = analyze_session(
            session_root=session_root,
            output_dir=output_dir,
            logger=logger,
            exclusion_mm_values=exclusion_mm_values,
            representative_count=args.representative_count,
        )

        print("analysis_output:", output_dir)
        print("sample_count:", summary["sample_count"])
        print("rule_a_mae_mm:", summary["rule_a_mae_mm"])
        print("rule_c_mae_mm:", summary["rule_c_mae_mm"])
        print("rule_d_mae_mm:", summary["rule_d_mae_mm"])
        print("best_rule_b_exclusion_mm:", summary["best_rule_b_exclusion_mm"])
        print("best_rule_b_mae_mm:", summary["best_rule_b_mae_mm"])
        return 0
    except Exception as exc:
        logger.exception("Sonar profile analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
