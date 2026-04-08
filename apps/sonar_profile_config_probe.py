from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from brping import Ping1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OSS profile configuration support and try candidate point counts."
    )
    parser.add_argument(
        "--candidates",
        default="200,400,800,1200",
        help="Comma-separated candidate number_of_points values to try.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="If set, keep this number_of_points at the end instead of restoring the original config.",
    )
    return parser.parse_args()


def load_sonar_config() -> dict:
    with (PROJECT_ROOT / "configs" / "sonar.yaml").open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return raw["sonar"]


def print_profile_summary(device: Ping1D, label: str) -> None:
    profile = device.get_profile()
    if not profile:
        print(f"{label}: get_profile -> None")
        return
    values = profile.get("profile_data")
    length = None if values is None else len(values)
    print(
        f"{label}: distance={profile.get('distance')} confidence={profile.get('confidence')} "
        f"profile_len={length} head={None if values is None else list(values[:12])}"
    )


def main() -> int:
    args = parse_args()
    sonar = load_sonar_config()
    port = sonar["port"]
    baudrate = int(sonar.get("baudrate", 115200))
    candidates = [int(value.strip()) for value in args.candidates.split(",") if value.strip()]

    device = Ping1D()
    print(f"Opening {port} at {baudrate} bps")
    device.connect_serial(port, baudrate)
    time.sleep(0.75)
    print("initialize:", bool(device.initialize()))

    current = device.get_oss_profile_configuration()
    print("current_oss_profile_configuration:", current)
    print_profile_summary(device, "current_profile")

    original = None if current is None else dict(current)
    normalization = 0 if current is None else int(current.get("normalization_enabled", 0))
    enhance = 0 if current is None else int(current.get("enhance_enabled", 0))

    for candidate in candidates:
        try:
            ok = device.set_oss_profile_configuration(candidate, normalization, enhance, verify=True)
            echoed = device.get_oss_profile_configuration()
            print(f"try number_of_points={candidate}: set_ok={ok} echoed={echoed}")
            print_profile_summary(device, f"profile_after_{candidate}")
        except Exception as exc:
            print(f"try number_of_points={candidate}: exception={exc.__class__.__name__}: {exc}")

    if args.keep is not None:
        ok = device.set_oss_profile_configuration(args.keep, normalization, enhance, verify=True)
        echoed = device.get_oss_profile_configuration()
        print(f"keeping number_of_points={args.keep}: set_ok={ok} echoed={echoed}")
    elif original is not None:
        ok = device.set_oss_profile_configuration(
            int(original["number_of_points"]),
            int(original["normalization_enabled"]),
            int(original["enhance_enabled"]),
            verify=True,
        )
        echoed = device.get_oss_profile_configuration()
        print(f"restored original config: set_ok={ok} echoed={echoed}")

    try:
        device.iodev.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
