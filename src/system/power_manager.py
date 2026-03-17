from __future__ import annotations

from dataclasses import dataclass
import logging
import subprocess
from typing import Any


@dataclass(slots=True)
class SystemConfig:
    idle_mode: int = 1
    recording_mode: int = 0
    heavy_mode: int = 2
    use_jetson_clocks: bool = False


def load_system_config(raw_config: dict[str, Any]) -> SystemConfig:
    section = raw_config.get("power", {})
    return SystemConfig(
        idle_mode=int(section.get("idle_mode", 1)),
        recording_mode=int(section.get("recording_mode", 0)),
        heavy_mode=int(section.get("heavy_mode", 2)),
        use_jetson_clocks=_optional_bool(section.get("use_jetson_clocks"), default=False),
    )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class JetsonPowerManager:
    def __init__(self, config: SystemConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def set_mode(self, logical_state: str) -> bool:
        mode_map = {
            "idle": self.config.idle_mode,
            "recording": self.config.recording_mode,
            "heavy": self.config.heavy_mode,
        }
        if logical_state not in mode_map:
            self.logger.warning("Unknown Jetson power state requested: %s", logical_state)
            return False

        mode_id = mode_map[logical_state]
        result = subprocess.run(
            ["nvpmodel", "-m", str(mode_id)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.logger.warning(
                "Failed to set Jetson power mode. state=%s mode_id=%s error=%s",
                logical_state,
                mode_id,
                result.stderr.strip() or result.stdout.strip(),
            )
            return False

        self.logger.info("Jetson power mode set. state=%s mode_id=%s", logical_state, mode_id)

        if self.config.use_jetson_clocks and logical_state in {"recording", "heavy"}:
            clocks_result = subprocess.run(
                ["jetson_clocks"],
                capture_output=True,
                text=True,
                check=False,
            )
            if clocks_result.returncode != 0:
                self.logger.warning("jetson_clocks failed: %s", clocks_result.stderr.strip() or clocks_result.stdout.strip())
                return False
            self.logger.info("jetson_clocks enabled for state=%s", logical_state)

        return True
