from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
from typing import Any


@dataclass(slots=True)
class SystemConfig:
    enabled: bool = True
    idle_mode: int = 1
    recording_mode: int = 0
    heavy_mode: int = 2
    use_jetson_clocks: bool = False
    nvpmodel_path: str = "nvpmodel"
    jetson_clocks_path: str = "jetson_clocks"


def load_system_config(raw_config: dict[str, Any]) -> SystemConfig:
    section = raw_config.get("power", {})
    return SystemConfig(
        enabled=_optional_bool(section.get("enabled"), default=True),
        idle_mode=int(section.get("idle_mode", 1)),
        recording_mode=int(section.get("recording_mode", 0)),
        heavy_mode=int(section.get("heavy_mode", 2)),
        use_jetson_clocks=_optional_bool(section.get("use_jetson_clocks"), default=False),
        nvpmodel_path=str(section.get("nvpmodel_path", "nvpmodel")),
        jetson_clocks_path=str(section.get("jetson_clocks_path", "jetson_clocks")),
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
        self.last_warning: str | None = None

    def _set_warning(self, message: str) -> bool:
        self.last_warning = message
        self.logger.warning(message)
        return False

    def clear_warning(self) -> None:
        self.last_warning = None

    def _resolve_command(self, configured_path: str) -> str:
        path_obj = Path(configured_path)
        if path_obj.is_absolute():
            return configured_path
        return configured_path

    def set_mode(self, logical_state: str) -> bool:
        self.clear_warning()
        if not self.config.enabled:
            self.logger.info("Jetson power management disabled in config. Skipping state=%s", logical_state)
            return True

        mode_map = {
            "idle": self.config.idle_mode,
            "recording": self.config.recording_mode,
            "heavy": self.config.heavy_mode,
        }
        if logical_state not in mode_map:
            return self._set_warning(f"Unknown Jetson power state requested: {logical_state}")

        mode_id = mode_map[logical_state]
        try:
            result = subprocess.run(
                [self._resolve_command(self.config.nvpmodel_path), "-m", str(mode_id)],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return self._set_warning(
                f"Power mode command not found: {self.config.nvpmodel_path}. Continuing without power-mode change."
            )
        except Exception as exc:
            return self._set_warning(
                f"Power mode command failed unexpectedly for state={logical_state}: {exc}. Continuing without power-mode change."
            )

        if result.returncode != 0:
            return self._set_warning(
                "Failed to set Jetson power mode. "
                f"state={logical_state} mode_id={mode_id} error={result.stderr.strip() or result.stdout.strip()}"
            )

        self.logger.info("Jetson power mode set. state=%s mode_id=%s", logical_state, mode_id)

        if self.config.use_jetson_clocks and logical_state in {"recording", "heavy"}:
            try:
                clocks_result = subprocess.run(
                    [self._resolve_command(self.config.jetson_clocks_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                return self._set_warning(
                    f"jetson_clocks command not found: {self.config.jetson_clocks_path}. Continuing without jetson_clocks."
                )
            except Exception as exc:
                return self._set_warning(
                    f"jetson_clocks failed unexpectedly: {exc}. Continuing without jetson_clocks."
                )
            if clocks_result.returncode != 0:
                return self._set_warning(
                    f"jetson_clocks failed: {clocks_result.stderr.strip() or clocks_result.stdout.strip()}"
                )
            self.logger.info("jetson_clocks enabled for state=%s", logical_state)

        return True
