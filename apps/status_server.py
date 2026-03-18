from __future__ import annotations

import logging
from pathlib import Path
import sys

from fastapi import FastAPI
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.session_controller import SessionController
from src.control.status_server import create_status_app
from src.network.wifi_monitor import WifiMonitor, load_network_config
from src.state.runtime_state import RuntimeState
from src.system.power_manager import JetsonPowerManager, load_system_config
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


server_raw = load_yaml_config(PROJECT_ROOT / "configs" / "server.yaml")
system_raw = load_yaml_config(PROJECT_ROOT / "configs" / "system.yaml")
network_raw = load_yaml_config(PROJECT_ROOT / "configs" / "network.yaml")

logger = get_app_logger("status_server", PROJECT_ROOT / "logs", level=logging.INFO)
level = resolve_log_level(server_raw, system_raw, network_raw)
logger.setLevel(level)
for handler in logger.handlers:
    handler.setLevel(level)

runtime_state = RuntimeState(PROJECT_ROOT / "data")
power_manager = JetsonPowerManager(load_system_config(system_raw), logger=logger)
session_controller = SessionController(PROJECT_ROOT, runtime_state, power_manager=power_manager)
wifi_monitor = WifiMonitor(load_network_config(network_raw), runtime_state, logger=logger)
app: FastAPI = create_status_app(
    PROJECT_ROOT,
    runtime_state,
    session_controller,
    logger=logger,
    wifi_monitor=wifi_monitor,
)
