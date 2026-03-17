from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
import yaml

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


def main() -> int:
    logger = get_app_logger("status_server", PROJECT_ROOT / "logs")
    runtime_state = RuntimeState(PROJECT_ROOT / "data")
    wifi_monitor: WifiMonitor | None = None

    try:
        server_raw = load_yaml_config(PROJECT_ROOT / "configs" / "server.yaml")
        network_raw = load_yaml_config(PROJECT_ROOT / "configs" / "network.yaml")
        system_raw = load_yaml_config(PROJECT_ROOT / "configs" / "system.yaml")
        battery_raw = load_yaml_config(PROJECT_ROOT / "configs" / "battery.yaml")

        level = resolve_log_level(server_raw, network_raw, system_raw, battery_raw)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

        host = str(server_raw.get("server", {}).get("host", "0.0.0.0"))
        port = int(server_raw.get("server", {}).get("port", 8000))

        power_manager = JetsonPowerManager(load_system_config(system_raw), logger=logger)
        session_controller = SessionController(PROJECT_ROOT, runtime_state, power_manager=power_manager)
        app = create_status_app(PROJECT_ROOT, runtime_state, session_controller)

        wifi_monitor = WifiMonitor(load_network_config(network_raw), runtime_state, logger=logger)
        wifi_monitor.start()
        runtime_state.update_component("server", ready=True, running=True, ok=True, last_error=None)

        logger.info("Starting status server on %s:%s", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info")
        return 0
    except Exception as exc:
        runtime_state.update_component("server", running=False, ok=False, last_error=str(exc))
        logger.exception("Status server failed: %s", exc)
        return 1
    finally:
        runtime_state.update_component("server", running=False, ok=runtime_state.server.ok, last_error=runtime_state.server.last_error)
        if wifi_monitor is not None:
            wifi_monitor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
