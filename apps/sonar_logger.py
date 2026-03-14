from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.sonar.ping_logger import PingSonarClient, load_sonar_config, log_sonar_stream
from src.utils.logger import get_app_logger


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_log_level(raw_config: dict) -> int:
    level_name = str(raw_config.get("logging", {}).get("level", "INFO")).upper()
    return getattr(logging, level_name, logging.INFO)


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "sonar.yaml"
    logger = get_app_logger("sonar_logger", PROJECT_ROOT / "logs")
    client: PingSonarClient | None = None

    try:
        raw_config = load_yaml_config(config_path)
        logger.setLevel(resolve_log_level(raw_config))
        for handler in logger.handlers:
            handler.setLevel(logger.level)

        sonar_config = load_sonar_config(raw_config)
        csv_path = Path(sonar_config.csv_path) if sonar_config.csv_path else PROJECT_ROOT / "data" / "sonar_log.csv"

        client = PingSonarClient(sonar_config, logger=logger)
        client.connect()
        sample_count = log_sonar_stream(client, sonar_config, logger, csv_path=csv_path)

        logger.info("Sonar logging finished. samples=%d csv_path=%s", sample_count, csv_path)
        return 0
    except FileNotFoundError:
        logger.exception("Sonar config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.exception(
            "Sonar logging failed: %s. This is expected on a laptop if the Ping device or required package is not connected.",
            exc,
        )
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
