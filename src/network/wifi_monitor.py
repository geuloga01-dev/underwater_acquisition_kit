from __future__ import annotations

from dataclasses import dataclass
import logging
import subprocess
import threading
from typing import Any

from src.state.runtime_state import RuntimeState


@dataclass(slots=True)
class NetworkConfig:
    enabled: bool = True
    ssid: str = ""
    connection_name: str = ""
    check_interval: float = 10.0
    reconnect_enabled: bool = True


def load_network_config(raw_config: dict[str, Any]) -> NetworkConfig:
    section = raw_config.get("network", {})
    return NetworkConfig(
        enabled=_optional_bool(section.get("enabled"), default=True),
        ssid=str(section.get("ssid", "")),
        connection_name=str(section.get("connection_name", "")),
        check_interval=float(section.get("check_interval", 10.0)),
        reconnect_enabled=_optional_bool(section.get("reconnect_enabled"), default=True),
    )


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class WifiMonitor:
    def __init__(self, config: NetworkConfig, state: RuntimeState, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.state = state
        self.logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ethernet_active_logged = False

    def start(self) -> None:
        if not self.config.enabled:
            self.logger.info("Wi-Fi monitor disabled in config.")
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self.run_loop, name="wifi-monitor", daemon=True)
        self._thread.start()
        self.logger.info("Wi-Fi monitor started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def run_loop(self) -> None:
        self.state.update_component("network", ready=True, running=True, ok=True, last_error=None)

        while not self._stop_event.is_set():
            try:
                ethernet_connected, ethernet_device = self.is_ethernet_connected()
                if ethernet_connected and not self._ethernet_active_logged:
                    self.logger.info("Ethernet connected; Wi-Fi monitoring still active.")
                    self._ethernet_active_logged = True
                if not ethernet_connected:
                    self._ethernet_active_logged = False

                wifi_connected, current_ssid = self.check_connection()
                overall_connected = ethernet_connected or wifi_connected
                network_label = current_ssid or (f"ethernet:{ethernet_device}" if ethernet_connected else None)
                self.state.set_network_status(overall_connected, network_label)

                reconnect_target = self.config.connection_name or self.config.ssid
                if self.config.ssid and current_ssid != self.config.ssid and self.config.reconnect_enabled:
                    self.logger.warning(
                        "Wi-Fi disconnected or on unexpected SSID. expected=%s current=%s reconnect_target=%s",
                        self.config.ssid,
                        current_ssid,
                        reconnect_target,
                    )
                    reconnect_ok = self.try_reconnect(reconnect_target)
                    if not reconnect_ok:
                        self.state.set_network_status(overall_connected, network_label, last_error="reconnect failed")
                elif wifi_connected:
                    self.logger.debug("Wi-Fi monitor check ok. ssid=%s", current_ssid)
            except Exception as exc:
                self.logger.warning("Wi-Fi monitor check failed: %s", exc)
                self.state.set_network_status(False, None, last_error=str(exc))

            self._stop_event.wait(self.config.check_interval)

        self.state.update_component(
            "network",
            running=False,
            ok=bool(self.state.network_connected),
            last_error=self.state.network.last_error,
        )

    def is_ethernet_connected(self) -> tuple[bool, str | None]:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "dev", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nmcli device status check failed")

        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
            device, dev_type, state = parts[0], parts[1], parts[2]
            if dev_type == "ethernet" and state == "connected":
                return True, device
        return False, None

    def check_connection(self) -> tuple[bool, str | None]:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nmcli wifi check failed")

        for line in result.stdout.splitlines():
            if line.startswith("yes:"):
                return True, line.split(":", maxsplit=1)[1] or None
        return False, None

    def try_reconnect(self, target: str | None = None) -> bool:
        if target is None:
            target = self.config.connection_name or self.config.ssid
        if not target:
            self.logger.warning("Wi-Fi reconnect skipped because no SSID/connection name is configured.")
            return False

        self.logger.info("Attempting Wi-Fi reconnect using nmcli. target=%s", target)
        result = subprocess.run(
            ["nmcli", "connection", "up", target],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            self.logger.warning("Wi-Fi reconnect failed. target=%s error=%s", target, result.stderr.strip() or result.stdout.strip())
            return False

        self.logger.info("Wi-Fi reconnect succeeded. target=%s", target)
        return True
