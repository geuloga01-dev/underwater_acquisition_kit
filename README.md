# underwater_acquisition_kit

Jetson-side underwater acquisition and control kit for a sealed vessel. Core acquisition is designed to stay local and keep recording even if Wi-Fi drops. Remote networking is optional for monitoring and control convenience only.

## Architecture

- Local acquisition is the priority.
- Camera recording, sonar logging, battery logging, and metadata writing run on Jetson without requiring a remote client.
- FastAPI status/control is optional and sits beside acquisition instead of underneath it.
- Wi-Fi monitoring and Jetson power-mode switching are helpers. Failures there should warn and continue rather than stop the core data path.

## Project Layout

```text
underwater_acquisition_kit/
|- apps/
|  |- camera_test.py
|  |- camera_record.py
|  |- sonar_logger.py
|  |- sonar_quick_test.py
|  |- run_session.py
|  `- status_server.py
|- configs/
|  |- camera.yaml
|  |- sonar.yaml
|  |- battery.yaml
|  |- network.yaml
|  |- system.yaml
|  `- server.yaml
|- data/
|- logs/
`- src/
   |- camera/
   |- control/
   |- network/
   |- sonar/
   |- state/
   |- system/
   |- telemetry/
   `- utils/
```

## New Modules

- `src/control/session_controller.py`
  Runs the integrated session flow and keeps the app layer thin.
- `src/control/status_server.py`
  Creates the FastAPI app with JSON status/control endpoints.
- `src/telemetry/battery_listener.py`
  Reads `BATTERY_STATUS` from Pixhawk MAVLink and writes battery CSV data.
- `src/network/wifi_monitor.py`
  Checks Wi-Fi state with `nmcli` and tries reconnect without blocking acquisition.
- `src/system/power_manager.py`
  Wraps `nvpmodel` and optional `jetson_clocks` in a safe helper.
- `src/state/runtime_state.py`
  Stores latest battery, session, camera, sonar, server, and network status in memory.

## Session Data Layout

Each session creates:

```text
data/sessions/<session_id>/
|- video/
|- sonar/
|- battery/
|- logs/
`- meta/
```

Typical outputs:

- `video/camera_record.<container>`
- `sonar/sonar_log.csv`
- `battery/battery_log.csv`
- `meta/session_metadata.json`
- `logs/run_session.log`

## Config Files

### `configs/camera.yaml`

- Camera source, backend, resolution, FPS, preview, autofocus
- Recording duration, preview override, codec/container

### `configs/sonar.yaml`

- Ping Sonar serial port and baudrate
- Sample interval, CSV output, telemetry hook flags

### `configs/battery.yaml`

- Pixhawk MAVLink port and baudrate
- Battery poll interval
- CSV output and low-battery threshold

### `configs/network.yaml`

- Target SSID / connection name
- Wi-Fi check interval
- Auto reconnect enable flag

### `configs/system.yaml`

- Logical Jetson states: `idle`, `recording`, `heavy`
- Mapped `nvpmodel` mode IDs
- Optional `jetson_clocks` usage

### `configs/server.yaml`

- FastAPI bind host and port

## Requirements

- Python 3.10+
- `opencv-python`
- `PyYAML`
- `brping`
- `pymavlink`
- `fastapi`
- `uvicorn`

Example:

```bash
pip install opencv-python PyYAML brping pymavlink fastapi uvicorn
```

## Local Session Run

Run a full local session on Jetson:

```bash
python3 apps/run_session.py
```

What it does:

- creates a new session folder
- prepares sonar first
- starts sonar logging
- starts battery logging
- opens camera and records locally
- writes metadata and logs
- continues locally even if Wi-Fi reconnect fails
- stops cleanly on Ctrl+C

## Status Server Run

Run the optional remote status/control server:

```bash
python3 apps/status_server.py
```

Current endpoints:

- `GET /status`
- `GET /battery`
- `GET /health`
- `POST /session/start`
- `POST /session/stop`

These endpoints return JSON only.

## Battery Logging

Battery logging listens for MAVLink `BATTERY_STATUS` from Pixhawk, by default on `/dev/ttyACM0`.

Logged fields:

- `timestamp_iso`
- `unix_time`
- `voltage_v`
- `current_a`
- `remaining_percent`

The latest battery state is also exposed through the runtime state object and the `/battery` endpoint.

If the battery listener fails, the failure is logged and camera/sonar acquisition can continue.

## Wi-Fi Auto Reconnect

Wi-Fi monitoring uses `nmcli`.

Behavior:

- periodically checks whether Jetson is connected to the configured SSID
- if disconnected, tries reconnect with `nmcli connection up`
- updates in-memory runtime state
- logs warnings on failure
- does not stop acquisition if reconnect fails

## Power Mode Manager

The power manager wraps `nvpmodel` with logical states:

- `idle`
- `recording`
- `heavy`

The actual numeric mode mapping comes from `configs/system.yaml`.

If a power-mode change fails, the system logs a warning and continues.

## Design Notes

- Acquisition is network-independent by design.
- App entry points stay thin and mostly wire config + runtime state + reusable modules together.
- Hardware-sensitive settings are kept in YAML instead of hard-coded constants.
- Camera and sonar MVP scripts remain available for isolated debugging.
