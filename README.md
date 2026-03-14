# underwater_acquisition_kit

Jetson Orin Nano based underwater data acquisition kit, developed first on a Windows laptop webcam and later extended to USB camera and sonar hardware on Jetson.

## Project Layout

```text
underwater_acquisition_kit/
|- apps/                  # Executable entry points
|  |- camera_test.py
|  |- camera_record.py
|  |- sonar_logger.py
|  `- run_session.py
|- configs/               # Runtime configuration
|  |- camera.yaml
|  `- sonar.yaml
|- data/                  # Test data or captured outputs
|- logs/                  # Runtime log files
`- src/                   # Reusable modules
   |- camera/
   |- sonar/
   `- utils/
```

## Overview

- `camera_test.py` is the first MVP target and should run on a Windows laptop webcam right away.
- `sonar_logger.py` is the first sonar logging scaffold for Ping Sonar R2.
- `run_session.py` creates a session folder layout that can later store video, sonar, logs, and metadata together.

## Why It Is Split This Way

- `apps/` holds only the runnable scripts.
- `src/camera/` keeps camera capture logic reusable so preview, recording, and future inference can share the same module.
- `src/sonar/` separates device read, normalized record creation, local save, and future telemetry publish hooks.
- `src/utils/` keeps shared logging and session helpers in one place.
- `configs/` isolates hardware-dependent settings, which lets us switch from a laptop webcam to `/dev/video0` or `/dev/ttyUSB0` by editing YAML instead of changing code.
- `logs/` is prepared from the start so camera, sonar, and session apps can share the same logging pattern.

## Requirements

- Python 3.10+
- `opencv-python`
- `PyYAML`
- `brping` for Ping Sonar R2 support on the Jetson or Ubuntu side

Example:

```powershell
pip install opencv-python PyYAML
```

For sonar on Jetson or Ubuntu:

```powershell
pip install brping
```

## Run Webcam Test

From the project root:

```powershell
py -3 apps/camera_test.py
```

What it does:

- loads `configs/camera.yaml`
- opens the configured camera source
- applies optional camera controls such as autofocus
- shows a live preview window
- exits when you press `q`
- writes logs to `logs/`

## Camera Configuration

`configs/camera.yaml` controls the camera source without touching code.

Examples:

- Windows laptop webcam:

```yaml
camera:
  source: 0
  backend: dshow
  preview: true
  autofocus: true
```

- Jetson USB camera:

```yaml
camera:
  source: "/dev/video0"
  backend: v4l2
```

Useful focus options:

- `autofocus: true` enables camera-side autofocus when supported.
- `focus: 10` can be used for manual focus on supported cameras and backends.
- `warmup_frames: 20` discards a few startup frames so focus and exposure can settle.

Useful recording options:

- `recording.duration_seconds: 300` runs a 5-minute capture for stability testing.
- `recording.preview: false` avoids preview overhead during long recordings.
- `recording.fourcc: mp4v` stores a simple `.mp4` file that is easy to inspect on Windows first.

## Sonar Logging MVP

`apps/sonar_logger.py` reads `configs/sonar.yaml`, tries to connect to Ping Sonar R2, normalizes each sample, optionally writes CSV, and leaves a telemetry packet creation hook in the reusable sonar module.

Expected Jetson-side settings:

```yaml
sonar:
  port: "/dev/ttyUSB0"
  baudrate: 115200
  sample_interval: 0.2
  csv_save: true
  telemetry_enabled: false
```

On a Windows laptop without the device attached, the app is expected to fail with a clear hardware-related message.

## Session Layout MVP

`apps/run_session.py` creates:

```text
data/sessions/<session_id>/
|- video/
|- sonar/
|- logs/
`- meta/
```

This is intentionally a small scaffold first so later we can plug camera recording, sonar logging, metadata, and real-time processing into one session flow without rewriting the directory structure.

Current behavior:

- creates one `session_id`
- records camera video to `video/camera_record.mp4`
- logs sonar samples to `sonar/sonar_log.csv`
- stores metadata in `meta/session_metadata.json`
- writes session logs to `logs/run_session.log`
- stops cleanly on `Ctrl+C`

## File Roles

- `apps/camera_test.py`: app entry point for preview testing.
- `apps/sonar_logger.py`: app entry point for Ping Sonar logging.
- `apps/run_session.py`: session scaffold that prepares per-run folders and metadata.
- `src/camera/webcam.py`: reusable webcam capture module.
- `src/sonar/ping_logger.py`: reusable Ping Sonar device read and CSV logging flow.
- `src/utils/logger.py`: shared file and console logger setup.
- `src/utils/session.py`: session ID, folder creation, and metadata helpers.

## Notes For Next Phase

- The current code avoids hardware-specific logic in the app layer.
- When moving to Jetson, we should mainly adjust `configs/camera.yaml` and extend `src/camera/` for ExploreHD or GStreamer-specific pipelines if needed.
- Sonar already exposes the flow `device read -> normalized record -> local save -> optional telemetry publish`, so a telemetry module can be attached later without changing the logger app shape.
- Camera is still focused on capture and preview first, but the reusable module boundary is ready for a future inference layer.
