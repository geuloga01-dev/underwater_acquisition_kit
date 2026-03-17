from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


@dataclass(slots=True)
class SessionPaths:
    session_id: str
    root: Path
    video: Path
    sonar: Path
    battery: Path
    logs: Path
    meta: Path


def create_session_id(session_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = slugify(session_name or "session")
    return f"{timestamp}_{suffix}"


def create_session_dirs(base_dir: Path, session_name: str | None = None) -> SessionPaths:
    session_id = create_session_id(session_name)
    root = base_dir / "sessions" / session_id
    paths = SessionPaths(
        session_id=session_id,
        root=root,
        video=root / "video",
        sonar=root / "sonar",
        battery=root / "battery",
        logs=root / "logs",
        meta=root / "meta",
    )

    for path in (paths.root, paths.video, paths.sonar, paths.battery, paths.logs, paths.meta):
        path.mkdir(parents=True, exist_ok=True)

    return paths


def save_metadata(target_path: Path, metadata: dict[str, Any]) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target_path


def session_paths_to_dict(paths: SessionPaths) -> dict[str, str]:
    data = asdict(paths)
    return {key: str(value) for key, value in data.items()}


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "session"
