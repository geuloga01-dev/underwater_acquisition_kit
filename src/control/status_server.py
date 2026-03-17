from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.control.session_controller import SessionController
from src.state.runtime_state import RuntimeState


class SessionStartRequest(BaseModel):
    session_name: Optional[str] = None


def create_status_app(project_root: Path, runtime_state: RuntimeState, session_controller: SessionController) -> FastAPI:
    app = FastAPI(title="Underwater Acquisition Kit Status Server")

    @app.get("/status")
    def get_status() -> dict:
        return runtime_state.snapshot()

    @app.get("/battery")
    def get_battery() -> dict:
        return runtime_state.snapshot()["latest_battery"]

    @app.get("/health")
    def get_health() -> dict:
        return runtime_state.health_snapshot()

    @app.post("/session/start")
    def start_session(payload: SessionStartRequest | None = None) -> dict:
        session_name = payload.session_name if payload is not None else None
        return session_controller.start_session(session_name=session_name)

    @app.post("/session/stop")
    def stop_session() -> dict:
        return session_controller.stop_session()

    return app
