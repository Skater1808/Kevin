"""FastAPI application: REST control plane + WebSocket telemetry.

Wires together the configuration, LLM client, sandboxed environment, tool
registry (with dynamically discovered plugins), memory, healer and the
:class:`~agent.core.AgentCore`. Exposes ``/start``, ``/pause``, ``/resume`` and
``/stop`` plus a ``/ws/logs`` WebSocket that streams the standardized event
packets to every connected dashboard.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from agent.core import AgentCore
from agent.healer import HealerModule
from agent.memory import Memory
from config import build_llm_client, get_settings
from execution.local_env import LocalEnvironment
from plugins.manager import PluginManager
from tools.registry import build_default_registry

logger = logging.getLogger("aegis.ui")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts event packets."""

    def __init__(self, backlog: int = 200) -> None:
        self._connections: set[WebSocket] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=backlog)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        for packet in list(self._history):
            await websocket.send_json(packet)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, packet: dict[str, Any]) -> None:
        self._history.append(packet)
        async with self._lock:
            targets = list(self._connections)
        stale: list[WebSocket] = []
        for connection in targets:
            try:
                await connection.send_json(packet)
            except Exception:  # noqa: BLE001 - drop broken sockets
                stale.append(connection)
        if stale:
            async with self._lock:
                for connection in stale:
                    self._connections.discard(connection)


class StartRequest(BaseModel):
    goal: str


def build_agent() -> AgentCore:
    """Construct a fully wired :class:`AgentCore` instance."""
    settings = get_settings()
    env = LocalEnvironment(settings.workspace_dir, command_timeout=settings.command_timeout)
    registry = build_default_registry(env)
    llm = build_llm_client(settings)
    memory = Memory()
    healer = HealerModule(llm, max_attempts=settings.max_healing_attempts)
    return AgentCore(
        settings=settings,
        llm=llm,
        registry=registry,
        env=env,
        memory=memory,
        healer=healer,
    )


def create_app() -> FastAPI:
    manager = ConnectionManager()
    agent = build_agent()
    agent.set_emitter(manager.broadcast)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        plugin_manager = PluginManager(agent.registry)
        loaded = await plugin_manager.discover()
        logger.info("Startup complete. Plugins: %s", [p.name for p in loaded])
        yield
        if agent.is_running:
            await agent.stop()
        agent.memory.close()

    app = FastAPI(title="Aegis-X Enterprise", version="1.0.0", lifespan=lifespan)
    app.state.agent = agent
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {"settings": get_settings()})

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(agent.snapshot())

    @app.post("/start")
    async def start(req: StartRequest) -> JSONResponse:
        started = await agent.start(req.goal)
        return JSONResponse({"started": started, "state": agent.state.value})

    @app.post("/pause")
    async def pause() -> JSONResponse:
        agent.pause()
        await manager.broadcast(
            {"type": "log", "payload": {"message": "Pause requested via API."}, "timestamp": agent.snapshot()["timestamp"]}
        )
        return JSONResponse({"state": agent.state.value, "paused": True})

    @app.post("/resume")
    async def resume() -> JSONResponse:
        agent.resume()
        return JSONResponse({"state": agent.state.value, "paused": False})

    @app.post("/stop")
    async def stop() -> JSONResponse:
        await agent.stop()
        return JSONResponse({"state": agent.state.value, "stopped": True})

    @app.websocket("/ws/logs")
    async def ws_logs(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        try:
            while True:
                # Keep the connection open; inbound messages are ignored.
                await websocket.receive_text()
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception:  # noqa: BLE001
            await manager.disconnect(websocket)

    return app


app = create_app()
