# Aegis-X Enterprise

A production-grade, fully **asynchronous** autonomous AI agent with an interactive
setup assistant, a dynamic plugin system, a real-time **FastAPI + WebSocket**
dashboard, and a **Hermes-style iterative self-healing engine**.

## Highlights

- **Strict state machine** — `IDLE → PLANNING → EXECUTING → TESTING → HEALING → COMPLETED | FAILED`.
- **Async ReAct loop** — reasoning + acting with one tool call per step, persisted to `agent_state.json`.
- **Sandboxed execution** — every file/terminal operation is confined to `./workspace/` via a central
  `validate_path()` boundary check; all commands run with a hard timeout (default 45s).
- **Hermes self-healing** — failing commands/tests trigger a reflexion prompt; the corrected code is applied
  and re-tested, escalating to `FAILED` after 3 unsuccessful heals at the same location.
- **Dynamic plugins** — `.py` files in `plugins/` are imported at boot via `importlib` and register tools.
- **Provider-agnostic** — OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), Google Gemini (1.5 Pro), Ollama (local).
- **Live dashboard** — TailwindCSS console, dynamic task list, and Start / Pause / Resume / Emergency-Stop controls.

## Quick start

```bash
cd aegis_x_enterprise
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Configure provider + credentials (writes .env)
python setup.py

# 2. Launch the dashboard
python main.py
# open http://127.0.0.1:8000
```

## Architecture

```
aegis_x_enterprise/
├── setup.py            # Interactive installer & provider assistant
├── config.py           # pydantic-settings + provider-agnostic async LLM client
├── main.py             # Entrypoint: colored logging + Uvicorn
├── workspace/          # Isolated, sandboxed agent working directory
├── plugins/            # Dynamically loaded extensions (+ sample_plugin.py)
├── agent/
│   ├── core.py         # Async ReAct loop & state machine
│   ├── memory.py       # SQLite history + JSON state persistence
│   └── healer.py       # Hermes failure analysis & code patching
├── execution/
│   └── local_env.py    # Sandboxed subprocess runner (CWD lock + timeouts)
├── tools/
│   ├── base.py         # Abstract Tool / Plugin classes
│   └── registry.py     # Filesystem, HTTP and terminal tools
└── ui/
    ├── app.py          # FastAPI app, REST routes & WebSocket server
    └── templates/index.html
```

## WebSocket protocol

The `/ws/logs` endpoint pushes standardized packets on every state change,
terminal output and LLM reflection:

```json
{ "type": "status|log|task_update|error", "payload": { }, "timestamp": "ISO-8601" }
```

## REST endpoints

| Method | Path        | Purpose                       |
|--------|-------------|-------------------------------|
| POST   | `/start`    | Start a mission (`{"goal": "..."}`) |
| POST   | `/pause`    | Pause the loop                |
| POST   | `/resume`   | Resume the loop               |
| POST   | `/stop`     | Emergency stop                |
| GET    | `/api/state`| Current `agent_state.json` snapshot |

## Writing a plugin

Drop a `.py` file into `plugins/` that subclasses `BasePlugin` and registers tools:

```python
from tools.base import BasePlugin, BaseTool, ToolResult
from tools.registry import ToolRegistry

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful."
    parameters = {"arg": "an argument"}

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(ok=True, output=f"got {kwargs.get('arg')}")

class MyPlugin(BasePlugin):
    name = "my_plugin"
    description = "Example plugin."

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(MyTool())
```

See `plugins/sample_plugin.py` for a working weather-lookup example.
