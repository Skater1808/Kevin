"""Tool registry and the built-in tool suite.

Provides the :class:`ToolRegistry` plus filesystem, HTTP and terminal tools that
operate strictly inside the sandboxed workspace.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from execution.local_env import LocalEnvironment
from tools.base import BaseTool, ToolResult

logger = logging.getLogger("aegis.tools")


class ToolRegistry:
    """Holds the set of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Overwriting already registered tool '%s'.", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                output="",
                error=f"Unknown tool '{name}'. Available: {', '.join(self.names())}",
            )
        try:
            return await tool.run(**arguments)
        except TypeError as exc:
            return ToolResult(ok=False, output="", error=f"Invalid arguments for '{name}': {exc}")
        except PermissionError as exc:
            return ToolResult(ok=False, output="", error=f"Permission denied: {exc}")
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            logger.exception("Tool '%s' raised an unexpected error", name)
            return ToolResult(ok=False, output="", error=f"Tool '{name}' failed: {exc}")


# --------------------------------------------------------------------------- #
# Built-in tools
# --------------------------------------------------------------------------- #
class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a UTF-8 text file inside the workspace."
    parameters = {"path": "Relative file path", "content": "Full file content"}

    def __init__(self, env: LocalEnvironment) -> None:
        super().__init__()
        self.env = env

    async def run(self, **kwargs: Any) -> ToolResult:
        path = str(kwargs["path"])
        content = str(kwargs.get("content", ""))
        target = self.env.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, output=f"Wrote {len(content)} bytes to {path}")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace."
    parameters = {"path": "Relative file path"}

    def __init__(self, env: LocalEnvironment) -> None:
        super().__init__()
        self.env = env

    async def run(self, **kwargs: Any) -> ToolResult:
        path = str(kwargs["path"])
        target = self.env.resolve(path)
        if not target.is_file():
            return ToolResult(ok=False, output="", error=f"File not found: {path}")
        content = target.read_text(encoding="utf-8", errors="replace")
        return ToolResult(ok=True, output=content, data={"path": path})


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List files and directories at a workspace-relative path."
    parameters = {"path": "Relative directory path (default: workspace root)"}

    def __init__(self, env: LocalEnvironment) -> None:
        super().__init__()
        self.env = env

    async def run(self, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "."))
        target = self.env.resolve(path)
        if not target.is_dir():
            return ToolResult(ok=False, output="", error=f"Not a directory: {path}")
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )
        return ToolResult(ok=True, output="\n".join(entries), data={"entries": entries})


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "Execute a shell command inside the workspace with a hard timeout. "
        "Returns combined stdout/stderr and the exit code."
    )
    parameters = {"command": "Shell command to execute"}

    def __init__(self, env: LocalEnvironment) -> None:
        super().__init__()
        self.env = env

    async def run(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs["command"])
        result = await self.env.run_command(command)
        combined = result.stdout
        if result.stderr:
            combined += ("\n" if combined else "") + result.stderr
        return ToolResult(
            ok=result.ok,
            output=combined,
            error="" if result.ok else result.stderr or "non-zero exit code",
            data={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = "Perform an HTTP request and return the response body (text)."
    parameters = {
        "url": "Target URL",
        "method": "HTTP method (default GET)",
        "body": "Optional request body (string)",
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs["url"])
        method = str(kwargs.get("method", "GET")).upper()
        body = kwargs.get("body")
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.request(method, url, content=body)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output="", error=f"HTTP request failed: {exc}")
        text = response.text
        truncated = text[:8000]
        return ToolResult(
            ok=response.is_success,
            output=truncated,
            error="" if response.is_success else f"HTTP {response.status_code}",
            data={"status_code": response.status_code, "url": str(response.url)},
        )


def build_default_registry(env: LocalEnvironment) -> ToolRegistry:
    """Construct a registry populated with the built-in tool suite."""
    registry = ToolRegistry()
    registry.register(WriteFileTool(env))
    registry.register(ReadFileTool(env))
    registry.register(ListDirTool(env))
    registry.register(RunCommandTool(env))
    registry.register(HttpRequestTool())
    return registry
