"""Secure local execution environment.

Provides workspace path validation (path-injection defense) and an asynchronous
subprocess runner that confines execution to the workspace and enforces a hard
timeout on every command.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("aegis.execution")


@dataclass(slots=True)
class CommandResult:
    """Outcome of a terminal command execution."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def validate_path(target_path: str, workspace: Path) -> Path:
    """Resolve ``target_path`` and ensure it stays inside ``workspace``.

    The workspace boundary is enforced using :meth:`Path.resolve` so that
    symlinks and ``..`` traversal cannot escape the sandbox. A
    :class:`PermissionError` is raised when the resolved path leaves the
    workspace.
    """

    workspace_root = workspace.resolve()
    candidate = Path(target_path)
    resolved = candidate if candidate.is_absolute() else workspace_root / candidate
    resolved = resolved.resolve()

    if resolved != workspace_root and workspace_root not in resolved.parents:
        raise PermissionError(
            f"Path '{target_path}' escapes the workspace boundary "
            f"('{workspace_root}')."
        )
    return resolved


class LocalEnvironment:
    """Sandboxed asynchronous command runner bound to a workspace directory."""

    def __init__(self, workspace: Path, command_timeout: int = 45) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.command_timeout = command_timeout

    def resolve(self, target_path: str) -> Path:
        """Public helper to validate a path against the workspace."""
        return validate_path(target_path, self.workspace)

    async def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        """Run a shell command inside the workspace with a hard timeout."""

        effective_timeout = timeout or self.command_timeout
        logger.info("Executing command (timeout=%ss): %s", effective_timeout, command)

        # Run in a dedicated process group so the whole tree (shell + children)
        # can be terminated reliably on timeout.
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            start_new_session=True,
        )

        communicate = asyncio.ensure_future(process.communicate())
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            await self._terminate(process)
            # Drain the (now-finished) communicate task to release the pipes
            # cleanly and avoid noisy transport warnings at GC time.
            try:
                await communicate
            except (asyncio.CancelledError, ProcessLookupError):
                pass
            logger.warning("Command timed out after %ss: %s", effective_timeout, command)
            return CommandResult(
                command=command,
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {effective_timeout} seconds.",
                timed_out=True,
            )

        result = CommandResult(
            command=command,
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )
        logger.info("Command exited with code %s", result.exit_code)
        return result

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """Terminate (then kill) the process group that exceeded its timeout."""
        if process.returncode is not None:
            return

        def _signal_group(sig: int) -> None:
            try:
                os.killpg(os.getpgid(process.pid), sig)
            except ProcessLookupError:
                pass

        _signal_group(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            _signal_group(signal.SIGKILL)
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:  # pragma: no cover - extreme edge
                logger.error("Failed to kill process group for pid %s", process.pid)
