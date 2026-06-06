"""Asynchronous ReAct loop and agent state machine.

``AgentCore`` drives the agent through a strict state machine
(``IDLE -> PLANNING -> EXECUTING -> TESTING -> HEALING -> COMPLETED|FAILED``)
following an asynchronous Reasoning + Acting (ReAct) pattern. All state is
persisted to ``agent_state.json`` and every transition, tool execution and LLM
reflection is pushed to subscribers through an async event emitter.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from agent.healer import HealerModule, parse_llm_json
from agent.memory import Memory
from config import LLMClient, LLMError, Settings
from execution.local_env import LocalEnvironment
from tools.registry import ToolRegistry

logger = logging.getLogger("aegis.core")

Emitter = Callable[[dict[str, Any]], Awaitable[None]]


class AgentState(str, enum.Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    TESTING = "TESTING"
    HEALING = "HEALING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class EventType(str, enum.Enum):
    STATUS = "status"
    LOG = "log"
    TASK_UPDATE = "task_update"
    ERROR = "error"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Task:
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TaskStatus = TaskStatus.PENDING
    error_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "error_log": self.error_log,
        }


_PLANNER_SYSTEM = (
    "You are Aegis-X, an autonomous principal engineer. You decompose goals into "
    "a concise ordered list of concrete, verifiable engineering tasks."
)
_ACTOR_SYSTEM = (
    "You are Aegis-X executing a single task with tools. Think step by step and "
    "act using exactly one tool per step. You always answer with a single JSON object."
)


class AgentCore:
    """The asynchronous autonomous agent core."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        registry: ToolRegistry,
        env: LocalEnvironment,
        memory: Memory,
        healer: HealerModule,
        emitter: Optional[Emitter] = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.registry = registry
        self.env = env
        self.memory = memory
        self.healer = healer
        self._emitter = emitter

        self.state: AgentState = AgentState.IDLE
        self.goal: str = ""
        self.task_tree: list[Task] = []
        self.iteration_count: int = 0

        self._pause_event = asyncio.Event()
        self._pause_event.set()  # not paused
        self._stop_requested = False
        self._task: asyncio.Task[None] | None = None
        self._last_artifact: dict[str, str] | None = None

    # --- public control surface ------------------------------------------- #
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, goal: str) -> bool:
        if self.is_running:
            await self._emit(EventType.LOG, {"message": "Agent already running; ignoring start."})
            return False
        self.goal = goal
        self.task_tree = []
        self.iteration_count = 0
        self._stop_requested = False
        self._pause_event.set()
        self.healer.reset()
        self._task = asyncio.create_task(self._run())
        return True

    def pause(self) -> None:
        self._pause_event.clear()
        logger.info("Pause requested.")

    def resume(self) -> None:
        self._pause_event.set()
        logger.info("Resume requested.")

    async def stop(self) -> None:
        self._stop_requested = True
        self._pause_event.set()  # release any pause so the loop can observe the stop
        logger.warning("Emergency stop requested.")
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    def snapshot(self) -> dict[str, Any]:
        """Return the structured state document."""
        return {
            "current_state": self.state.value,
            "goal": self.goal,
            "iteration_count": self.iteration_count,
            "max_iterations": self.settings.max_iterations,
            "task_tree": [t.to_dict() for t in self.task_tree],
            "timestamp": _utc_now(),
        }

    # --- event + state plumbing ------------------------------------------- #
    def set_emitter(self, emitter: Emitter) -> None:
        self._emitter = emitter

    async def _emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        packet = {
            "type": event_type.value,
            "payload": payload,
            "timestamp": _utc_now(),
        }
        if self._emitter is not None:
            try:
                await self._emitter(packet)
            except Exception:  # noqa: BLE001 - broadcasting must never crash the loop
                logger.exception("Event emitter failed")

    async def _transition(self, new_state: AgentState, message: str = "") -> None:
        logger.info("State transition: %s -> %s %s", self.state.value, new_state.value, message)
        self.state = new_state
        self.memory.save_state(self.snapshot())
        await self._emit(
            EventType.STATUS,
            {"state": new_state.value, "message": message, "snapshot": self.snapshot()},
        )

    async def _log(self, message: str, level: str = "info") -> None:
        getattr(logger, level, logger.info)(message)
        await self._emit(EventType.LOG, {"message": message, "level": level})

    async def _emit_task_update(self, task: Task) -> None:
        self.memory.save_state(self.snapshot())
        await self._emit(EventType.TASK_UPDATE, {"task": task.to_dict(), "snapshot": self.snapshot()})

    async def _fail(self, reason: str) -> None:
        await self._emit(EventType.ERROR, {"message": reason})
        await self._transition(AgentState.FAILED, reason)

    async def _wait_if_paused(self) -> None:
        if not self._pause_event.is_set():
            await self._log("Agent paused.")
            await self._pause_event.wait()
            if not self._stop_requested:
                await self._log("Agent resumed.")

    # --- main loop -------------------------------------------------------- #
    async def _run(self) -> None:
        try:
            await self._transition(AgentState.PLANNING, "Decomposing goal into tasks.")
            self.memory.add("user", self.goal)
            await self._plan()
            if self._stop_requested:
                await self._fail("Stopped during planning.")
                return
            if not self.task_tree:
                await self._fail("Planning produced no tasks.")
                return

            for task in self.task_tree:
                if self._stop_requested:
                    await self._fail("Emergency stop.")
                    return
                await self._wait_if_paused()
                if self._stop_requested:
                    await self._fail("Emergency stop.")
                    return

                task.status = TaskStatus.RUNNING
                await self._transition(AgentState.EXECUTING, f"Task: {task.title}")
                await self._emit_task_update(task)

                ok = await self._execute_task(task)
                if not ok:
                    task.status = TaskStatus.FAILED
                    await self._emit_task_update(task)
                    await self._fail(f"Task failed: {task.title}")
                    return

                task.status = TaskStatus.DONE
                await self._emit_task_update(task)

            await self._transition(AgentState.COMPLETED, "All tasks completed successfully.")
        except LLMError as exc:
            await self._fail(f"LLM error: {exc}")
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("Unhandled error in agent loop")
            await self._fail(f"Unhandled error: {exc}")

    async def _plan(self) -> None:
        prompt = (
            f"Goal:\n{self.goal}\n\n"
            "Decompose this into an ordered list of concrete engineering tasks. "
            "Keep it minimal but complete. "
            'Respond ONLY with JSON: {"tasks": [{"title": "..."}, ...]}.'
        )
        raw = await self.llm.complete(prompt, system=_PLANNER_SYSTEM)
        parsed = parse_llm_json(raw)
        tasks_data = (parsed or {}).get("tasks", []) if isinstance(parsed, dict) else []
        for item in tasks_data:
            title = str(item.get("title", "")).strip() if isinstance(item, dict) else str(item).strip()
            if title:
                self.task_tree.append(Task(title=title))
        await self._log(f"Planned {len(self.task_tree)} task(s).")
        for task in self.task_tree:
            await self._emit_task_update(task)

    async def _execute_task(self, task: Task, max_steps: int = 8) -> bool:
        """Run a bounded ReAct sub-loop for a single task."""
        for _ in range(max_steps):
            if self._stop_requested:
                return False
            await self._wait_if_paused()

            self.iteration_count += 1
            if self.iteration_count > self.settings.max_iterations:
                task.error_log.append("Global iteration hard-limit reached.")
                await self._log("Global iteration hard-limit reached.", level="error")
                return False

            decision = await self._decide(task)
            if decision is None:
                task.error_log.append("Could not parse an action from the LLM.")
                return False

            if decision.get("final"):
                await self._log(f"Task '{task.title}' finished: {decision['final']}")
                self.memory.add("assistant", f"Finished '{task.title}': {decision['final']}")
                return True

            action = decision.get("action") or {}
            tool_name = str(action.get("tool", ""))
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            await self._log(f"Action: {tool_name} {args}")
            result = await self.registry.execute(tool_name, args)

            if tool_name == "write_file" and result.ok:
                self._last_artifact = {"path": str(args.get("path", "")), "content": str(args.get("content", ""))}

            observation = result.output if result.ok else f"ERROR: {result.error}"
            self.memory.add("tool", f"{tool_name} -> {observation[:1500]}")
            await self._emit(
                EventType.LOG,
                {"message": observation[:4000], "level": "info" if result.ok else "error", "tool": tool_name},
            )

            # A failing terminal command triggers the testing/healing pipeline.
            if tool_name == "run_command" and not result.ok and not result.data.get("timed_out"):
                healed = await self._heal_cycle(task, result)
                if not healed:
                    return False

        task.error_log.append("Reached the per-task step limit without finishing.")
        await self._log(f"Task '{task.title}' exhausted its step budget.", level="error")
        return False

    async def _decide(self, task: Task) -> dict[str, Any] | None:
        tools_desc = "\n".join(
            f"- {spec['name']}: {spec['description']} args={list(spec['parameters'])}"
            for spec in self.registry.specs()
        )
        prompt = (
            f"Overall goal: {self.goal}\n"
            f"Current task: {task.title}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Recent history:\n{self.memory.transcript(12)}\n\n"
            "Decide the next single step. To call a tool respond ONLY with JSON: "
            '{"thought": "...", "action": {"tool": "<name>", "args": {...}}}. '
            "When the task is fully complete respond ONLY with JSON: "
            '{"thought": "...", "final": "<summary>"}.'
        )
        raw = await self.llm.complete(prompt, system=_ACTOR_SYSTEM)
        decision = parse_llm_json(raw)
        if isinstance(decision, dict) and decision.get("thought"):
            await self._emit(EventType.LOG, {"message": f"Reflection: {decision['thought']}", "level": "debug"})
        return decision if isinstance(decision, dict) else None

    async def _heal_cycle(self, task: Task, failed_result: Any) -> bool:
        """Iteratively heal the most recent artifact until the command passes."""
        command = str(failed_result.data.get("command", ""))
        location = self._last_artifact["path"] if self._last_artifact else command or "unknown"

        while True:
            if self._stop_requested:
                return False
            await self._transition(AgentState.HEALING, f"Healing failure at '{location}'.")

            failed_code = self._last_artifact["content"] if self._last_artifact else ""
            stderr = str(failed_result.data.get("stderr", "")) or failed_result.error
            result = await self.healer.heal_code(failed_code, command, stderr, location)

            if result.escalated or (not result.success and self.healer.attempts_for(location) >= self.settings.max_healing_attempts):
                task.error_log.append(result.error or "Healing escalated.")
                await self._log(f"Healing escalated for '{location}': {result.error}", level="error")
                return False
            if not result.success:
                task.error_log.append(result.error or "Healing attempt failed.")
                await self._log(f"Healing attempt failed: {result.error}", level="error")
                continue

            await self._log(f"Healer fix ({result.attempts}): {result.explanation}")

            # Apply the correction directly to the failing artifact.
            if self._last_artifact:
                apply = await self.registry.execute(
                    "write_file",
                    {"path": location, "content": result.corrected_code},
                )
                if apply.ok:
                    self._last_artifact = {"path": location, "content": result.corrected_code}

            await self._transition(AgentState.TESTING, f"Re-running: {command}")
            failed_result = await self.registry.execute("run_command", {"command": command})
            if failed_result.ok:
                await self._log("Healing succeeded; command now passes.")
                self.healer.reset(location)
                await self._transition(AgentState.EXECUTING, f"Task: {task.title}")
                return True
