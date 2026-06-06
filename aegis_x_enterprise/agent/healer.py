"""Hermes-style self-healing module.

When a subprocess command or unit test exits with a non-zero status, the
:class:`AgentCore` pauses and delegates to :class:`HealerModule`. The healer
asks the LLM to reflect on the failure, returns a corrected code block plus a
one-sentence explanation, and tracks attempts per failure location. After
``max_attempts`` unsuccessful heals at the same location it escalates.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from config import LLMClient, LLMError

logger = logging.getLogger("aegis.healer")

_REFLECTION_SYSTEM = (
    "You are Hermes, an elite debugging agent. You analyze failing code and "
    "produce a corrected version. You always answer with a single JSON object."
)


def parse_llm_json(raw: str) -> dict | None:
    """Best-effort extraction of the first JSON object from an LLM response."""
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


@dataclass(slots=True)
class HealingResult:
    """Outcome of a single healing attempt."""

    success: bool
    corrected_code: str
    explanation: str
    attempts: int
    escalated: bool
    error: str = ""


class HealerModule:
    """Iterative reflexion-based code patcher."""

    def __init__(self, llm: LLMClient, max_attempts: int = 3) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self._attempts: dict[str, int] = {}

    def attempts_for(self, location: str) -> int:
        return self._attempts.get(location, 0)

    def reset(self, location: str | None = None) -> None:
        if location is None:
            self._attempts.clear()
        else:
            self._attempts.pop(location, None)

    async def heal_code(
        self,
        failed_code: str,
        command: str,
        stderr: str,
        location: str,
    ) -> HealingResult:
        """Reflect on a failure and return a corrected code block."""

        attempt = self._attempts.get(location, 0) + 1
        self._attempts[location] = attempt
        logger.warning("Healing attempt %d/%d for '%s'", attempt, self.max_attempts, location)

        if attempt > self.max_attempts:
            return HealingResult(
                success=False,
                corrected_code=failed_code,
                explanation="",
                attempts=attempt,
                escalated=True,
                error=f"Exceeded {self.max_attempts} healing attempts at '{location}'.",
            )

        prompt = (
            "You produced the following error:\n"
            f"--- COMMAND ---\n{command}\n"
            f"--- STDERR ---\n{stderr}\n"
            f"--- CURRENT CODE ---\n{failed_code}\n\n"
            "Analyze the root cause (syntax, missing library, logic error). "
            "Generate a corrected code block and explain the correction in exactly one sentence. "
            'Respond ONLY with JSON: {"corrected_code": "<full corrected file>", '
            '"explanation": "<one sentence>"}.'
        )

        try:
            raw = await self.llm.complete(prompt, system=_REFLECTION_SYSTEM)
        except LLMError as exc:
            return HealingResult(
                success=False,
                corrected_code=failed_code,
                explanation="",
                attempts=attempt,
                escalated=attempt >= self.max_attempts,
                error=str(exc),
            )

        parsed = parse_llm_json(raw)
        if not parsed or "corrected_code" not in parsed:
            return HealingResult(
                success=False,
                corrected_code=failed_code,
                explanation="",
                attempts=attempt,
                escalated=attempt >= self.max_attempts,
                error="Healer could not parse a corrected code block from the LLM response.",
            )

        return HealingResult(
            success=True,
            corrected_code=str(parsed["corrected_code"]),
            explanation=str(parsed.get("explanation", "")).strip(),
            attempts=attempt,
            escalated=False,
        )
