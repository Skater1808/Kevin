"""Abstract base classes for tools and plugins.

A :class:`BaseTool` is a single callable capability exposed to the agent. A
:class:`BasePlugin` is a packaged extension that registers one or more tools
with the :class:`~tools.registry.ToolRegistry` at load time.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolResult:
    """Normalized result returned by every tool invocation."""

    ok: bool
    output: str
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "data": self.data,
        }


class BaseTool(abc.ABC):
    """Abstract base class for an asynchronous agent tool."""

    #: Unique tool name used by the agent to address the tool.
    name: str = ""
    #: Human readable description surfaced to the LLM during planning.
    description: str = ""
    #: JSON-schema-like mapping describing accepted arguments.
    parameters: dict[str, str] = {}

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a 'name'.")

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a :class:`ToolResult`."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """Return a serializable specification for prompting the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class BasePlugin(abc.ABC):
    """Abstract base class for a dynamically loadable plugin."""

    #: Unique plugin name.
    name: str = ""
    #: Short plugin description.
    description: str = ""

    @abc.abstractmethod
    def register_tools(self, registry: "ToolRegistry") -> None:
        """Register the plugin's tools with the provided registry."""
        raise NotImplementedError
