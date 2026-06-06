"""Central configuration and LLM initialization for Aegis-X Enterprise.

This module loads configuration from a ``.env`` file using ``pydantic-settings``
(v2) and exposes a provider-agnostic asynchronous LLM client. Provider SDKs are
imported lazily so the application can start and the dashboard can render even
when an individual SDK is not installed.
"""

from __future__ import annotations

import enum
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("aegis.config")

# Absolute path to the project root (directory containing this file).
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_WORKSPACE: Path = PROJECT_ROOT / "workspace"
ENV_FILE: Path = PROJECT_ROOT / ".env"


class LLMProvider(str, enum.Enum):
    """Supported large language model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


#: Sensible default model identifiers per provider.
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-latest",
    LLMProvider.GEMINI: "gemini-1.5-pro",
    LLMProvider.OLLAMA: "llama3.1",
}


class Settings(BaseSettings):
    """Strongly typed application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Active LLM provider.",
    )
    model: str = Field(
        default="",
        description="Model identifier. Falls back to a provider default when empty.",
    )
    api_key: str = Field(
        default="",
        description="API key for the active provider (not required for Ollama).",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    workspace_dir: Path = Field(
        default=DEFAULT_WORKSPACE,
        description="Absolute path of the isolated agent workspace.",
    )

    # --- Runtime safety knobs -------------------------------------------------
    command_timeout: int = Field(
        default=45,
        ge=1,
        description="Hard timeout (seconds) for any terminal command.",
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        description="Hard cap on the agent's main loop iterations.",
    )
    max_healing_attempts: int = Field(
        default=3,
        ge=1,
        description="Healing attempts at the same location before escalation.",
    )

    host: str = Field(default="127.0.0.1", description="Web server bind host.")
    port: int = Field(default=8000, ge=1, le=65535, description="Web server port.")

    @field_validator("workspace_dir")
    @classmethod
    def _resolve_workspace(cls, value: Path) -> Path:
        resolved = Path(value).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @property
    def effective_model(self) -> str:
        """Return the configured model or the provider default."""
        return self.model or DEFAULT_MODELS[self.provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


class LLMError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns an error."""


class LLMClient:
    """Provider-agnostic asynchronous LLM wrapper.

    A single :meth:`complete` coroutine is exposed regardless of the backing
    provider. Provider SDKs are imported lazily inside the relevant branch.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.provider
        self.model = settings.effective_model

    def _require_key(self) -> str:
        if not self.settings.api_key:
            raise LLMError(
                f"No API key configured for provider '{self.provider.value}'. "
                "Run `python setup.py` to configure credentials."
            )
        return self.settings.api_key

    async def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a completion for ``prompt`` with an optional ``system`` prompt."""
        logger.debug("LLM completion via %s (%s)", self.provider.value, self.model)
        if self.provider is LLMProvider.OPENAI:
            return await self._complete_openai(prompt, system)
        if self.provider is LLMProvider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system)
        if self.provider is LLMProvider.GEMINI:
            return await self._complete_gemini(prompt, system)
        if self.provider is LLMProvider.OLLAMA:
            return await self._complete_ollama(prompt, system)
        raise LLMError(f"Unsupported provider: {self.provider}")

    async def _complete_openai(self, prompt: str, system: Optional[str]) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError("The 'openai' package is not installed.") from exc

        client = AsyncOpenAI(api_key=self._require_key())
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001 - normalize SDK errors
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content
        return content or ""

    async def _complete_anthropic(self, prompt: str, system: Optional[str]) -> str:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError("The 'anthropic' package is not installed.") from exc

        client = AsyncAnthropic(api_key=self._require_key())
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(parts)

    async def _complete_gemini(self, prompt: str, system: Optional[str]) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError("The 'google-genai' package is not installed.") from exc

        client = genai.Client(api_key=self._require_key())
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.1,
        )
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return response.text or ""

    async def _complete_ollama(self, prompt: str, system: Optional[str]) -> str:
        import httpx

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        url = self.settings.ollama_base_url.rstrip("/") + "/api/generate"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Ollama request failed: {exc}") from exc
        data = response.json()
        return str(data.get("response", ""))


def build_llm_client(settings: Optional[Settings] = None) -> LLMClient:
    """Instantiate an :class:`LLMClient` from the given (or cached) settings."""
    return LLMClient(settings or get_settings())
