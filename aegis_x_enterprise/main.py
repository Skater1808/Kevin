"""Application entrypoint.

Configures colored logging and launches the FastAPI dashboard (and the agent
background task plumbing) via Uvicorn.
"""

from __future__ import annotations

import logging

import uvicorn

from config import get_settings


class _ColorFormatter(logging.Formatter):
    """Minimal ANSI color formatter for transparent state/tool logging."""

    COLORS = {
        logging.DEBUG: "\033[38;5;245m",
        logging.INFO: "\033[38;5;39m",
        logging.WARNING: "\033[38;5;214m",
        logging.ERROR: "\033[38;5;196m",
        logging.CRITICAL: "\033[48;5;196m\033[97m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def main() -> None:
    setup_logging()
    settings = get_settings()
    logger = logging.getLogger("aegis.main")
    logger.info("Starting Aegis-X Enterprise on http://%s:%s", settings.host, settings.port)
    logger.info("Provider: %s | Model: %s | Workspace: %s", settings.provider.value, settings.effective_model, settings.workspace_dir)
    uvicorn.run(
        "ui.app:app",
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
