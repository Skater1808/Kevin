"""Context history and state persistence.

``Memory`` keeps the running interaction history (persisted to SQLite) and
provides JSON persistence for the agent's structured ``agent_state.json``
document that lives in the core directory.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("aegis.memory")

CORE_DIR: Path = Path(__file__).resolve().parent
STATE_FILE: Path = CORE_DIR / "agent_state.json"
HISTORY_DB: Path = CORE_DIR / "history.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryEntry:
    """A single entry in the agent's interaction history."""

    role: str
    content: str
    timestamp: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class Memory:
    """SQLite-backed interaction history plus JSON state persistence."""

    def __init__(self, db_path: Path = HISTORY_DB, state_path: Path = STATE_FILE) -> None:
        self.db_path = db_path
        self.state_path = state_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add(self, role: str, content: str) -> MemoryEntry:
        entry = MemoryEntry(role=role, content=content, timestamp=_utc_now())
        with self._lock:
            self._conn.execute(
                "INSERT INTO history (role, content, timestamp) VALUES (?, ?, ?)",
                (entry.role, entry.content, entry.timestamp),
            )
            self._conn.commit()
        return entry

    def recent(self, limit: int = 20) -> list[MemoryEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, timestamp FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [MemoryEntry(role=r, content=c, timestamp=t) for r, c, t in reversed(rows)]

    def transcript(self, limit: int = 20) -> str:
        """Return a flattened transcript of recent history for prompting."""
        lines = [f"[{entry.role}] {entry.content}" for entry in self.recent(limit)]
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM history")
            self._conn.commit()

    # --- JSON state persistence ------------------------------------------- #
    def save_state(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt agent_state.json; ignoring.")
            return None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
