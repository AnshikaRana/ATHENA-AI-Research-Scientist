"""``SQLiteSession`` - durable conversation memory for a session id.

Runs that share a session id share a transcript, so a follow-up question knows
what earlier runs established. The store keeps ordered transcript *items* (the
same plain dicts the runner passes around) in a single SQLite table.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    item_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_items_session
    ON session_items (session_id, id);
"""


class SQLiteSession:
    """A minimal, async-facing transcript store backed by SQLite."""

    def __init__(self, session_id: str, db_path: str) -> None:
        self.session_id = session_id
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            query = "SELECT item_json FROM session_items WHERE session_id = ? ORDER BY id"
            params: tuple[Any, ...] = (self.session_id,)
            if limit is not None:
                query += " LIMIT ?"
                params = (self.session_id, int(limit))
            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()
        return [json.loads(row["item_json"]) for row in rows]

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        async with self._lock:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO session_items (session_id, item_json) VALUES (?, ?)",
                    [(self.session_id, json.dumps(item, default=str)) for item in items],
                )

    async def clear_session(self) -> None:
        async with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM session_items WHERE session_id = ?", (self.session_id,)
                )
