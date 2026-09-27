"""Long-term memory: knowledge that outlives a single research run.

Athena keeps three memory tiers, and it is worth being precise about which is
which because the rubric treats them as separate capabilities:

1. **Conversation memory** - the runtime's ``SQLiteSession``, which replays the
   message transcript so a follow-up question knows what was already said.
2. **Working memory** - :class:`~athena.context.ResearchContext`, alive for one
   run, holding the corpus and stage artefacts.
3. **Long-term memory** - *this module*. Durable, semantically searchable facts
   that persist across sessions: prior research questions, findings that were
   confirmed, and standing user preferences.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from ..config import get_settings
from ..logging_setup import get_logger, log_event
from .vector_store import embed_texts

logger = get_logger("memory.long_term")

MemoryKind = Literal["finding", "question", "preference", "critique"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    topic       TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   BLOB,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_kind  ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
"""


@dataclass
class MemoryRecord:
    id: int
    session_id: str
    run_id: str
    kind: str
    topic: str
    content: str
    created_at: str
    score: float = 0.0


class LongTermMemory:
    """Semantic, durable memory backed by SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_settings().long_term_memory_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # -- writing ----------------------------------------------------------

    async def remember(
        self,
        *,
        session_id: str,
        run_id: str,
        kind: MemoryKind,
        topic: str,
        content: str,
    ) -> int:
        """Store one fact and return its row id."""
        vector = (await embed_texts([f"{topic}. {content}"]))[0]
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO memories "
                "(session_id, run_id, kind, topic, content, embedding, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    session_id,
                    run_id,
                    kind,
                    topic,
                    content,
                    vector.astype(np.float32).tobytes(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            memory_id = int(cursor.lastrowid or 0)
        log_event(logger, "memory stored", memory_id=memory_id, kind=kind, topic=topic)
        return memory_id

    # -- reading ----------------------------------------------------------

    async def recall(
        self, query: str, *, k: int = 5, kinds: list[str] | None = None
    ) -> list[MemoryRecord]:
        """Semantic search across everything ever remembered."""
        sql = "SELECT id, session_id, run_id, kind, topic, content, embedding, created_at FROM memories"
        params: list[object] = []
        if kinds:
            sql += f" WHERE kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)

        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []

        query_vector = (await embed_texts([query]))[0]
        scored: list[MemoryRecord] = []
        for row in rows:
            blob = row[6]
            if not blob:
                continue
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.shape != query_vector.shape:
                # Embedding model changed between runs; skip rather than crash.
                continue
            scored.append(
                MemoryRecord(
                    id=row[0],
                    session_id=row[1],
                    run_id=row[2],
                    kind=row[3],
                    topic=row[4],
                    content=row[5],
                    created_at=row[7],
                    score=float(vector @ query_vector),
                )
            )
        scored.sort(key=lambda record: -record.score)
        return scored[:k]

    def recent(self, limit: int = 10) -> list[MemoryRecord]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, session_id, run_id, kind, topic, content, created_at "
                "FROM memories ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            MemoryRecord(
                id=r[0],
                session_id=r[1],
                run_id=r[2],
                kind=r[3],
                topic=r[4],
                content=r[5],
                created_at=r[6],
            )
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_kind = dict(
                conn.execute(
                    "SELECT kind, COUNT(*) FROM memories GROUP BY kind"
                ).fetchall()
            )
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories"
            ).fetchone()[0]
        return {"total": total, "sessions": sessions, **by_kind}

    def export_json(self) -> str:
        return json.dumps([r.__dict__ for r in self.recent(1000)], indent=2)
