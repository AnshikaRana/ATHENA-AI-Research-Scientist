"""Agent-facing long-term memory: write durable facts, recall them later."""

from __future__ import annotations

from athena.runtime import RunContextWrapper, function_tool

from ..context import ResearchContext
from ..logging_setup import get_logger
from ..memory.long_term import LongTermMemory

logger = get_logger("tools.memory")

_store: LongTermMemory | None = None


def get_memory() -> LongTermMemory:
    """Process-wide long-term memory handle (the DB file is the real state)."""
    global _store
    if _store is None:
        _store = LongTermMemory()
    return _store


@function_tool
async def remember_finding(
    wrapper: RunContextWrapper[ResearchContext],
    kind: str,
    topic: str,
    content: str,
) -> str:
    """Save a durable fact that future research sessions should know.

    Use sparingly, for conclusions that generalise beyond this run - a confirmed
    finding, a recurring methodological pitfall, or a stated user preference.

    Args:
        kind: One of "finding", "question", "preference", "critique".
        topic: A short subject label used for grouping, e.g. "GNN expressivity".
        content: The fact itself, self-contained and understandable months later.
    """
    ctx = wrapper.context
    valid = {"finding", "question", "preference", "critique"}
    normalised = kind if kind in valid else "finding"
    memory_id = await get_memory().remember(
        session_id=ctx.session_id,
        run_id=ctx.run_id,
        kind=normalised,  # type: ignore[arg-type]
        topic=topic,
        content=content,
    )
    return f"Stored long-term memory #{memory_id} ({normalised}: {topic})."


@function_tool
async def recall_memory(
    wrapper: RunContextWrapper[ResearchContext], query: str, limit: int
) -> str:
    """Search everything Athena has learned in previous sessions.

    Call this at the start of a run: prior work on the same topic saves the
    whole pipeline from rediscovering what is already known.

    Args:
        query: What you want to remember, in natural language.
        limit: Maximum memories to return (1-10).
    """
    k = max(1, min(int(limit or 5), 10))
    records = await get_memory().recall(query, k=k)
    if not records:
        return "No prior memories match. This appears to be a new topic."
    lines = [f"Recalled {len(records)} prior memories:"]
    for record in records:
        lines.append(
            f"- [{record.kind}] {record.topic} (relevance {record.score:.2f}, "
            f"from {record.created_at[:10]}): {record.content}"
        )
    return "\n".join(lines)
