"""Artefact persistence, gated behind human approval.

This is where the human-in-the-loop requirement lives. Two different gating
styles are demonstrated deliberately, because they answer different questions:

* ``save_research_artifact`` uses **static** approval - writing a research
  dossier to disk under the user's name is a consequential, outward-facing act,
  so a person signs off every time.
* ``commission_deep_sweep`` uses **dynamic** approval - a narrow search runs
  freely, but one wide enough to cost real money stops for authorisation. The
  gate is a function of the arguments, not of the tool.

When a gated tool is called, the SDK suspends the run and surfaces a
``ToolApprovalItem``. The pipeline serialises that state, gets a decision from
the CLI or the Streamlit UI, and resumes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from athena.runtime import RunContextWrapper, function_tool

from ..config import get_settings
from ..context import ResearchContext
from ..logging_setup import get_logger, log_event

logger = get_logger("tools.artifacts")

# A sweep wider than this is expensive enough to need a human to say yes.
DEEP_SWEEP_APPROVAL_THRESHOLD = 30


async def _approve_if_expensive(
    wrapper: RunContextWrapper[ResearchContext],
    params: dict[str, Any],
    call_id: str,
) -> bool:
    """Dynamic gate: only interrupt the run when the request is costly."""
    requested = int(params.get("total_results") or 0)
    needs_gate = requested > DEEP_SWEEP_APPROVAL_THRESHOLD
    log_event(
        logger,
        "approval policy evaluated",
        tool="commission_deep_sweep",
        requested=requested,
        threshold=DEEP_SWEEP_APPROVAL_THRESHOLD,
        approval_required=needs_gate,
        call_id=call_id,
    )
    return needs_gate


def _approval_required(
    wrapper: RunContextWrapper[ResearchContext],
    params: dict[str, Any],
    call_id: str,
) -> bool:
    """Static gate, still respecting the global opt-out in settings.

    The human rules once per run. A writer that saves again after an approved
    save (a revision, or a model repeating itself) is not asking a new question,
    and re-gating it turned one click into a loop of pauses and duplicate files.
    """
    if not get_settings().require_human_approval:
        return False
    ctx = getattr(wrapper, "context", None)
    already_approved = any(
        a.tool_name == "save_research_artifact" and a.decision == "approved"
        for a in getattr(ctx, "approvals", [])
    )
    return not already_approved


@function_tool(needs_approval=_approval_required)  # type: ignore[arg-type]
async def save_research_artifact(
    wrapper: RunContextWrapper[ResearchContext],
    filename: str,
    content: str,
    artifact_type: str,
) -> str:
    """Write a research artefact to disk. Requires human approval.

    Args:
        filename: Base name without a path, e.g. "gnn_expressivity_report".
        content: The full Markdown or JSON body to persist.
        artifact_type: One of "report", "design", "synthesis", "notes".
    """
    ctx = wrapper.context
    settings = get_settings()

    safe_stem = "".join(
        ch for ch in filename if ch.isalnum() or ch in ("-", "_", ".")
    ).strip(".") or "artifact"
    extension = ".json" if content.lstrip().startswith(("{", "[")) else ".md"
    # One file per run and type: a repeated save overwrites rather than piling
    # up near-identical copies.
    previous = ctx.saved_artifacts.get(artifact_type) if ctx is not None else None
    if previous and Path(previous).suffix == extension:
        path = Path(previous)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = settings.artifacts_dir / f"{stamp}_{safe_stem}{extension}"
    if ctx is not None:
        ctx.saved_artifacts[artifact_type] = str(path)
    path.write_text(content, encoding="utf-8")

    # Sidecar provenance: what produced this file, and under whose approval.
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "run_id": ctx.run_id,
                "session_id": ctx.session_id,
                "research_question": ctx.research_question,
                "artifact_type": artifact_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "corpus_size": len(ctx.corpus),
                "reflection_rounds": ctx.reflection_round,
                "approvals": [a.__dict__ for a in ctx.approvals],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_event(
        logger,
        "artifact saved",
        path=str(path),
        artifact_type=artifact_type,
        bytes=len(content.encode("utf-8")),
    )
    return f"Saved {artifact_type} to {path.name} ({len(content):,} chars), with provenance sidecar."


@function_tool(needs_approval=_approve_if_expensive)  # type: ignore[arg-type]
async def commission_deep_sweep(
    wrapper: RunContextWrapper[ResearchContext],
    queries: list[str],
    total_results: int,
) -> str:
    """Run a wide multi-query literature sweep. Large sweeps need approval.

    Sweeps requesting more than 30 total results pause for human authorisation
    because of their API cost; smaller ones proceed immediately.

    Args:
        queries: Distinct search phrases to run, typically 2-5.
        total_results: Total records to retrieve across all queries.
    """
    ctx = wrapper.context
    settings = get_settings()
    from .literature import SOURCES

    import asyncio

    queries = [q for q in (queries or []) if q.strip()][:5]
    if not queries:
        return "No queries supplied."

    ceiling = settings.max_papers_per_source
    per_source = max(
        1, min(total_results // (len(queries) * len(SOURCES)) or 1, ceiling)
    )

    tasks = [
        search(query, per_source) for query in queries for search in SOURCES.values()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    papers = []
    failures = 0
    for result in results:
        if isinstance(result, Exception):
            failures += 1
            ctx.record_error(f"deep sweep leg failed: {result}")
            continue
        papers.extend(result)

    added = ctx.add_papers(papers)
    ctx.costs.api_calls += len(tasks)
    log_event(
        logger,
        "deep sweep complete",
        queries=queries,
        legs=len(tasks),
        failures=failures,
        retrieved=len(papers),
        added=added,
    )
    return (
        f"Deep sweep over {len(queries)} queries x {len(SOURCES)} sources "
        f"({len(tasks)} parallel calls, {failures} failed): retrieved {len(papers)} "
        f"records, {added} new. Corpus now holds {len(ctx.corpus)} papers."
    )
