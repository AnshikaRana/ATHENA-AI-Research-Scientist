"""Lifecycle hooks: observability, cost accounting and the handoff audit trail.

``AthenaRunHooks`` is attached to every ``Runner.run`` call. It is the single
place where we learn what the orchestration actually did, which is what makes
the agent trace in the UI and the JSONL audit log possible.
"""

from __future__ import annotations

import time
from typing import Any

from athena.runtime import Agent, AgentHookContext, RunContextWrapper, RunHooks, Tool
from athena.runtime import ModelResponse

from .context import ResearchContext
from .logging_setup import get_logger, log_event

logger = get_logger("hooks")


class AthenaRunHooks(RunHooks[ResearchContext]):
    """Record every agent start/end, handoff, tool call and model response."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._tool_started_at: dict[str, float] = {}
        self._agent_started_at: dict[str, float] = {}

    # -- internal ---------------------------------------------------------

    def _emit(self, kind: str, **fields: Any) -> None:
        event = {"kind": kind, "ts": time.time(), **fields}
        self.events.append(event)
        log_event(logger, f"{kind}: {fields.get('detail', '')}".strip(), **event)

    # -- agents -----------------------------------------------------------

    async def on_agent_start(
        self, context: AgentHookContext[ResearchContext], agent: Agent[ResearchContext]
    ) -> None:
        self._agent_started_at[agent.name] = time.time()
        ctx = context.context
        self._emit(
            "agent_start",
            agent=agent.name,
            stage=ctx.stage.value if ctx else None,
            detail=agent.name,
        )

    async def on_agent_end(
        self,
        context: AgentHookContext[ResearchContext],
        agent: Agent[ResearchContext],
        output: Any,
    ) -> None:
        started = self._agent_started_at.pop(agent.name, None)
        self._emit(
            "agent_end",
            agent=agent.name,
            output_type=type(output).__name__,
            duration_s=round(time.time() - started, 3) if started else None,
            detail=f"{agent.name} -> {type(output).__name__}",
        )

    # -- handoffs ---------------------------------------------------------

    async def on_handoff(
        self,
        context: RunContextWrapper[ResearchContext],
        from_agent: Agent[ResearchContext],
        to_agent: Agent[ResearchContext],
    ) -> None:
        self._emit(
            "handoff",
            from_agent=from_agent.name,
            to_agent=to_agent.name,
            detail=f"{from_agent.name} -> {to_agent.name}",
        )

    # -- tools ------------------------------------------------------------

    async def on_tool_start(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent[ResearchContext],
        tool: Tool,
    ) -> None:
        key = f"{agent.name}:{tool.name}"
        self._tool_started_at[key] = time.time()
        if context.context:
            context.context.costs.tool_calls += 1
        self._emit("tool_start", agent=agent.name, tool=tool.name, detail=tool.name)

    async def on_tool_end(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent[ResearchContext],
        tool: Tool,
        result: object,
    ) -> None:
        key = f"{agent.name}:{tool.name}"
        started = self._tool_started_at.pop(key, None)
        preview = str(result)
        self._emit(
            "tool_end",
            agent=agent.name,
            tool=tool.name,
            duration_s=round(time.time() - started, 3) if started else None,
            result_chars=len(preview),
            result_preview=preview[:280],
            detail=tool.name,
        )

    # -- model ------------------------------------------------------------

    async def on_llm_end(
        self,
        context: RunContextWrapper[ResearchContext],
        agent: Agent[ResearchContext],
        response: ModelResponse,
    ) -> None:
        usage = getattr(response, "usage", None)
        if usage and context.context:
            context.context.costs.record(
                str(agent.model),
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            )
        self._emit(
            "llm_end",
            agent=agent.name,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

    # -- reporting --------------------------------------------------------

    def handoff_chain(self) -> list[str]:
        """The ordered list of handoffs, for the architecture write-up."""
        return [
            f"{e['from_agent']} -> {e['to_agent']}"
            for e in self.events
            if e["kind"] == "handoff"
        ]

    def tool_usage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            if event["kind"] == "tool_start":
                counts[event["tool"]] = counts.get(event["tool"], 0) + 1
        return counts
