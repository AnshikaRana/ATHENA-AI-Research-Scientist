"""Lifecycle hooks.

``RunHooks`` is a set of ``async`` callbacks the runner invokes as a run
progresses. All methods default to no-ops so a subclass overrides only what it
cares about. ``AgentHookContext`` is just the run-context wrapper under the name
the original hook signatures used.
"""

from __future__ import annotations

from typing import Any, Generic

from .core import RunContextWrapper, TContext

AgentHookContext = RunContextWrapper


class RunHooks(Generic[TContext]):
    """Override any subset of these; the runner calls them positionally."""

    async def on_agent_start(self, context: Any, agent: Any) -> None:  # noqa: D401
        ...

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        ...

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        ...

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        ...

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        ...

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        ...
