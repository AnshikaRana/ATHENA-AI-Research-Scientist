"""Test helpers for invoking runtime function tools without a full Runner."""

from __future__ import annotations

import json
from typing import Any

from athena.runtime import ToolContext, Usage

from athena.context import ResearchContext


def make_context(**overrides: Any) -> ResearchContext:
    defaults = {
        "run_id": "test_run",
        "session_id": "test_session",
        "research_question": "test question",
    }
    defaults.update(overrides)
    return ResearchContext(**defaults)  # type: ignore[arg-type]


async def call_tool(tool: Any, ctx: ResearchContext, **kwargs: Any) -> str:
    """Invoke a @function_tool directly, the way the Runner would."""
    tool_ctx = ToolContext(
        context=ctx,
        usage=Usage(),
        tool_name=tool.name,
        tool_call_id="test_call",
        tool_arguments=json.dumps(kwargs),
    )
    return await tool.on_invoke_tool(tool_ctx, json.dumps(kwargs))
