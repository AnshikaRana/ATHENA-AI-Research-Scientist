"""Handoffs - one agent transferring control to another.

A handoff is surfaced to the model as an ordinary tool named
``transfer_to_<agent>``. When the model calls it, the runner swaps the active
agent, optionally filters the transcript the new agent sees, and fires the
``on_handoff`` callback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

# Prepended to a routing agent's instructions so it understands the mechanism.
RECOMMENDED_PROMPT_PREFIX = (
    "# System context\n"
    "You are part of a multi-agent system designed to make agent coordination "
    "and execution easy. Agents use two primary abstractions: **Agents** and "
    "**Handoffs**. An agent encompasses instructions and tools and can hand off "
    "a conversation to another agent when appropriate. Handoffs are achieved by "
    "calling a handoff function, generally named `transfer_to_<agent_name>`. "
    "Transfers between agents are handled seamlessly in the background; do not "
    "mention or draw attention to these transfers in your conversation with the "
    "user.\n"
)


def _snake(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return slug or "agent"


@dataclass
class Handoff:
    """A wired transfer target."""

    agent: Any  # athena.runtime.agent.Agent - avoid a circular import
    on_handoff: Callable[[Any], Awaitable[None]] | None = None
    input_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None
    tool_name: str = ""
    agent_name: str = ""

    def __post_init__(self) -> None:
        self.agent_name = self.agent_name or getattr(self.agent, "name", "agent")
        self.tool_name = self.tool_name or f"transfer_to_{_snake(self.agent_name)}"

    @property
    def name(self) -> str:  # roster compatibility
        return self.agent_name


def handoff(
    agent: Any,
    *,
    on_handoff: Callable[[Any], Awaitable[None]] | None = None,
    input_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> Handoff:
    return Handoff(agent=agent, on_handoff=on_handoff, input_filter=input_filter)


class handoff_filters:  # noqa: N801 - kept lowercase to match the original import
    """Namespace of ready-made transcript filters for ``input_filter``."""

    @staticmethod
    def remove_all_tools(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop tool calls and their outputs, keeping only plain messages."""
        return [item for item in items if item.get("type") == "message"]
