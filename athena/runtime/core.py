"""Core primitives shared across the runtime.

Athena was originally written against the OpenAI Agents SDK. This package is a
small, self-contained replacement that provides only the subset of that SDK the
project actually used, implemented directly on the official ``openai`` Python
SDK (Chat Completions with function tools) - no agent framework in between.

``core`` holds the leaf types every other runtime module depends on: the usage
ledger, the model-response envelope, the run-context wrapper that tools and
guardrails receive, and the exception hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

TContext = TypeVar("TContext")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentsException(Exception):
    """Base class for every error the runtime raises deliberately."""


class ModelBehaviorError(AgentsException):
    """The model returned something the runtime could not use.

    Raised when a structured-output agent produces text that does not parse
    into its declared Pydantic type, or when a tool call names a tool that
    does not exist.
    """


class MaxTurnsExceeded(AgentsException):
    """An agent ran for more model turns than ``max_turns`` allowed."""


class _TripwireTriggered(AgentsException):
    """Shared implementation for the two guardrail tripwire exceptions."""

    def __init__(self, guardrail_result: Any) -> None:
        self.guardrail_result = guardrail_result
        info = getattr(guardrail_result, "output_info", guardrail_result)
        super().__init__(f"Guardrail tripwire triggered: {info}")


class InputGuardrailTripwireTriggered(_TripwireTriggered):
    """An input guardrail refused to let the run start."""


class OutputGuardrailTripwireTriggered(_TripwireTriggered):
    """An output guardrail rejected an agent's final output."""


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Token and request counters for a single model call (or a sum of them)."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # True when the model hit its output budget and the reply was cut short.
    truncated: bool = False

    def add(self, other: "Usage") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens


@dataclass
class ModelResponse:
    """What a model backend returns for one turn.

    ``output`` is a list of transcript *items* (plain dicts). Each item is one
    of:

    * ``{"type": "message", "role": "assistant", "content": <str>}``
    * ``{"type": "function_call", "call_id": <str>, "name": <str>,
       "arguments": <json str>}``
    """

    output: list[dict[str, Any]]
    usage: Usage = field(default_factory=Usage)
    response_id: str = ""


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


class RunContextWrapper(Generic[TContext]):
    """Wraps the user-supplied context object handed to tools and guardrails.

    The wrapped object (``.context``) is never serialised into the model
    transcript; it is Athena's working memory, mutated locally by tools.
    """

    __slots__ = ("context", "usage")

    def __init__(self, context: TContext | None = None, usage: Usage | None = None) -> None:
        self.context = context
        self.usage = usage or Usage()

    def __class_getitem__(cls, _item: Any) -> type:  # allow RunContextWrapper[X]
        return cls


class ToolContext(RunContextWrapper[TContext]):
    """A :class:`RunContextWrapper` plus metadata about the in-flight tool call."""

    __slots__ = ("tool_name", "tool_call_id", "tool_arguments")

    def __init__(
        self,
        context: TContext | None = None,
        usage: Usage | None = None,
        *,
        tool_name: str = "",
        tool_call_id: str = "",
        tool_arguments: str = "",
    ) -> None:
        super().__init__(context, usage)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.tool_arguments = tool_arguments


# A structural alias: the hook API refers to a "Tool" type. The runtime only
# has function tools, so the two names are interchangeable.
class Tool:  # noqa: D401 - marker/alias type
    """Marker base type for tools (see :mod:`athena.runtime.tool`)."""
