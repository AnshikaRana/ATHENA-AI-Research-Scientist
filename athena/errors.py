"""Typed exceptions so failures are attributable to a subsystem, not a stack trace."""

from __future__ import annotations


class AthenaError(Exception):
    """Base class for every error this system raises deliberately."""


class ConfigurationError(AthenaError):
    """Missing API key, unwritable data directory, etc."""


class ToolExecutionError(AthenaError):
    """A tool failed in a way the agent should hear about and route around."""

    def __init__(self, tool: str, message: str, *, retryable: bool = False) -> None:
        self.tool = tool
        self.retryable = retryable
        super().__init__(f"[{tool}] {message}")


class ExternalAPIError(ToolExecutionError):
    """An upstream HTTP API failed after exhausting retries."""


class DocumentParseError(ToolExecutionError):
    """A PDF / image / audio input could not be decoded."""


class GuardrailBlocked(AthenaError):
    """An input or output guardrail refused to let the run continue."""


class ApprovalRejected(AthenaError):
    """A human explicitly rejected a tool call that required approval."""


class ApprovalDeferred(AthenaError):
    """A human declined to decide *now*.

    The run is checkpointed to disk and abandoned. It can be resumed later -
    in another process, or after the user reopens the UI - via
    ``ResearchPipeline.resume_from_checkpoint``.
    """

    def __init__(self, tool_name: str, checkpoint: str) -> None:
        self.tool_name = tool_name
        self.checkpoint = checkpoint
        super().__init__(
            f"approval for '{tool_name}' deferred; state saved to {checkpoint}"
        )
