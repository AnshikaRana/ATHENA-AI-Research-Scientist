"""Athena's agent runtime.

A small, self-contained reimplementation of the subset of the OpenAI Agents SDK
that Athena used, built directly on the official ``openai`` Python SDK (Chat
Completions with function tools). Downstream modules import from here instead of
``agents``.
"""

from __future__ import annotations

from .agent import Agent, AgentOutputSchema, ModelSettings
from .core import (
    AgentsException,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelResponse,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Tool,
    ToolContext,
    Usage,
)
from .guardrail import (
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    input_guardrail,
    output_guardrail,
)
from .handoff import RECOMMENDED_PROMPT_PREFIX, Handoff, handoff, handoff_filters
from .hooks import AgentHookContext, RunHooks
from .models import Model, ModelProvider, OpenAIModel, OpenAIModelProvider
from .runner import (
    RunConfig,
    Runner,
    RunResult,
    RunState,
    ToolApprovalItem,
    set_tracing_disabled,
)
from .session import SQLiteSession
from .tool import FunctionTool, function_tool

# The old SDK exposed a submodule ``agents.extensions.handoff_prompt`` and
# ``agents.extensions.handoff_filters``; ``handoff_filters`` above and this alias
# cover both import styles.
handoff_prompt = type(
    "handoff_prompt", (), {"RECOMMENDED_PROMPT_PREFIX": RECOMMENDED_PROMPT_PREFIX}
)

__all__ = [
    "Agent",
    "AgentHookContext",
    "AgentOutputSchema",
    "AgentsException",
    "FunctionTool",
    "function_tool",
    "GuardrailFunctionOutput",
    "Handoff",
    "InputGuardrail",
    "InputGuardrailTripwireTriggered",
    "MaxTurnsExceeded",
    "Model",
    "ModelProvider",
    "OpenAIModel",
    "OpenAIModelProvider",
    "ModelBehaviorError",
    "ModelResponse",
    "ModelSettings",
    "OutputGuardrail",
    "OutputGuardrailTripwireTriggered",
    "RECOMMENDED_PROMPT_PREFIX",
    "RunConfig",
    "RunContextWrapper",
    "RunHooks",
    "RunResult",
    "RunState",
    "Runner",
    "SQLiteSession",
    "Tool",
    "ToolApprovalItem",
    "ToolContext",
    "Usage",
    "handoff",
    "handoff_filters",
    "handoff_prompt",
    "input_guardrail",
    "output_guardrail",
    "set_tracing_disabled",
]
