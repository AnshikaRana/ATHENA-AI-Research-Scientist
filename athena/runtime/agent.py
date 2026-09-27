"""The ``Agent`` definition and its output-schema helper."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Generic

from .core import TContext


@dataclass
class ModelSettings:
    """Per-agent sampling knobs, forwarded verbatim to the Chat Completions call."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


class AgentOutputSchema:
    """Describes an agent's structured output type.

    Wraps a Pydantic model (or ``None`` / ``str`` for a plain-text agent) and
    exposes what the runner, the offline mock and the schema tests need: a name,
    the JSON Schema, and a strictness check. "Strict" here means the same thing
    it meant under the old provider - every object closed with
    ``additionalProperties: false`` and every field required - because a schema
    that is not strict-clean fails on the first real API call.
    """

    def __init__(self, output_type: Any, *, strict_json_schema: bool = True) -> None:
        self.output_type = output_type
        self._plain = output_type is None or output_type is str
        self._strict = strict_json_schema

    def is_plain_text(self) -> bool:
        return self._plain

    def is_strict_json_schema(self) -> bool:
        return self._strict

    def name(self) -> str:
        return getattr(self.output_type, "__name__", "PlainText")

    def json_schema(self) -> dict[str, Any]:
        if self._plain:
            return {"type": "string"}
        schema = self.output_type.model_json_schema()
        return _ensure_strict(schema) if self._strict else schema

    def validate_json(self, text: str) -> Any:
        if self._plain:
            return text
        return self.output_type.model_validate_json(_strip_code_fences(text))


def _ensure_strict(node: Any) -> Any:
    """Recursively close every object in a JSON Schema and require all fields."""
    if isinstance(node, list):
        return [_ensure_strict(item) for item in node]
    if not isinstance(node, dict):
        return node

    node = {key: _ensure_strict(value) for key, value in node.items()}
    if node.get("type") == "object" or "properties" in node:
        node.setdefault("properties", {})
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
    return node


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ```json ... ``` fence if the model added one."""
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S)
    return match.group(1).strip() if match else stripped


@dataclass
class Agent(Generic[TContext]):
    """A single agent: instructions, tools, an optional typed output, a model."""

    name: str
    instructions: str = ""
    handoff_description: str = ""
    tools: list[Any] = field(default_factory=list)
    handoffs: list[Any] = field(default_factory=list)
    output_type: Any | None = None
    model: str = ""
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    input_guardrails: list[Any] = field(default_factory=list)
    output_guardrails: list[Any] = field(default_factory=list)

    def output_schema(self) -> AgentOutputSchema:
        return AgentOutputSchema(self.output_type)

    def system_prompt(self) -> str:
        """The instructions, with a JSON-output contract appended when typed."""
        schema = self.output_schema()
        if schema.is_plain_text():
            return self.instructions
        return (
            f"{self.instructions}\n\n"
            "## Output format\n"
            "Your final message MUST be a single JSON object that validates "
            "against this JSON Schema. Emit only the JSON - no prose, no "
            "Markdown fences:\n"
            f"{json.dumps(schema.json_schema())}"
        )
