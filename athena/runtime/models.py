"""Model backends.

``Model`` is the interface the runner calls once per turn. ``OpenAIModel`` is
the live implementation over the official ``openai`` SDK (Chat Completions with
function tools); the offline mock in ``athena.mock.model`` implements the same
interface with fixtures.
"""

from __future__ import annotations

from typing import Any, Protocol

from .agent import AgentOutputSchema, ModelSettings
from .core import ModelResponse, Usage

DEFAULT_MAX_TOKENS = 16384


class Model(Protocol):
    """One turn of generation."""

    async def get_response(
        self,
        *,
        system_instructions: str | None,
        input: list[dict[str, Any]],
        model_settings: ModelSettings,
        model_name: str,
        tools: list[Any],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Any],
        force_output: bool = False,
    ) -> ModelResponse:
        ...


class ModelProvider(Protocol):
    def get_model(self, model_name: str | None) -> Model:
        ...


# ---------------------------------------------------------------------------
# Transcript <-> Chat Completions message conversion
# ---------------------------------------------------------------------------


def items_to_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert runtime transcript items into an OpenAI ``messages`` array.

    Two shape rules the API enforces and this has to honour:

    * tool calls the model issued in one turn belong in **one** assistant
      message carrying every ``tool_call`` - emitting one message per call is
      rejected with "must be followed by tool messages responding to each
      tool_call_id";
    * every ``tool_call_id`` must be answered. A run suspended at an approval
      gate has a call with no recorded result, so a placeholder is synthesised
      rather than letting the request 400.
    """
    messages: list[dict[str, Any]] = []
    index = 0
    total = len(items)

    while index < total:
        item = items[index]
        kind = item.get("type")

        if kind == "message":
            role = item.get("role", "user")
            content = item.get("content", "")
            previous = messages[-1] if messages else None
            if (
                previous is not None
                and previous.get("role") == role
                and isinstance(previous.get("content"), str)
                and "tool_calls" not in previous
            ):
                previous["content"] = f"{previous['content']}\n{content}".strip()
            else:
                messages.append({"role": role, "content": content})
            index += 1
            continue

        if kind == "function_call":
            batch: list[dict[str, Any]] = []
            while index < total and items[index].get("type") == "function_call":
                call = items[index]
                batch.append(
                    {
                        "id": call["call_id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call.get("arguments") or "{}",
                        },
                    }
                )
                index += 1
            messages.append({"role": "assistant", "content": None, "tool_calls": batch})

            answered: set[str] = set()
            while index < total and items[index].get("type") == "function_call_output":
                output = items[index]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": output["call_id"],
                        "content": str(output.get("output", "")),
                    }
                )
                answered.add(output["call_id"])
                index += 1
            for call in batch:
                if call["id"] not in answered:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "(no result recorded)",
                        }
                    )
            continue

        # An output with no preceding call in this transcript has nothing to
        # attach to; carrying it would be rejected.
        index += 1

    if messages and messages[0]["role"] not in ("user", "system"):
        messages.insert(0, {"role": "user", "content": "Begin."})
    return messages


def _tool_schemas(tools: list[Any], handoffs: list[Any]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", tool.name),
                    "parameters": getattr(
                        tool,
                        "params_json_schema",
                        {"type": "object", "properties": {}},
                    ),
                },
            }
        )
    for hop in handoffs:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": hop.tool_name,
                    "description": f"Transfer the conversation to the {hop.agent_name}.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )
    return schemas


# ---------------------------------------------------------------------------
# Live backend
# ---------------------------------------------------------------------------


class OpenAIModel:
    """Calls the OpenAI Chat Completions API for one turn."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            from ..config import get_settings

            self._client = AsyncOpenAI(api_key=get_settings().openai_api_key)
        return self._client

    async def get_response(
        self,
        *,
        system_instructions: str | None,
        input: list[dict[str, Any]],
        model_settings: ModelSettings,
        model_name: str,
        tools: list[Any],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Any],
        force_output: bool = False,
    ) -> ModelResponse:
        """Run one turn.

        ``force_output=True`` is the repair path: withhold the tools and pin the
        reply to the output schema with OpenAI structured outputs, so the model
        cannot answer with a tool-status object instead of the artefact.
        """
        client = self._get_client()

        messages = items_to_messages(input)
        if system_instructions:
            messages.insert(0, {"role": "system", "content": system_instructions})

        tool_schemas = _tool_schemas(tools, handoffs)

        kwargs: dict[str, Any] = {
            "model": model_name or self.model_name,
            "messages": messages,
        }
        if tool_schemas and not force_output:
            kwargs["tools"] = tool_schemas
        if force_output and output_schema is not None and not output_schema.is_plain_text():
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.name(),
                    "schema": output_schema.json_schema(),
                    "strict": True,
                },
            }
        # Always send a budget. Leaving it unset lets a long structured reply run
        # to the model ceiling and come back truncated mid-string, which then
        # fails JSON parsing with no obvious cause.
        kwargs["max_tokens"] = model_settings.max_tokens or DEFAULT_MAX_TOKENS
        if model_settings.temperature is not None:
            kwargs["temperature"] = model_settings.temperature
        if model_settings.top_p is not None:
            kwargs["top_p"] = model_settings.top_p

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception:
            # A schema the strict decoder will not accept must not lose the run;
            # fall back to plain JSON mode, which the prompt already asks for.
            if "response_format" not in kwargs:
                raise
            kwargs["response_format"] = {"type": "json_object"}
            response = await client.chat.completions.create(**kwargs)

        usage = Usage(
            requests=1,
            input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )
        usage.total_tokens = usage.input_tokens + usage.output_tokens

        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            # Surfaced so the caller can ask for a shorter answer instead of
            # puzzling over invalid JSON.
            usage.truncated = True
        message = choice.message
        if getattr(message, "tool_calls", None):
            return ModelResponse(
                output=[
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    }
                    for call in message.tool_calls
                ],
                usage=usage,
                response_id=getattr(response, "id", ""),
            )

        return ModelResponse(
            output=[
                {"type": "message", "role": "assistant", "content": message.content or ""}
            ],
            usage=usage,
            response_id=getattr(response, "id", ""),
        )


class OpenAIModelProvider:
    """Default provider: one :class:`OpenAIModel` per model name."""

    def __init__(self) -> None:
        self._cache: dict[str, OpenAIModel] = {}

    def get_model(self, model_name: str | None) -> Model:
        key = model_name or "gpt-4o"
        if key not in self._cache:
            self._cache[key] = OpenAIModel(key)
        return self._cache[key]
