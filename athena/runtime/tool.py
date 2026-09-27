"""``function_tool`` - turn an ``async def`` into a schema-described tool.

The decorator introspects the wrapped function's signature and docstring to
build a JSON Schema for its parameters, and exposes ``on_invoke_tool`` which the
runner calls with the raw JSON arguments string. The first positional parameter
is always the run-context wrapper and is never part of the schema.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Awaitable, Callable, get_args, get_origin, get_type_hints

from .core import Tool, ToolContext

# needs_approval may be a bare bool, or a callable ``(wrapper, params, call_id)``
# returning a bool (optionally async).
ApprovalPredicate = bool | Callable[..., Any]
FailureHandler = Callable[[Any, Exception], str]


def _json_type(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation onto a JSON Schema fragment."""
    if annotation in (inspect.Parameter.empty, Any, None):
        return {"type": "string"}

    origin = get_origin(annotation)
    if origin in (list, set, tuple):
        args = get_args(annotation)
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}
    if origin is dict:
        return {"type": "object"}

    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    return mapping.get(annotation, {"type": "string"})


def _arg_descriptions(docstring: str | None) -> dict[str, str]:
    """Pull ``name: description`` pairs out of a Google-style ``Args:`` block."""
    if not docstring or "Args:" not in docstring:
        return {}
    body = docstring.split("Args:", 1)[1]
    body = re.split(r"\n\s*(?:Returns|Raises|Yields):", body, maxsplit=1)[0]
    out: dict[str, str] = {}
    for match in re.finditer(r"^\s*([A-Za-z_]\w*):\s*(.+?)(?=\n\s*[A-Za-z_]\w*:|\Z)", body, re.S | re.M):
        out[match.group(1)] = " ".join(match.group(2).split())
    return out


class FunctionTool(Tool):
    """A single callable tool with a generated parameter schema."""

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        needs_approval: ApprovalPredicate = False,
        failure_error_function: FailureHandler | None = None,
    ) -> None:
        self._func = func
        self.name = func.__name__
        doc = inspect.getdoc(func) or ""
        self.description = doc.split("\n\n", 1)[0].strip() or self.name
        self.needs_approval = needs_approval
        self.failure_error_function = failure_error_function

        signature = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        descriptions = _arg_descriptions(doc)

        properties: dict[str, Any] = {}
        required: list[str] = []
        for index, (param_name, param) in enumerate(signature.parameters.items()):
            if index == 0:
                continue  # the run-context wrapper
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            fragment = _json_type(hints.get(param_name, param.annotation))
            if param_name in descriptions:
                fragment["description"] = descriptions[param_name]
            properties[param_name] = fragment
            if param.default is param.empty:
                required.append(param_name)

        self.params_json_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    # -- invocation -----------------------------------------------------------

    async def on_invoke_tool(self, tool_context: ToolContext, arguments: str) -> str:
        """Execute the tool. Mirrors the signature the old SDK's runner used."""
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}

        try:
            result = await self._func(tool_context, **parsed)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent as text
            if self.failure_error_function is not None:
                return self.failure_error_function(tool_context, exc)
            # A failing tool is information the agent can act on, not a reason to
            # lose the whole stage. Observed live: the model omitted a required
            # argument, Python raised TypeError, and the citations stage died.
            # Returning the error lets the model correct itself on the next turn.
            return f"Tool {self.name} failed: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, default=str)

    async def approval_required(
        self, wrapper: Any, params: dict[str, Any], call_id: str
    ) -> bool:
        """Resolve ``needs_approval`` for a specific call."""
        predicate = self.needs_approval
        if predicate is False or predicate is None:
            return False
        if predicate is True:
            return True
        outcome = predicate(wrapper, params, call_id)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        return bool(outcome)


def function_tool(
    func: Callable[..., Awaitable[Any]] | None = None,
    *,
    needs_approval: ApprovalPredicate = False,
    failure_error_function: FailureHandler | None = None,
) -> Any:
    """Decorator usable bare (``@function_tool``) or called (``@function_tool(...)``)."""

    def wrap(target: Callable[..., Awaitable[Any]]) -> FunctionTool:
        return FunctionTool(
            target,
            needs_approval=needs_approval,
            failure_error_function=failure_error_function,
        )

    if func is not None:
        return wrap(func)
    return wrap
