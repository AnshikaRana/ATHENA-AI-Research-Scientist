"""The runner: drives an agent through model turns, tools, handoffs and the
human-approval interrupt.

The approval interrupt is the reason ``RunState`` exists. When a gated tool is
reached, the run cannot block waiting for a person, so the runner returns a
``RunResult`` carrying ``interruptions`` and a serialisable ``RunState``. The
caller records a decision on the state and calls :meth:`Runner.run` again with
it to resume - possibly in a different process.
"""

from __future__ import annotations

import json
from typing import Any

from .agent import Agent, _strip_code_fences
from .core import (
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    ToolContext,
)
from .hooks import RunHooks
from .models import OpenAIModelProvider

_DEFAULT_PROVIDER = OpenAIModelProvider()

# How many times to re-ask when a structured reply fails to validate.
_MAX_OUTPUT_REPAIRS = 2


def set_tracing_disabled(value: bool = True) -> None:
    """No-op. The old Agents SDK uploaded traces to the OpenAI platform; this
    runtime does not."""


class RunConfig:
    """Run-wide settings. Only ``model_provider`` changes behaviour here."""

    def __init__(
        self,
        *,
        model_provider: Any = None,
        workflow_name: str = "",
        tracing_disabled: bool = True,
        **_ignored: Any,
    ) -> None:
        self.model_provider = model_provider
        self.workflow_name = workflow_name
        self.tracing_disabled = tracing_disabled


class ToolApprovalItem:
    """A single gated tool call awaiting a human decision."""

    def __init__(self, name: str, arguments: str, call_id: str) -> None:
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class RunState:
    """A suspended run. Serialisable so a pending approval can outlive a process."""

    def __init__(
        self,
        *,
        agent: Agent[Any],
        items: list[dict[str, Any]],
        pending_calls: list[dict[str, Any]],
        context: RunContextWrapper[Any] | None = None,
    ) -> None:
        self.agent = agent
        self.items = items
        self.pending_calls = pending_calls
        self.context = context
        self.decisions: dict[str, tuple[bool, str | None]] = {}

    # -- decisions ----------------------------------------------------------

    def get_interruptions(self) -> list[ToolApprovalItem]:
        return [
            ToolApprovalItem(call["name"], call["arguments"], call["call_id"])
            for call in self.pending_calls
        ]

    def approve(self, item: ToolApprovalItem) -> None:
        self.decisions[item.call_id] = (True, None)

    def reject(self, item: ToolApprovalItem, rejection_message: str = "") -> None:
        self.decisions[item.call_id] = (False, rejection_message or "Rejected by human.")

    # -- persistence ------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "agent": getattr(self.agent, "name", ""),
            "items": self.items,
            "pending_calls": self.pending_calls,
            "decisions": {
                call_id: {"approved": approved, "message": message}
                for call_id, (approved, message) in self.decisions.items()
            },
        }

    @classmethod
    async def from_json(
        cls,
        agent: Agent[Any],
        payload: dict[str, Any],
        *,
        context_override: RunContextWrapper[Any] | None = None,
    ) -> "RunState":
        state = cls(
            agent=agent,
            items=list(payload.get("items", [])),
            pending_calls=list(payload.get("pending_calls", [])),
            context=context_override,
        )
        for call_id, decision in (payload.get("decisions") or {}).items():
            state.decisions[call_id] = (
                bool(decision.get("approved")),
                decision.get("message"),
            )
        return state


class RunResult:
    """The outcome of :meth:`Runner.run` - either a final output or a suspension."""

    def __init__(
        self,
        *,
        final_output: Any = None,
        last_agent: Agent[Any] | None = None,
        new_items: list[dict[str, Any]] | None = None,
        interruptions: list[ToolApprovalItem] | None = None,
        state: RunState | None = None,
    ) -> None:
        self.final_output = final_output
        self.last_agent = last_agent
        self.new_items = new_items or []
        self._interruptions = interruptions
        self._state = state

    @property
    def interruptions(self) -> list[ToolApprovalItem] | None:
        return self._interruptions

    def to_state(self) -> RunState | None:
        return self._state

    def final_output_as(self, cls: type) -> Any:
        output = self.final_output
        if isinstance(output, cls):
            return output
        if isinstance(output, str) and hasattr(cls, "model_validate_json"):
            return cls.model_validate_json(_strip_code_fences(output))
        return output


# ---------------------------------------------------------------------------


def _first_text(output: list[dict[str, Any]]) -> str:
    for item in output:
        if item.get("type") == "message":
            return item.get("content", "") or ""
    return ""


def _parse_final(agent: Agent[Any], text: str) -> Any:
    schema = agent.output_schema()
    if schema.is_plain_text():
        return text
    try:
        return schema.validate_json(text)
    except Exception as exc:  # noqa: BLE001
        raise ModelBehaviorError(
            f"{agent.name} did not return valid {schema.name()}: {exc}"
        ) from exc


class _Engine:
    def __init__(self, *, hooks: RunHooks[Any], provider: Any, max_turns: int) -> None:
        self.hooks = hooks
        self.provider = provider
        self.max_turns = max_turns

    async def start(
        self,
        agent: Agent[Any],
        prompt: str,
        ctx: RunContextWrapper[Any],
        session: Any,
    ) -> RunResult:
        history = await session.get_items() if session is not None else []
        items = list(history) + [
            {"type": "message", "role": "user", "content": prompt}
        ]
        for guardrail in agent.input_guardrails:
            verdict = await guardrail.run(ctx, agent, prompt)
            if verdict.tripwire_triggered:
                raise InputGuardrailTripwireTriggered(verdict)
        return await self._loop(
            agent, items, ctx, len(history), {}, None, session
        )

    async def resume(
        self, state: RunState, ctx: RunContextWrapper[Any]
    ) -> RunResult:
        return await self._loop(
            state.agent,
            list(state.items),
            ctx,
            len(state.items),
            dict(state.decisions),
            list(state.pending_calls),
            None,
        )

    async def _loop(
        self,
        agent: Agent[Any],
        items: list[dict[str, Any]],
        ctx: RunContextWrapper[Any],
        base_len: int,
        decisions: dict[str, tuple[bool, str | None]],
        pending_from_state: list[dict[str, Any]] | None,
        session: Any,
    ) -> RunResult:
        active = agent
        await self.hooks.on_agent_start(ctx, active)

        # Tool calls already completed in this run, so a model that keeps
        # re-issuing the same call can be pushed to its final answer instead.
        executed: set[str] = set()
        force_output = False

        if pending_from_state:
            active, suspended, repeated = await self._handle_calls(
                active,
                pending_from_state,
                items,
                ctx,
                decisions,
                already_recorded=True,
                executed=executed,
            )
            if suspended is not None:
                return suspended
            force_output = force_output or repeated

        turns = 0
        while True:
            turns += 1
            if turns > self.max_turns:
                raise MaxTurnsExceeded(
                    f"{active.name} exceeded the {self.max_turns}-turn budget"
                )

            model = self.provider.get_model(active.model)
            response = await model.get_response(
                system_instructions=active.system_prompt(),
                input=items,
                model_settings=active.model_settings,
                model_name=active.model,
                tools=list(active.tools),
                output_schema=active.output_schema(),
                handoffs=list(active.handoffs),
                force_output=force_output,
            )
            force_output = False
            await self.hooks.on_llm_end(ctx, active, response)

            calls = [o for o in response.output if o.get("type") == "function_call"]
            if not calls:
                text = _first_text(response.output)
                output, text = await self._finalize(active, text, items, ctx, model)
                for guardrail in active.output_guardrails:
                    verdict = await guardrail.run(ctx, active, output)
                    if verdict.tripwire_triggered:
                        raise OutputGuardrailTripwireTriggered(verdict)
                items.append(
                    {"type": "message", "role": "assistant", "content": text}
                )
                await self.hooks.on_agent_end(ctx, active, output)
                if session is not None:
                    await session.add_items(items[base_len:])
                return RunResult(
                    final_output=output,
                    last_agent=active,
                    new_items=items[base_len:],
                )

            active, suspended, repeated = await self._handle_calls(
                active,
                calls,
                items,
                ctx,
                decisions,
                already_recorded=False,
                executed=executed,
            )
            if suspended is not None:
                return suspended
            force_output = force_output or repeated

    async def _finalize(
        self,
        active: Agent[Any],
        text: str,
        items: list[dict[str, Any]],
        ctx: RunContextWrapper[Any],
        model: Any,
    ) -> tuple[Any, str]:
        """Parse the final message, repairing it if the model wandered off-schema.

        A model that has just run a tool sometimes answers with the tool's own
        status payload instead of the artefact it was asked for. Rather than
        losing the whole stage to that, re-ask with the tools withheld and the
        reply pinned to the output schema.
        """
        schema = active.output_schema()
        if schema.is_plain_text():
            return text, text

        attempt_text = text
        last_error: Exception | None = None
        for attempt in range(_MAX_OUTPUT_REPAIRS + 1):
            try:
                return schema.validate_json(attempt_text), attempt_text
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
            if attempt == _MAX_OUTPUT_REPAIRS:
                break

            repair_items = items + [
                {"type": "message", "role": "assistant", "content": attempt_text},
                {
                    "type": "message",
                    "role": "user",
                    "content": (
                        f"That reply is not a valid {schema.name()} object "
                        f"({last_error}). Do not call any tool and do not report "
                        f"tool status. Using the work already completed in this "
                        f"conversation, reply with ONLY the JSON object for "
                        f"{schema.name()}, matching its schema exactly. Keep every "
                        f"field short - the previous attempt may have been cut off "
                        f"for length, so favour brevity over completeness."
                    ),
                },
            ]
            response = await model.get_response(
                system_instructions=active.system_prompt(),
                input=repair_items,
                model_settings=active.model_settings,
                model_name=active.model,
                tools=[],
                output_schema=schema,
                handoffs=[],
                force_output=True,
            )
            await self.hooks.on_llm_end(ctx, active, response)
            attempt_text = _first_text(response.output)

        raise ModelBehaviorError(
            f"{active.name} did not return valid {schema.name()} after "
            f"{_MAX_OUTPUT_REPAIRS} repair attempts: {last_error}"
        )

    async def _handle_calls(
        self,
        active: Agent[Any],
        calls: list[dict[str, Any]],
        items: list[dict[str, Any]],
        ctx: RunContextWrapper[Any],
        decisions: dict[str, tuple[bool, str | None]],
        *,
        already_recorded: bool,
        executed: set[str] | None = None,
    ) -> tuple[Agent[Any], RunResult | None, bool]:
        executed = executed if executed is not None else set()
        repeated = False
        pending: list[dict[str, Any]] = []
        for call in calls:
            if not already_recorded:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["call_id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                )

            hop = next(
                (h for h in active.handoffs if h.tool_name == call["name"]), None
            )
            if hop is not None:
                if hop.on_handoff is not None:
                    await hop.on_handoff(ctx)
                await self.hooks.on_handoff(ctx, active, hop.agent)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": f"Transferred to {hop.agent_name}.",
                    }
                )
                if hop.input_filter is not None:
                    items[:] = hop.input_filter(list(items))
                active = hop.agent
                await self.hooks.on_agent_start(ctx, active)
                continue

            signature = f"{call['name']}|{call['arguments']}"
            if signature in executed:
                # Observed live: after its artefact was saved the Scientific
                # Writer kept re-calling the same tool instead of returning the
                # report, and the run ended with no output. Re-running the call
                # cannot help, so refuse it and push for the final answer.
                repeated = True
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": (
                            f"{call['name']} already completed earlier in this run "
                            f"with identical arguments; it was not run again. Do not "
                            f"call any further tools - reply now with your final answer."
                        ),
                    }
                )
                continue

            tool = next((t for t in active.tools if t.name == call["name"]), None)
            if tool is None:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": f"Error: no tool named {call['name']}.",
                    }
                )
                continue

            try:
                params = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                params = {}
            if not isinstance(params, dict):
                params = {}

            decision = decisions.get(call["call_id"])
            needs_approval = await tool.approval_required(ctx, params, call["call_id"])

            if needs_approval and decision is None:
                pending.append(call)
                continue
            if needs_approval and decision is not None and decision[0] is False:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": decision[1] or "Rejected by human.",
                    }
                )
                continue

            await self.hooks.on_tool_start(ctx, active, tool)
            tool_ctx = ToolContext(
                context=ctx.context,
                tool_name=tool.name,
                tool_call_id=call["call_id"],
                tool_arguments=call["arguments"],
            )
            result = await tool.on_invoke_tool(tool_ctx, call["arguments"])
            await self.hooks.on_tool_end(ctx, active, tool, result)
            executed.add(signature)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": str(result),
                }
            )

        if pending:
            state = RunState(
                agent=active,
                items=list(items),
                pending_calls=pending,
                context=ctx,
            )
            state.decisions.update(decisions)
            return active, RunResult(
                interruptions=[
                    ToolApprovalItem(c["name"], c["arguments"], c["call_id"])
                    for c in pending
                ],
                state=state,
                last_agent=active,
            ), repeated
        return active, None, repeated


class Runner:
    """Entry point. ``run`` starts a fresh run or resumes a :class:`RunState`."""

    @classmethod
    async def run(
        cls,
        agent: Agent[Any],
        input: Any,
        *,
        context: Any = None,
        hooks: RunHooks[Any] | None = None,
        run_config: RunConfig | None = None,
        session: Any = None,
        max_turns: int = 10,
    ) -> RunResult:
        provider = (
            run_config.model_provider
            if run_config is not None and run_config.model_provider is not None
            else _DEFAULT_PROVIDER
        )
        engine = _Engine(hooks=hooks or RunHooks(), provider=provider, max_turns=max_turns)

        if isinstance(input, RunState):
            ctx = (
                context
                if isinstance(context, RunContextWrapper)
                else RunContextWrapper(context)
                if context is not None
                else input.context or RunContextWrapper(None)
            )
            return await engine.resume(input, ctx)

        ctx = (
            context
            if isinstance(context, RunContextWrapper)
            else RunContextWrapper(context)
        )
        return await engine.start(agent, str(input), ctx, session)
