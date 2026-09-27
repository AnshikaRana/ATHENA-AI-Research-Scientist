"""Input and output guardrails.

A guardrail is an ``async`` function that inspects either the run's input (before
the first model turn) or an agent's final output (after it), and returns a
:class:`GuardrailFunctionOutput`. When ``tripwire_triggered`` is true the runner
raises the matching exception and the stage is aborted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

GuardrailFn = Callable[..., Awaitable["GuardrailFunctionOutput"]]


@dataclass
class GuardrailFunctionOutput:
    """The verdict a guardrail returns."""

    output_info: Any
    tripwire_triggered: bool


@dataclass
class GuardrailResult:
    """What :meth:`_Guardrail.run` hands back.

    ``.output`` is the raw :class:`GuardrailFunctionOutput`; the two fields are
    also proxied on this object so the runner can check ``.tripwire_triggered``
    directly without unwrapping.
    """

    output: GuardrailFunctionOutput

    @property
    def tripwire_triggered(self) -> bool:
        return self.output.tripwire_triggered

    @property
    def output_info(self) -> Any:
        return self.output.output_info


class _Guardrail:
    """Wraps a guardrail function and gives it a stable name for the roster."""

    def __init__(self, func: GuardrailFn) -> None:
        self._func = func
        self.name = getattr(func, "__name__", "guardrail")

    def get_name(self) -> str:
        return self.name

    async def run(self, *args: Any, **kwargs: Any) -> GuardrailResult:
        return GuardrailResult(output=await self._func(*args, **kwargs))


class InputGuardrail(_Guardrail):
    """Runs against the raw run input before any model call."""


class OutputGuardrail(_Guardrail):
    """Runs against an agent's parsed final output."""


def input_guardrail(func: GuardrailFn) -> InputGuardrail:
    return InputGuardrail(func)


def output_guardrail(func: GuardrailFn) -> OutputGuardrail:
    return OutputGuardrail(func)
