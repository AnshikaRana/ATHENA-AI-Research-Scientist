"""Quantitative tools for the Experiment Planner.

Two capabilities, both aimed at the same failure mode: language models are
fluent but unreliable at arithmetic, and an experiment design whose sample size
was *guessed* is worthless. So the numbers in a design come from actual
computation.

``statistical_power_analysis`` is a closed-form calculation. ``run_python_analysis``
is a restricted sandbox for anything else the designer needs to compute.
"""

from __future__ import annotations

import ast
import asyncio
import math
import sys
import textwrap
from statistics import NormalDist

from athena.runtime import RunContextWrapper, function_tool

from ..context import ResearchContext
from ..errors import ToolExecutionError
from ..logging_setup import get_logger, log_event

logger = get_logger("tools.analysis")

# ---------------------------------------------------------------------------
# Sandbox policy
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = {
    "math", "statistics", "random", "json", "itertools", "collections",
    "decimal", "fractions", "datetime", "re", "numpy",
}

BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}

EXECUTION_TIMEOUT_SECONDS = 15


class UnsafeCodeError(ToolExecutionError):
    """The submitted code violated the sandbox policy before it ever ran."""


def validate_code(code: str) -> None:
    """Static AST check performed *before* execution.

    This is defence in depth, not the only defence - the code also runs in a
    separate short-lived process. The point of the AST pass is to reject the
    obvious escapes with a clear error the agent can act on, rather than letting
    them fail opaquely at runtime.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCodeError("run_python_analysis", f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        # Imports must be on the allowlist.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise UnsafeCodeError(
                        "run_python_analysis",
                        f"import of '{alias.name}' is not permitted. "
                        f"Allowed: {sorted(ALLOWED_IMPORTS)}",
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(
                    "run_python_analysis",
                    f"import from '{node.module}' is not permitted. "
                    f"Allowed: {sorted(ALLOWED_IMPORTS)}",
                )
        # Dangerous builtins.
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise UnsafeCodeError(
                "run_python_analysis", f"use of '{node.id}' is not permitted"
            )
        # Dunder attribute access is the classic sandbox escape
        # (e.g. ().__class__.__bases__[0].__subclasses__()).
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(
                "run_python_analysis",
                f"access to dunder attribute '{node.attr}' is not permitted",
            )


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------


def required_sample_size(
    effect_size: float, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True
) -> int:
    """Per-group n for a two-sample comparison of means.

    Uses the standard normal approximation
    ``n = 2 * (z_(1-alpha/2) + z_power)^2 / d^2``, then applies a small-sample
    correction of +2 which brings it in line with the exact t-based result over
    the range of effect sizes that matter in practice.
    """
    if effect_size <= 0:
        raise ToolExecutionError(
            "statistical_power_analysis", "effect_size must be greater than 0"
        )
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ToolExecutionError(
            "statistical_power_analysis", "alpha and power must both be in (0, 1)"
        )

    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2) if two_sided else normal.inv_cdf(1 - alpha)
    z_power = normal.inv_cdf(power)
    n = 2 * ((z_alpha + z_power) ** 2) / (effect_size**2)
    return int(math.ceil(n)) + 2


@function_tool
async def statistical_power_analysis(
    wrapper: RunContextWrapper[ResearchContext],
    effect_size: float,
    alpha: float,
    power: float,
) -> str:
    """Compute the sample size an experiment needs to detect a given effect.

    Always call this before stating a sample size. Never estimate one by eye.

    Args:
        effect_size: Expected Cohen's d. Conventionally 0.2 small, 0.5 medium, 0.8 large.
        alpha: Significance level, normally 0.05.
        power: Desired power, normally 0.8 (or 0.9 for a confirmatory study).
    """
    n = required_sample_size(effect_size, alpha, power)
    log_event(
        logger,
        "power analysis",
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        n_per_group=n,
    )
    interpretation = (
        "small" if effect_size < 0.35 else "medium" if effect_size < 0.65 else "large"
    )
    return (
        f"Power analysis (two-sample, two-sided):\n"
        f"  effect size d = {effect_size} ({interpretation})\n"
        f"  alpha = {alpha}, target power = {power}\n"
        f"  REQUIRED n = {n} per group ({n * 2} total)\n"
        f"State this number and these assumptions explicitly in the design."
    )


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

_PREAMBLE = """
import sys
sys.setrecursionlimit(2000)
"""


@function_tool(
    failure_error_function=lambda ctx, err: f"Analysis code failed: {err}"
)
async def run_python_analysis(
    wrapper: RunContextWrapper[ResearchContext], code: str, purpose: str
) -> str:
    """Execute a short Python snippet to compute something exactly.

    Use for simulations, effect-size conversions, sanity checks on estimates -
    anything where a wrong number would invalidate the design. Print results;
    stdout is what comes back.

    Restrictions: no file, network or OS access. Imports limited to math,
    statistics, random, json, itertools, collections, decimal, fractions,
    datetime, re and numpy. 15-second limit.

    Args:
        code: Python source. Use print() to emit results.
        purpose: One line on what this computes, recorded in the audit log.
    """
    ctx = wrapper.context
    validate_code(code)  # raises UnsafeCodeError with an actionable message

    script = _PREAMBLE + "\n" + textwrap.dedent(code)
    log_event(logger, "sandbox exec", purpose=purpose, code_chars=len(code))

    # A separate process gives us a hard timeout and prevents a runaway snippet
    # from taking the agent run down with it.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",  # isolated mode: ignore env vars and user site-packages
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=EXECUTION_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ToolExecutionError(
            "run_python_analysis",
            f"execution exceeded {EXECUTION_TIMEOUT_SECONDS}s and was terminated",
        ) from None

    out = stdout.decode("utf-8", errors="replace").strip()
    err = stderr.decode("utf-8", errors="replace").strip()
    ctx.costs.tool_calls += 0  # counted by hooks; kept explicit for clarity

    if process.returncode != 0:
        # Return the error to the agent rather than raising: a traceback is
        # information it can use to fix its own code on the next turn.
        return f"Code exited with status {process.returncode}.\nstderr:\n{err[:2000]}"

    if not out:
        return "Code ran successfully but printed nothing. Use print() to emit results."
    return f"Output ({purpose}):\n{out[:4000]}"
