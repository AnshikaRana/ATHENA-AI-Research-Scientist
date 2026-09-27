"""Dataset loading and statistical analysis for the Data Interpreter.

The interpretation is the model's job. The *numbers* are not: every statistic
here is computed by pandas/numpy and handed to the agent as fact. An agent that
eyeballs a CSV and reports a correlation is guessing, and this module exists so
it never has to.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from athena.runtime import RunContextWrapper, function_tool

from ..config import get_settings
from ..context import ResearchContext
from ..errors import DocumentParseError
from ..logging_setup import get_logger, log_event

logger = get_logger("tools.datasets")

MAX_PREVIEW_ROWS = 8
MAX_CELLS = 2_000_000  # refuse to load something that will exhaust memory


def _log_gamma(x: float) -> float:
    """Lanczos approximation to log(Gamma(x)); accurate to ~1e-10 for x > 0."""
    coefficients = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    series = 1.000000000190015
    for coefficient in coefficients:
        y += 1.0
        series += coefficient / y
    return -tmp + math.log(2.5066282746310005 * series / x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    max_iterations, epsilon, tiny = 200, 3.0e-7, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def two_sided_p_value(t_stat: float, df: float) -> float:
    """Two-sided p-value for a t statistic.

    Implemented here rather than pulled from scipy: scipy is a heavy dependency
    to add for one function, and a research tool that cannot report a p-value is
    not much of a research tool.
    """
    if df <= 0:
        return float("nan")
    return _betai(0.5 * df, 0.5, df / (df + t_stat * t_stat))


def _resolve(path_str: str) -> Path:
    settings = get_settings()
    path = Path(path_str).expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = settings.uploads_dir / path.name
        if candidate.exists():
            return candidate
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load CSV/TSV/JSON/Excel into a DataFrame, with honest failures."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        elif suffix in {".tsv", ".tab"}:
            frame = pd.read_csv(path, sep="\t")
        elif suffix == ".json":
            frame = pd.read_json(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            raise DocumentParseError(
                "analyze_dataset", f"unsupported file type '{suffix}'"
            )
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("analyze_dataset", f"could not parse {path.name}: {exc}") from exc

    if frame.size > MAX_CELLS:
        raise DocumentParseError(
            "analyze_dataset",
            f"{path.name} has {frame.size:,} cells, above the {MAX_CELLS:,} limit",
        )
    return frame


@function_tool(failure_error_function=lambda ctx, err: f"Dataset analysis failed: {err}")
async def analyze_dataset(
    wrapper: RunContextWrapper[ResearchContext], file_path: str, target_column: str
) -> str:
    """Load a dataset and compute descriptive statistics and correlations.

    Args:
        file_path: Path to a .csv, .tsv, .json or .xlsx file. Bare filenames are
            looked up in data/uploads.
        target_column: Column of primary interest for correlations. Pass an
            empty string to skip the targeted analysis.
    """
    ctx = wrapper.context
    path = _resolve(file_path)
    if not path.exists():
        raise DocumentParseError("analyze_dataset", f"no such file: {path}")

    frame = load_dataframe(path)
    ctx.attachments.append(
        {"kind": "dataset", "path": str(path), "rows": len(frame), "cols": len(frame.columns)}
    )

    numeric = frame.select_dtypes(include=[np.number])
    lines = [
        f"Dataset: {path.name}",
        f"Shape: {len(frame):,} rows x {len(frame.columns)} columns",
        f"Columns: {', '.join(map(str, frame.columns[:40]))}",
        "",
        "## Missing values",
    ]

    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("None.")
    else:
        for column, count in missing.items():
            lines.append(f"  {column}: {count:,} ({count / len(frame):.1%})")

    if not numeric.empty:
        lines.append("\n## Descriptive statistics (numeric columns)")
        lines.append(numeric.describe().round(4).to_string())

        # Correlations are only meaningful with at least two numeric columns.
        if numeric.shape[1] >= 2:
            correlations = numeric.corr(numeric_only=True)
            lines.append("\n## Correlation matrix (Pearson)")
            lines.append(correlations.round(3).to_string())

            # Surface the strongest relationships explicitly, so the agent does
            # not have to read a matrix to find them.
            pairs = []
            columns = list(correlations.columns)
            for i, first in enumerate(columns):
                for second in columns[i + 1 :]:
                    value = correlations.loc[first, second]
                    if pd.notna(value):
                        pairs.append((abs(value), first, second, value))
            pairs.sort(reverse=True)
            if pairs:
                lines.append("\n## Strongest relationships")
                for _, first, second, value in pairs[:6]:
                    strength = (
                        "strong" if abs(value) >= 0.7
                        else "moderate" if abs(value) >= 0.4
                        else "weak"
                    )
                    lines.append(
                        f"  {first} ~ {second}: r = {value:+.3f} ({strength}). "
                        "Correlation is not causation; do not report it as an effect."
                    )
    else:
        lines.append("\nNo numeric columns - only categorical analysis is possible.")

    if target_column and target_column in numeric.columns:
        target_correlations = (
            numeric.corr(numeric_only=True)[target_column].drop(target_column).sort_values(
                key=abs, ascending=False
            )
        )
        lines.append(f"\n## Correlations with '{target_column}'")
        for column, value in target_correlations.head(10).items():
            lines.append(f"  {column}: r = {value:+.3f}")
    elif target_column:
        lines.append(
            f"\nRequested target '{target_column}' is not a numeric column; skipped."
        )

    lines.append(f"\n## Preview (first {MAX_PREVIEW_ROWS} rows)")
    lines.append(frame.head(MAX_PREVIEW_ROWS).to_string())

    log_event(
        logger, "dataset analysed", path=str(path), rows=len(frame), cols=len(frame.columns)
    )
    return "\n".join(lines)


@function_tool(failure_error_function=lambda ctx, err: f"Group comparison failed: {err}")
async def compare_groups(
    wrapper: RunContextWrapper[ResearchContext],
    file_path: str,
    group_column: str,
    value_column: str,
) -> str:
    """Compare a numeric measure across groups with a Welch t-test and Cohen's d.

    Welch's test is used rather than Student's because equal variances are an
    assumption that real data rarely honours.

    Args:
        file_path: Path to the dataset.
        group_column: Categorical column defining the groups.
        value_column: Numeric column to compare.
    """
    path = _resolve(file_path)
    if not path.exists():
        raise DocumentParseError("compare_groups", f"no such file: {path}")

    frame = load_dataframe(path)
    for column in (group_column, value_column):
        if column not in frame.columns:
            return f"Column '{column}' not found. Available: {list(frame.columns)[:30]}"

    groups = [
        (str(name), subset[value_column].dropna().to_numpy(dtype=float))
        for name, subset in frame.groupby(group_column)
    ]
    groups = [(name, values) for name, values in groups if len(values) >= 2]
    if len(groups) < 2:
        return f"Need at least two groups with 2+ observations; found {len(groups)}."

    lines = [f"Comparing '{value_column}' across '{group_column}' ({len(groups)} groups)\n"]
    for name, values in groups:
        lines.append(
            f"  {name}: n={len(values)}, mean={values.mean():.4f}, sd={values.std(ddof=1):.4f}"
        )

    if len(groups) == 2:
        (name_a, a), (name_b, b) = groups
        mean_diff = a.mean() - b.mean()

        # Welch's t-test, computed directly to avoid a scipy dependency.
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(var_a / len(a) + var_b / len(b))
        t_stat = mean_diff / se if se > 0 else 0.0
        df_num = (var_a / len(a) + var_b / len(b)) ** 2
        df_den = (var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1)
        df = df_num / df_den if df_den > 0 else 1.0

        pooled_sd = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
        cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

        p_value = two_sided_p_value(float(t_stat), float(df))
        magnitude = (
            "negligible" if abs(cohens_d) < 0.2
            else "small" if abs(cohens_d) < 0.5
            else "medium" if abs(cohens_d) < 0.8
            else "large"
        )
        lines.append(
            f"\nWelch's t-test: t({df:.1f}) = {t_stat:.3f}, p = {p_value:.5g} (two-sided)\n"
            f"Mean difference ({name_a} - {name_b}): {mean_diff:+.4f}\n"
            f"Cohen's d = {cohens_d:+.3f} ({magnitude} effect)\n\n"
            "Report the effect size alongside p: with a large n, a trivial "
            "difference will still reach significance."
        )
    else:
        lines.append(
            "\nMore than two groups: pairwise t-tests would need a "
            "multiple-comparison correction. Use run_python_analysis for ANOVA."
        )

    log_event(logger, "groups compared", group_column=group_column, value_column=value_column)
    return "\n".join(lines)
