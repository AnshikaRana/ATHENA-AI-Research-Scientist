"""Tool-layer behaviour, especially the parts that must be exactly right."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from athena.tools.analysis import UnsafeCodeError, required_sample_size, validate_code
from athena.tools.citations import _bibtex_key, render_reference
from athena.tools.datasets import two_sided_p_value
from athena.tools.literature import _reconstruct_abstract, _safe_year
from tests.helpers import call_tool, make_context


# --- Statistics -------------------------------------------------------------


@pytest.mark.parametrize(
    "effect_size,expected",
    [(0.2, 394), (0.5, 64), (0.8, 26)],  # Cohen's published tables
)
def test_power_analysis_matches_cohen_tables(effect_size, expected):
    n = required_sample_size(effect_size, alpha=0.05, power=0.80)
    assert abs(n - expected) <= 2, f"n={n} deviates from Cohen's {expected}"


def test_power_analysis_rejects_nonsense():
    from athena.errors import ToolExecutionError

    for bad in [(0.0, 0.05, 0.8), (0.5, 0.0, 0.8), (0.5, 0.05, 1.5)]:
        with pytest.raises(ToolExecutionError):
            required_sample_size(*bad)


@pytest.mark.parametrize(
    "t_stat,df,expected",
    [(2.0, 10, 0.0734), (2.228, 10, 0.0500), (1.96, 1000, 0.0502), (0.0, 10, 1.0)],
)
def test_p_value_matches_t_tables(t_stat, df, expected):
    assert two_sided_p_value(t_stat, df) == pytest.approx(expected, abs=5e-4)


# --- Sandbox ---------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "open('/etc/passwd')",
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('os').system('ls')",
        "eval('1+1')",
    ],
)
def test_sandbox_blocks_escapes(code):
    with pytest.raises(UnsafeCodeError):
        validate_code(code)


def test_sandbox_allows_legitimate_analysis():
    validate_code("import math\nimport numpy as np\nprint(math.sqrt(2))")


async def test_sandbox_returns_runtime_errors_as_text():
    """A traceback is information the agent can act on, not a crash."""
    from athena.tools.analysis import run_python_analysis

    result = await call_tool(
        run_python_analysis, make_context(), code="print(1/0)", purpose="error path"
    )
    assert "ZeroDivisionError" in result


async def test_sandbox_executes_and_returns_stdout():
    from athena.tools.analysis import run_python_analysis

    result = await call_tool(
        run_python_analysis, make_context(),
        code="print(sum(range(101)))", purpose="sum test",
    )
    assert "5050" in result


# --- Literature parsing -----------------------------------------------------


@pytest.mark.parametrize(
    "date_parts,expected",
    [([[2024, 3]], 2024), ([[None]], 0), ([[]], 0), (None, 0), ([], 0), ([[2020]], 2020)],
)
def test_crossref_year_parsing_survives_real_shapes(date_parts, expected):
    """Crossref really does emit every one of these."""
    assert _safe_year(date_parts) == expected


def test_openalex_inverted_index_reconstruction():
    inverted = {"Graph": [0], "neural": [1], "networks": [2], "learn": [3]}
    assert _reconstruct_abstract(inverted) == "Graph neural networks learn"
    assert _reconstruct_abstract(None) == ""


# --- Citations --------------------------------------------------------------


def test_citation_styles_render_distinctly(sample_papers):
    paper = sample_papers[1]  # Scarselli 2008
    rendered = {style: render_reference(paper, style) for style in ("APA", "IEEE", "MLA", "Nature")}
    assert len(set(rendered.values())) == 4
    assert "Scarselli, F." in rendered["APA"]
    assert "F. Scarselli" in rendered["IEEE"]
    for text in rendered.values():
        assert "2008" in text
        assert ".." not in text  # no doubled punctuation


def test_bibtex_keys_are_stable_and_slugged(sample_papers):
    key = _bibtex_key(sample_papers[1])
    assert key == _bibtex_key(sample_papers[1])
    assert key.isalnum() and key.islower()


async def test_format_citations_refuses_unknown_ids(sample_papers):
    from athena.tools.citations import format_citations

    ctx = make_context()
    ctx.add_papers(sample_papers)
    output = await call_tool(
        format_citations, ctx,
        paper_ids=["arxiv:2205.13328", "arxiv:0000.00000"], style="APA",
    )
    assert "arxiv:0000.00000" in output
    assert "UNVERIFIED" in output


# --- Datasets ---------------------------------------------------------------


async def test_dataset_analysis_reports_shape_and_missing(tmp_settings):
    from athena.tools.datasets import analyze_dataset

    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    path = tmp_settings.uploads_dir / "t.csv"
    frame.to_csv(path, index=False)

    output = await call_tool(
        analyze_dataset, make_context(), file_path=str(path), target_column="b"
    )
    assert "4 rows x 2 columns" in output
    assert "a: 1" in output  # the missing value is reported


async def test_group_comparison_detects_a_real_effect(tmp_settings):
    from athena.tools.datasets import compare_groups

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "group": ["control"] * 50 + ["treatment"] * 50,
        "score": np.concatenate([rng.normal(0.70, 0.05, 50), rng.normal(0.80, 0.05, 50)]),
    })
    path = tmp_settings.uploads_dir / "g.csv"
    frame.to_csv(path, index=False)

    output = await call_tool(
        compare_groups, make_context(),
        file_path=str(path), group_column="group", value_column="score",
    )
    assert "Welch's t-test" in output
    assert "large effect" in output
    assert "p = " in output


# --- Upload ids -------------------------------------------------------------


def test_uploaded_pdf_gets_a_citable_id():
    """A filename with spaces used to produce an uncitable paper_id.

    The citation pattern stops at whitespace, so
    "upload:LitSearch A Retrieval Benchmark" was only ever extracted as
    "upload:LitSearch" - which matched nothing and was reported as a
    fabricated citation, blocking every agent that cited the user's own paper.
    """
    from athena.guardrails import extract_citations
    from athena.tools.documents import _slug

    paper_id = "upload:" + _slug("LitSearch A Retrieval Benchmark for Scientific Literature Search")
    assert " " not in paper_id
    assert extract_citations(f"as shown in {paper_id}, retrieval is hard") == {paper_id}


def test_upload_slug_is_stable_and_never_empty():
    from athena.tools.documents import _slug

    assert _slug("A B  C") == _slug("A_B_C") == "a_b_c"
    assert _slug("!!!") == "document"
    assert len(_slug("x" * 200)) <= 48
