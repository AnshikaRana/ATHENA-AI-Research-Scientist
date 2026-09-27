"""Guardrails, especially the deterministic citation check."""

from __future__ import annotations

import pytest
from athena.runtime import RunContextWrapper

from athena.guardrails import (
    _heuristic_relevance,
    citation_integrity_guardrail,
    extract_citations,
)
from tests.helpers import make_context


def test_extracts_every_citation_scheme():
    text = (
        "Supported by [arxiv:2301.00001] and doi:10.1109/tnn.2008.2005605, "
        "plus openalex:W2116341502 and upload:my_paper."
    )
    assert extract_citations(text) == {
        "arxiv:2301.00001",
        "doi:10.1109/tnn.2008.2005605",
        "openalex:W2116341502",
        "upload:my_paper",
    }


def test_trailing_punctuation_is_stripped():
    assert extract_citations("see (arxiv:1234.5678).") == {"arxiv:1234.5678"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hi", False),
        ("hello", False),
        ("What limits the expressive power of message passing GNNs?", True),
    ],
)
def test_offline_relevance_heuristic(text, expected):
    assert _heuristic_relevance(text).is_research_question is expected


class _FakeAgent:
    name = "Test Agent"


class _Output:
    def __init__(self, text: str) -> None:
        self._text = text

    def model_dump_json(self) -> str:
        return self._text


async def test_fabricated_citation_trips_the_guardrail(sample_papers):
    ctx = make_context()
    ctx.add_papers(sample_papers)
    result = await citation_integrity_guardrail.run(
        RunContextWrapper(ctx), _FakeAgent(), _Output("Claim grounded in arxiv:9999.99999")
    )
    assert result.output.tripwire_triggered is True
    assert "arxiv:9999.99999" in result.output.output_info["fabricated"]


async def test_real_citation_passes(sample_papers):
    ctx = make_context()
    ctx.add_papers(sample_papers)
    result = await citation_integrity_guardrail.run(
        RunContextWrapper(ctx), _FakeAgent(), _Output("Grounded in arxiv:2205.13328")
    )
    assert result.output.tripwire_triggered is False


async def test_guardrail_stays_quiet_before_the_corpus_exists():
    """Firing on an empty corpus would block the planning stage every run."""
    result = await citation_integrity_guardrail.run(
        RunContextWrapper(make_context()), _FakeAgent(), _Output("arxiv:1234.5678")
    )
    assert result.output.tripwire_triggered is False
    assert result.output.output_info["checked"] is False
