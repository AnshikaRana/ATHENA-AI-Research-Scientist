"""Working-memory behaviour: deduplication, citation validation, persistence."""

from __future__ import annotations

import pytest

from athena.context import ResearchContext
from athena.schemas import Paper, Stage


def make_paper(paper_id: str, title: str) -> Paper:
    return Paper(
        paper_id=paper_id, title=title, authors=["A Author"], year=2024,
        venue="Venue", abstract="Abstract text.", url="http://example.com",
        source="arxiv", citation_count=1,
    )


def test_dedup_by_id(sample_papers):
    ctx = ResearchContext(run_id="r", session_id="s", research_question="q")
    assert ctx.add_papers(sample_papers) == 2
    assert ctx.add_papers(sample_papers) == 0  # same ids, nothing new


def test_dedup_by_normalised_title():
    """The same paper from two databases has different ids but one title."""
    ctx = ResearchContext(run_id="r", session_id="s", research_question="q")
    added = ctx.add_papers([
        make_paper("arxiv:1", "Graph Neural Networks"),
        make_paper("doi:10.1/x", "graph   neural networks!"),
    ])
    assert added == 1
    assert len(ctx.corpus) == 1


def test_unknown_citations_detects_fabrication(sample_papers):
    ctx = ResearchContext(run_id="r", session_id="s", research_question="q")
    ctx.add_papers(sample_papers)
    unknown = ctx.unknown_citations(["arxiv:2205.13328", "arxiv:9999.99999"])
    assert unknown == ["arxiv:9999.99999"]


def test_serialisation_round_trip(sample_papers, tmp_settings):
    ctx = ResearchContext(run_id="r1", session_id="s1", research_question="q")
    ctx.add_papers(sample_papers)
    ctx.stage = Stage.CITATIONS
    ctx.reflection_round = 2
    ctx.record_error("openalex unavailable")
    ctx.record_approval("save_research_artifact", "{}", "approved", "looks fine")

    restored = ResearchContext.from_dict(ctx.to_dict())

    assert restored.run_id == "r1"
    assert restored.stage is Stage.CITATIONS
    assert restored.reflection_round == 2
    assert len(restored.corpus) == 2
    assert restored.corpus["doi:10.1109/tnn.2008.2005605"].citation_count == 9686
    assert restored.errors == ctx.errors
    assert restored.approvals[0].decision == "approved"


def test_cost_ledger_estimates():
    ctx = ResearchContext(run_id="r", session_id="s", research_question="q")
    ctx.costs.input_tokens = 1_000_000
    ctx.costs.output_tokens = 500_000
    assert ctx.costs.total_tokens == 1_500_000
    # Explicit rates: 1M in at $2 + 0.5M out at $8 = 2 + 4
    assert ctx.costs.estimated_usd(in_rate=2.00, out_rate=8.00) == 6.0


def test_cost_ledger_uses_the_configured_model_rates():
    """A gpt-4o-mini run must not be priced at gpt-4o rates.

    The ledger used to hardcode the expensive tier, overstating spend ~13x in
    the UI's "Est. $" tile.
    """
    from athena.context import _MODEL_RATES

    ctx = ResearchContext(run_id="r", session_id="s", research_question="q")
    ctx.costs.input_tokens = 1_000_000
    ctx.costs.output_tokens = 1_000_000

    mini_in, mini_out = _MODEL_RATES["gpt-4o-mini"]
    full_in, full_out = _MODEL_RATES["gpt-4o"]
    assert ctx.costs.estimated_usd(in_rate=mini_in, out_rate=mini_out) == pytest.approx(0.75)
    assert ctx.costs.estimated_usd(in_rate=full_in, out_rate=full_out) == pytest.approx(12.50)
    # Longest-prefix match: "gpt-4o-mini" must not resolve to the "gpt-4o" row.
    best = max((p for p in _MODEL_RATES if "gpt-4o-mini".startswith(p)), key=len)
    assert best == "gpt-4o-mini"
