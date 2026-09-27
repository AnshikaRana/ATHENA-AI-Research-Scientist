"""End-to-end orchestration tests.

These run the real pipeline - real agents, real tools, real guardrails, real
approval interrupts - with only two things stubbed: the model (already a fixture
in offline mode) and the outbound HTTP calls, so the suite is hermetic and fast.
"""

from __future__ import annotations

import pytest

from athena.errors import ApprovalDeferred
from athena.pipeline import ResearchPipeline
from athena.schemas import Paper, Stage

QUESTION = "What limits the expressive power of message passing graph neural networks?"


def _fake_papers(prefix: str, count: int = 4) -> list[Paper]:
    return [
        Paper(
            paper_id=f"{prefix}:{i}",
            title=f"{prefix.title()} paper {i} on message passing expressivity",
            authors=[f"Author {i}"],
            year=2020 + i,
            venue="Test Venue",
            abstract=(
                "Message passing neural networks are bounded above by the "
                f"Weisfeiler-Lehman test. Study {i} examines subgraph methods "
                "that exceed 1-WL expressive power on molecular graphs."
            ),
            url=f"http://example.com/{prefix}/{i}",
            source="arxiv",
            citation_count=10 * i,
        )
        for i in range(count)
    ]


@pytest.fixture
def stub_network(monkeypatch):
    """Replace the three literature APIs with deterministic local fixtures."""
    from athena.tools import literature

    async def fake_arxiv(query, limit):
        return _fake_papers("arxiv", limit)

    async def fake_openalex(query, limit):
        return _fake_papers("openalex", limit)

    async def fake_crossref(query, limit):
        return _fake_papers("crossref", limit)

    monkeypatch.setitem(literature.SOURCES, "arxiv", fake_arxiv)
    monkeypatch.setitem(literature.SOURCES, "openalex", fake_openalex)
    monkeypatch.setitem(literature.SOURCES, "crossref", fake_crossref)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_full_study_produces_every_artefact(tmp_settings, stub_network):
    pipeline = ResearchPipeline(session_id="t_full")
    ctx = await pipeline.run_study(QUESTION)

    assert ctx.plan is not None
    assert ctx.literature is not None
    assert ctx.synthesis is not None
    assert ctx.gap_analysis is not None
    assert ctx.hypotheses is not None
    assert ctx.experiment is not None
    assert ctx.citations is not None
    assert ctx.report is not None


async def test_all_eight_stages_execute_in_order(tmp_settings, stub_network):
    pipeline = ResearchPipeline(session_id="t_order")
    await pipeline.run_study(QUESTION)

    executed = [stage.stage for stage in pipeline.stages]
    expected = [
        Stage.PLANNING, Stage.LITERATURE, Stage.SYNTHESIS, Stage.GAP_ANALYSIS,
        Stage.HYPOTHESIS, Stage.EXPERIMENT, Stage.CITATIONS, Stage.REPORT,
    ]
    assert executed == expected
    assert all(stage.ok for stage in pipeline.stages)


async def test_tools_actually_run(tmp_settings, stub_network):
    pipeline = ResearchPipeline(session_id="t_tools")
    await pipeline.run_study(QUESTION)

    used = pipeline.hooks.tool_usage()
    for tool in ("search_literature", "index_corpus", "rag_search",
                 "statistical_power_analysis", "format_citations",
                 "save_research_artifact"):
        assert tool in used, f"{tool} was never called"


async def test_corpus_is_deduplicated_across_sources(tmp_settings, stub_network):
    pipeline = ResearchPipeline(session_id="t_dedup")
    ctx = await pipeline.run_study(QUESTION)
    assert len(ctx.corpus) == len({p.title.lower() for p in ctx.corpus.values()})


# ---------------------------------------------------------------------------
# Human approval
# ---------------------------------------------------------------------------


async def test_approval_gate_fires_and_is_audited(tmp_settings, stub_network):
    seen = []

    async def approve(item):
        seen.append(item.name)
        return True, "approved in test"

    pipeline = ResearchPipeline(session_id="t_approve", approval_handler=approve)
    ctx = await pipeline.run_study(QUESTION)

    assert "save_research_artifact" in seen
    assert ctx.approvals[0].decision == "approved"
    assert ctx.approvals[0].note == "approved in test"


async def test_rejection_is_recorded_and_run_continues(tmp_settings, stub_network):
    async def reject(item):
        return False, "not authorised"

    pipeline = ResearchPipeline(session_id="t_reject", approval_handler=reject)
    ctx = await pipeline.run_study(QUESTION)

    assert ctx.approvals[0].decision == "rejected"
    assert ctx.report is not None


async def test_deferred_approval_checkpoints_and_resumes(tmp_settings, stub_network):
    """The decision must be able to outlive the process that asked for it."""
    async def defer(item):
        raise ApprovalDeferred(item.name, "checkpoint")

    pipeline = ResearchPipeline(
        session_id="t_defer", run_id="defer_run", approval_handler=defer
    )
    with pytest.raises(ApprovalDeferred):
        await pipeline.run_study(QUESTION)

    assert pipeline.state_checkpoint_path().exists()
    assert pipeline.context_checkpoint_path().exists()

    # A fresh pipeline, as if the process had restarted.
    resumed = ResearchPipeline(session_id="t_resume", run_id="defer_run")
    ctx = resumed.load_checkpointed_context()
    assert ctx is not None
    assert len(ctx.corpus) > 0, "the corpus must survive the restart"
    assert ctx.synthesis is not None

    result = await resumed.resume_from_checkpoint(
        resumed.team.scientific_writer, ctx, approve=True, note="approved later"
    )
    assert result is not None
    assert ctx.approvals[-1].decision == "approved"


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


async def test_a_failing_source_does_not_kill_the_search(tmp_settings, monkeypatch):
    """One dead database must degrade the corpus, not the run."""
    from athena.tools import literature

    async def working(query, limit):
        return _fake_papers("arxiv", limit)

    async def broken(query, limit):
        raise ConnectionError("simulated outage")

    monkeypatch.setitem(literature.SOURCES, "arxiv", working)
    monkeypatch.setitem(literature.SOURCES, "openalex", broken)
    monkeypatch.setitem(literature.SOURCES, "crossref", broken)

    pipeline = ResearchPipeline(session_id="t_degrade")
    ctx = await pipeline.run_study(QUESTION)

    assert len(ctx.corpus) > 0, "the working source should still populate the corpus"
    assert any("openalex" in error for error in ctx.errors)
    assert ctx.report is not None


async def test_empty_corpus_aborts_before_ungrounded_synthesis(tmp_settings, monkeypatch):
    """With no evidence, the pipeline must stop rather than invent findings."""
    from athena.tools import literature

    async def nothing(query, limit):
        return []

    for name in ("arxiv", "openalex", "crossref"):
        monkeypatch.setitem(literature.SOURCES, name, nothing)

    pipeline = ResearchPipeline(session_id="t_empty")
    ctx = await pipeline.run_study(QUESTION)

    assert ctx.synthesis is None
    assert ctx.report is None
    assert any("No papers retrieved" in error for error in ctx.errors)


async def test_manifest_captures_the_run(tmp_settings, stub_network):
    pipeline = ResearchPipeline(session_id="t_manifest")
    ctx = await pipeline.run_study(QUESTION)

    manifest = pipeline.run_manifest(ctx)
    assert manifest["mode"] == "offline"
    assert len(manifest["stages"]) == 8
    assert manifest["tool_usage"]
    assert manifest["approvals"]
    assert pipeline.save_manifest(ctx).exists()


# ---------------------------------------------------------------------------
# Structured-output recovery
# ---------------------------------------------------------------------------


async def test_off_schema_final_answer_is_repaired(tmp_settings, stub_network, monkeypatch):
    """A model that echoes a tool's status payload must not lose the stage.

    Observed live: after the human approved `save_research_artifact`, the
    Scientific Writer answered with the tool's own {"status": "success", ...}
    object instead of a ResearchReport, and the run died. The runner now
    re-asks with the tools withheld and the reply pinned to the schema.
    """
    from athena.mock import model as mock_model

    original = mock_model._fixture
    state = {"report_calls": 0}

    def status_object_first(schema_name):
        if schema_name == "ResearchReport":
            state["report_calls"] += 1
            if state["report_calls"] == 1:
                return {
                    "status": "success",
                    "message": "The research dossier has been successfully saved.",
                }
        return original(schema_name)

    monkeypatch.setattr(mock_model, "_fixture", status_object_first)

    pipeline = ResearchPipeline(session_id="t_repair")
    ctx = await pipeline.run_study(QUESTION)

    assert state["report_calls"] >= 2, "the runner should have re-asked for the report"
    assert ctx.report is not None, "the repaired reply must complete the stage"
    assert ctx.report.title


async def test_repeated_tool_call_is_refused_and_forces_the_answer(
    tmp_settings, stub_network, monkeypatch
):
    """An agent that re-issues a completed call must not loop forever.

    Observed live: after `save_research_artifact` succeeded, the Scientific
    Writer kept calling it again instead of returning the report, and the run
    came back with no output at all. The runner now refuses the duplicate and
    pushes the agent to its final answer.
    """
    from athena.mock import model as mock_model

    calls: list[str] = []
    real_next = mock_model.MockModel._next_tool_call

    def always_resave(self, schema_name, available, calls_made):
        # The Writer never voluntarily stops calling the gated save tool.
        if schema_name == "ResearchReport" and "save_research_artifact" in available:
            calls.append(schema_name)
            return ("save_research_artifact", {"artifact_type": "report"})
        return real_next(self, schema_name, available, calls_made)

    monkeypatch.setattr(mock_model.MockModel, "_next_tool_call", always_resave)

    pipeline = ResearchPipeline(session_id="t_loop")
    ctx = await pipeline.run_study(QUESTION)

    assert ctx.report is not None, "the run must still produce a report"
    saves = pipeline.hooks.tool_usage().get("save_research_artifact", 0)
    assert saves <= 2, f"the duplicate call should have been refused, ran {saves}x"


async def test_runs_do_not_share_a_conversation_transcript(tmp_settings, stub_network):
    """Two runs under one session id must not replay each other.

    They used to: the UI pins a single session id, so every run prepended every
    earlier run's transcript. The prompt grew without bound, and the Scientific
    Writer cited paper_ids from a previous run's corpus - which the citation
    guardrail then correctly rejected as fabricated.
    """
    first = ResearchPipeline(session_id="shared")
    await first.run_study(QUESTION)

    second = ResearchPipeline(session_id="shared")
    assert second.session.session_id != first.session.session_id

    carried_over = await second.session.get_items()
    assert carried_over == [], "a fresh run must start with an empty transcript"


async def test_report_brief_names_the_citable_ids(tmp_settings, stub_network):
    """The writer is told exactly which ids exist, so it cannot cite from memory."""
    pipeline = ResearchPipeline(session_id="t_brief")
    ctx = await pipeline.run_study(QUESTION)

    brief = pipeline._build_report_brief(ctx, QUESTION)
    assert "Citable paper_ids" in brief
    for paper_id in list(ctx.corpus)[:3]:
        assert paper_id in brief

    # Every citing stage gets the same list, not just the writer.
    note = pipeline._citable_ids(ctx)
    assert "Citable paper_ids" in note
    for paper_id in list(ctx.corpus)[:3]:
        assert paper_id in note


async def test_report_stage_does_not_replay_the_run_transcript(
    tmp_settings, stub_network, monkeypatch
):
    """The writer's brief is built from the context, so the transcript is waste.

    Replaying every earlier stage on top of an already-complete brief pushed a
    single request to 42,606 tokens against a 30,000 TPM account ceiling, and
    the report stage died with a 429 that no retry could clear.
    """
    seen: list[bool] = []
    original = ResearchPipeline._run_agent

    async def record(self, agent, prompt, ctx, *, use_session=True, **kwargs):
        if agent.name == "Scientific Writer":
            seen.append(use_session)
        return await original(self, agent, prompt, ctx, use_session=use_session, **kwargs)

    monkeypatch.setattr(ResearchPipeline, "_run_agent", record)

    pipeline = ResearchPipeline(session_id="t_nosession")
    await pipeline.run_study(QUESTION)

    assert seen, "the writer should have run"
    assert not any(seen), "the report stage must not replay the session transcript"


# ---------------------------------------------------------------------------
# Document-grounded mode and the approval resume
# ---------------------------------------------------------------------------

_SOURCE_TEXT = (
    "EyeTest: Identity-Preserving Test Synthesis\n"
    "Ada Lovelace, Alan Turing\n"
    "arXiv:2312.12028v5 [cs.CV] 1 May 2025\n"
    "Abstract\nWe propose a model whose AUC is 0.996 on the test set. "
    + "Message passing results are reported in Table 1. " * 40
)


@pytest.fixture
def source_pdf(tmp_path, monkeypatch):
    """A stand-in PDF: the file exists, and its parsed text is fixed."""
    from athena.tools import documents

    path = tmp_path / "EyeTest paper.pdf"
    path.write_bytes(b"%PDF-1.4 stand-in")
    monkeypatch.setattr(
        documents,
        "extract_pdf_text",
        lambda p, max_pages=40: (
            "EyeTest: Identity-Preserving Test Synthesis", _SOURCE_TEXT, 3
        ),
    )
    return str(path)


async def test_document_mode_uses_only_the_upload(tmp_settings, stub_network, source_pdf):
    """A document-grounded study must not reach for the web at all."""
    pipeline = ResearchPipeline(session_id="t_docs")
    ctx = await pipeline.run_study(QUESTION, attachments=[source_pdf], source_mode="documents")

    assert list(ctx.corpus) == ["upload:eyetest_paper"]
    assert ctx.source_texts["upload:eyetest_paper"].startswith("EyeTest")
    usage = pipeline.hooks.tool_usage()
    assert not {"search_literature", "rag_search", "recall_memory"} & set(usage)
    assert ctx.report is not None

    paper = ctx.corpus["upload:eyetest_paper"]
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.year == 2025
    assert paper.url == "https://arxiv.org/abs/2312.12028"


async def test_document_mode_without_a_pdf_stops_cleanly(tmp_settings, stub_network):
    pipeline = ResearchPipeline(session_id="t_docs_empty")
    ctx = await pipeline.run_study(QUESTION, attachments=[], source_mode="documents")
    assert ctx.report is None
    assert any("needs at least one readable PDF" in e for e in ctx.errors)


async def test_a_checkpoint_is_resumed_exactly_once(tmp_settings, stub_network, source_pdf):
    """Two Approve clicks on one pause must produce one report, not two runs.

    Observed live: repeated clicks resumed the same checkpoint dozens of times,
    each paying for a writer turn and saving another copy, while the report
    itself was never handed back to the UI.
    """
    import asyncio

    async def defer(item):
        raise ApprovalDeferred(item.name, "checkpoint")

    first = ResearchPipeline(session_id="t_once", run_id="once_run", approval_handler=defer)
    with pytest.raises(ApprovalDeferred):
        await first.run_study(QUESTION, attachments=[source_pdf], source_mode="documents")

    a = ResearchPipeline(session_id="t_once", run_id="once_run")
    b = ResearchPipeline(session_id="t_once", run_id="once_run")
    ctx_a, ctx_b = a.load_checkpointed_context(), b.load_checkpointed_context()
    results = await asyncio.gather(
        a.resume_from_checkpoint(a.team.scientific_writer, ctx_a, approve=True),
        b.resume_from_checkpoint(b.team.scientific_writer, ctx_b, approve=True),
    )

    assert sum(r is not None for r in results) == 1
    winner = ctx_a if results[0] is not None else ctx_b
    assert winner.report is not None, "the resumed report must land on the context"
    reports = list(tmp_settings.artifacts_dir.glob("*.md"))
    assert len(reports) == 1


def test_an_approved_save_is_not_gated_again(tmp_settings):
    from athena.context import ResearchContext
    from athena.tools.artifacts import _approval_required

    class Wrapper:
        context = ResearchContext(run_id="r", session_id="s", research_question="q")

    assert _approval_required(Wrapper(), {}, "c1") is True
    Wrapper.context.record_approval("save_research_artifact", "{}", "approved")
    assert _approval_required(Wrapper(), {}, "c2") is False
