"""Streamlit interface for Athena.

    streamlit run app.py

The approval flow here is the reason the checkpoint machinery exists. A web
request cannot block waiting for a human to click a button, so when a gated tool
is reached the run is checkpointed to disk and abandoned; the buttons that appear
resume it in a later request. The same mechanism lets the user close the tab and
come back.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from athena.config import get_settings
from athena.context import ResearchContext
from athena.errors import ApprovalDeferred
from athena.logging_setup import read_run_events
from athena.memory.long_term import LongTermMemory
from athena.pipeline import ResearchPipeline
from athena.schemas import ResearchReport


def _ensure_streamlit_runtime() -> None:
    """Re-launch under ``streamlit run`` when executed as a plain script.

    An IDE's Run button runs ``python app.py``, which Streamlit cannot serve:
    it prints a wall of "missing ScriptRunContext" warnings and exits without
    ever starting a server. Detect that and exec the correct command instead,
    so the Run button just works.
    """
    if st.runtime.exists():
        return

    import subprocess
    import sys

    print(
        "app.py is a Streamlit app and cannot run as a plain script; "
        f"relaunching via: {sys.executable} -m streamlit run {__file__}",
        flush=True,
    )
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]
        )
    )


_ensure_streamlit_runtime()

st.set_page_config(page_title="Athena - AI Research Scientist", page_icon="🔬", layout="wide")

STAGE_ORDER = [
    "planning", "literature", "synthesis", "hypothesis", "experiment", "review", "report",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro):
    """Streamlit reruns synchronously; give each call its own event loop."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def save_uploads(files) -> list[str]:
    settings = get_settings()
    paths: list[str] = []
    for file in files or []:
        destination = settings.uploads_dir / file.name
        destination.write_bytes(file.getbuffer())
        paths.append(str(destination))
    return paths


async def deferring_handler(item):
    """Never decide inside the request; checkpoint and hand control to the UI."""
    raise ApprovalDeferred(item.name, "awaiting UI decision")


def init_state() -> None:
    defaults = {
        "session_id": "streamlit",
        "pending_run_id": None,
        "pending_tool": None,
        "pending_args": None,
        "result_ctx": None,
        "manifest": None,
        "last_run_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_trace(run_id: str) -> None:
    events = read_run_events(run_id)
    if not events:
        st.info("No trace events recorded yet.")
        return

    interesting = [
        e for e in events
        if e.get("kind") in {"agent_start", "agent_end", "handoff", "tool_start", "tool_end"}
    ]
    st.caption(f"{len(events)} log records, {len(interesting)} agent/tool events")

    icons = {
        "agent_start": "▶", "agent_end": "■", "handoff": "→",
        "tool_start": "🔧", "tool_end": "✓",
    }
    lines = []
    for event in interesting[-160:]:
        kind = event.get("kind", "")
        icon = icons.get(kind, "·")
        if kind == "handoff":
            lines.append(f"{icon} HANDOFF  {event.get('from_agent')} → {event.get('to_agent')}")
        elif kind in {"tool_start", "tool_end"}:
            duration = event.get("duration_s")
            suffix = f"  ({duration}s)" if duration else ""
            lines.append(f"{icon} {kind:10} {event.get('agent', ''):24} {event.get('tool', '')}{suffix}")
        else:
            lines.append(f"{icon} {kind:10} {event.get('agent', '')}")
    st.code("\n".join(lines), language=None)


def render_report(report: ResearchReport) -> None:
    st.subheader(report.title)
    st.markdown(f"**Executive summary**\n\n{report.executive_summary}")
    for section in report.sections:
        with st.expander(section.heading, expanded=True):
            st.markdown(section.body_markdown)
    columns = st.columns(2)
    with columns[0]:
        st.markdown("**Key findings**")
        for finding in report.key_findings:
            st.markdown(f"- {finding}")
    with columns[1]:
        st.markdown("**Limitations**")
        for limitation in report.limitations:
            st.markdown(f"- {limitation}")
    if report.references:
        with st.expander(f"References ({len(report.references)})"):
            for reference in report.references:
                st.markdown(f"- {reference}")
    st.caption(f"Overall confidence: {report.confidence.value}")


def render_context(ctx: ResearchContext) -> None:
    summary = ctx.summary()
    cols = st.columns(4)
    cols[0].metric("Papers", summary["corpus_size"])
    cols[1].metric("Chunks", summary["indexed_chunks"])
    cols[2].metric("Approvals", summary["approvals"])
    cols[3].metric("Est. $", summary["estimated_usd"])

    tabs = st.tabs(
        [
            "Report", "Synthesis", "Gaps", "Hypotheses", "Experiment",
            "Citations", "Corpus", "Issues",
        ]
    )
    with tabs[0]:
        if ctx.report:
            render_report(ctx.report)
            saved = ctx.saved_artifacts.get("report")
            if saved and Path(saved).exists():
                st.download_button(
                    "Download report (.md)",
                    Path(saved).read_text(encoding="utf-8"),
                    file_name=Path(saved).name,
                    mime="text/markdown",
                )
                st.caption(f"Saved with your approval to `{saved}`")
        else:
            st.warning("No report was produced. The Issues tab records why.")
    with tabs[1]:
        if ctx.synthesis:
            st.markdown(ctx.synthesis.summary)
            for claim in ctx.synthesis.key_claims:
                flag = " ⚠️ contested" if claim.contested else ""
                st.markdown(
                    f"- **{claim.statement}** "
                    f"`{claim.confidence.value}`{flag} — {', '.join(claim.supporting_paper_ids)}"
                )
            st.markdown("**Gaps**")
            for gap in ctx.synthesis.gaps:
                st.markdown(f"- {gap.gap} — *{gap.why_it_matters}*")
        else:
            st.info("No synthesis produced.")
    with tabs[2]:
        if ctx.gap_analysis:
            st.caption(f"Priority gap: {ctx.gap_analysis.priority_gap_id}")
            st.markdown(f"*{ctx.gap_analysis.prioritisation_rationale}*")
            for gap in ctx.gap_analysis.gaps:
                with st.expander(f"[{gap.gap_id}] {gap.title[:80]}"):
                    st.markdown(gap.description)
                    meta = st.columns(3)
                    meta[0].metric("Severity", gap.severity.value)
                    meta[1].metric("Type", gap.gap_type)
                    meta[2].metric("Tractability", f"{gap.tractability}/10")
                    st.markdown(f"**Why unaddressed.** {gap.why_unaddressed}")
                    st.markdown(f"**Potential impact.** {gap.potential_impact}")
                    st.caption(f"Evidence: {', '.join(gap.evidence_paper_ids)}")
            st.info(f"Corpus limitations: {ctx.gap_analysis.corpus_limitations}")
        else:
            st.info("No gap analysis produced.")
    with tabs[3]:
        if ctx.hypotheses:
            st.caption(f"Recommended: {ctx.hypotheses.recommended_hypothesis_id}")
            for hypothesis in ctx.hypotheses.hypotheses:
                with st.expander(f"{hypothesis.hypothesis_id}: {hypothesis.statement[:90]}"):
                    st.markdown(f"**Rationale.** {hypothesis.rationale}")
                    st.markdown(f"**Null hypothesis.** {hypothesis.null_hypothesis}")
                    score_cols = st.columns(3)
                    score_cols[0].metric("Novelty", hypothesis.novelty_score)
                    score_cols[1].metric("Feasibility", hypothesis.feasibility_score)
                    score_cols[2].metric("Impact", hypothesis.impact_score)
        else:
            st.info("No hypotheses produced.")
    with tabs[4]:
        if ctx.experiment:
            experiment = ctx.experiment
            st.markdown(f"### {experiment.title}")
            st.markdown(f"**Methodology.** {experiment.methodology}")
            power = experiment.power_analysis
            st.success(
                f"Computed sample size: **n = {power.required_sample_size} per group** "
                f"(d={power.effect_size}, α={power.alpha}, power={power.power})"
            )
            st.markdown(f"*{power.justification}*")
            st.markdown("**Procedure**")
            for index, step in enumerate(experiment.procedure_steps, start=1):
                st.markdown(f"{index}. {step}")
            st.markdown("**Threats to validity**")
            for threat in experiment.threats_to_validity:
                st.markdown(f"- {threat}")
        else:
            st.info("No experiment design produced.")
    with tabs[5]:
        if ctx.citations:
            bundle = ctx.citations
            st.caption(
                f"{len(bundle.citations)} verified in {bundle.style} style · "
                f"{len(bundle.unverified_ids)} unverified"
            )
            if bundle.unverified_ids:
                st.error(f"Unverified ids excluded: {bundle.unverified_ids}")
            st.markdown(bundle.reference_list_markdown)
            with st.expander("BibTeX"):
                st.code(bundle.bibtex_file, language="bibtex")
                st.download_button(
                    "Download .bib", bundle.bibtex_file,
                    file_name="references.bib", mime="text/plain",
                )
            st.caption(bundle.integrity_note)
        else:
            st.info("No bibliography produced.")
    with tabs[6]:
        if ctx.corpus:
            st.dataframe(
                [
                    {
                        "id": p.paper_id,
                        "title": p.title,
                        "year": p.year,
                        "citations": p.citation_count,
                        "source": p.source,
                    }
                    for p in sorted(ctx.corpus.values(), key=lambda p: -p.citation_count)
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Corpus is empty.")
    with tabs[7]:
        if ctx.errors:
            for error in ctx.errors:
                st.warning(error)
        else:
            st.success("No errors recorded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


init_state()
settings = get_settings()

st.title("🔬 Athena — AI Research Scientist")
st.caption(
    "Eight specialised agents · 18 tools · RAG over live literature · "
    "deterministic citation integrity · human-approved artefacts · checkpoint/resume"
)

with st.sidebar:
    st.header("Configuration")
    if settings.live:
        st.success("Live mode — OpenAI API configured")
        st.caption(
            f"Models: {settings.fast_model} (planning, gaps, citations) · "
            f"{settings.reasoning_model} (synthesis, hypotheses, experiment, report). "
            f"Hard cap per run: ${settings.max_run_usd:.2f}."
        )
    else:
        st.warning(
            "**Offline mode.** No `OPENAI_API_KEY_2` found, so model outputs are "
            "fixtures. Literature search, RAG, PDF parsing, audio transcription "
            "and the analysis sandbox are still real."
        )
    st.session_state.session_id = st.text_input("Session id", st.session_state.session_id)
    st.caption("Runs sharing a session id share conversation memory.")

    st.divider()
    st.subheader("Long-term memory")
    memory = LongTermMemory()
    st.json(memory.stats())
    recall_query = st.text_input("Recall across sessions")
    if recall_query:
        for record in run_async(memory.recall(recall_query, k=5)):
            st.markdown(f"**{record.topic}** `{record.kind}` ({record.score:.2f})")
            st.caption(record.content[:200])

    st.divider()
    st.subheader("Team")
    from athena.agents import build_team

    for spec in build_team().describe():
        st.markdown(f"**{spec['name']}** → `{spec['output_type']}`")
        st.caption(f"{len(spec['tools'])} tools · {len(spec['handoffs'])} handoffs")

# --- Pending approval takes over the page ---------------------------------

if st.session_state.pending_run_id:
    st.warning("### ⏸ Human approval required")
    st.markdown(
        f"Run `{st.session_state.pending_run_id}` is paused. The agent wants to call "
        f"**`{st.session_state.pending_tool}`**."
    )
    with st.expander("Arguments", expanded=True):
        st.code(st.session_state.pending_args or "{}", language="json")

    note = st.text_input("Note for the audit trail", "")
    approve_col, reject_col = st.columns(2)

    def _resume(approve: bool) -> None:
        run_id = st.session_state.pending_run_id
        # Clear the pending run *before* the slow part. A second click while the
        # writer is finishing triggers a rerun; with the flag already cleared it
        # finds nothing to resume instead of resuming the same run again.
        st.session_state.pending_run_id = None
        st.session_state.pending_tool = None
        st.session_state.pending_args = None
        st.session_state.last_run_id = run_id

        pipeline = ResearchPipeline(session_id=st.session_state.session_id, run_id=run_id)
        ctx = pipeline.load_checkpointed_context()
        if ctx is None:
            st.session_state.resume_error = "Checkpoint could not be loaded."
            return
        with st.spinner("Finishing the report…"):
            run_async(
                pipeline.resume_from_checkpoint(
                    pipeline.team.scientific_writer, ctx, approve=approve, note=note
                )
            )
        st.session_state.result_ctx = ctx
        st.session_state.manifest = pipeline.run_manifest(ctx)
        pipeline.save_manifest(ctx)

    if approve_col.button("✅ Approve and continue", type="primary", width="stretch"):
        _resume(True)
        st.rerun()
    if reject_col.button("❌ Reject", width="stretch"):
        _resume(False)
        st.rerun()

else:
    st.caption("Runs the deterministic eight-stage study pipeline.")
    if st.session_state.get("resume_error"):
        st.error(st.session_state.pop("resume_error"))

    question = st.text_area(
        "Research question",
        "",
        height=90,
        placeholder=(
            "e.g. What limits the expressive power of message passing graph "
            "neural networks?"
        ),
    )
    uploads = st.file_uploader(
        "Attach source material (PDF, figure image, or audio)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a"],
        accept_multiple_files=True,
        help=(
            "PDFs are parsed and indexed for retrieval. Images and audio require "
            "live mode."
        ),
    )

    has_pdf = any(f.name.lower().endswith(".pdf") for f in uploads or [])
    source_choice = st.radio(
        "Evidence source",
        ["Uploaded documents only", "Online literature (plus any uploads)"],
        index=0 if has_pdf else 1,
        horizontal=True,
        help=(
            "Documents only: the study is grounded solely in your PDFs - every "
            "agent reads their full text and web search is switched off. Online "
            "literature: agents search arXiv, OpenAlex and Crossref."
        ),
    )
    source_mode = "documents" if source_choice.startswith("Uploaded") else "web"
    if source_mode == "documents" and not has_pdf:
        st.info("Attach at least one PDF to run a document-grounded study.")

    can_run = bool(question.strip()) and (source_mode == "web" or has_pdf)
    if st.button("Run", type="primary", width="stretch", disabled=not can_run):
        attachments = save_uploads(uploads)
        pipeline = ResearchPipeline(
            session_id=st.session_state.session_id, approval_handler=deferring_handler
        )
        st.session_state.last_run_id = pipeline.run_id

        with st.status("Running the research pipeline…", expanded=True) as status:
            try:
                ctx = run_async(
                    pipeline.run_study(
                        question.strip(),
                        attachments=attachments or None,
                        source_mode=source_mode,
                    )
                )
                st.session_state.result_ctx = ctx
                st.session_state.manifest = pipeline.run_manifest(ctx)
                pipeline.save_manifest(ctx)
                status.update(label="Pipeline complete", state="complete")

            except ApprovalDeferred as exc:
                st.session_state.pending_run_id = pipeline.run_id
                st.session_state.pending_tool = exc.tool_name
                checkpoint = pipeline.state_checkpoint_path()
                arguments = "{}"
                if checkpoint.exists():
                    arguments = json.dumps(
                        {"checkpoint": str(checkpoint), "note": "resume to see full arguments"},
                        indent=2,
                    )
                st.session_state.pending_args = arguments
                status.update(label="Paused for human approval", state="error")
                st.rerun()

# --- Results ---------------------------------------------------------------

if st.session_state.result_ctx is not None:
    st.divider()
    render_context(st.session_state.result_ctx)

if st.session_state.last_run_id:
    with st.expander("Agent trace", expanded=False):
        render_trace(st.session_state.last_run_id)
if st.session_state.manifest:
    with st.expander("Run manifest (JSON)"):
        st.json(st.session_state.manifest)
