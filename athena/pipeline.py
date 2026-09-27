"""Orchestration: the eight-stage research pipeline.

This module is where the design decisions that matter actually live.

**Why a deterministic pipeline instead of pure LLM orchestration.** Research
stages have hard dependencies - you cannot synthesise evidence you have not
retrieved, and you cannot design an experiment for a hypothesis you have not
chosen. A model rediscovering that order on every run costs tokens, adds
failure modes, and produces a demo that behaves differently each time. So the
*sequence* is code. The *judgement* inside each stage is the model's.

The eight stages: plan, literature, synthesis, gap analysis, hypotheses,
experiment design, citations, report. The genuine parallelism is inside
``search_literature``, which fans out across three databases at once.

**Grounding.** Fabricated citations are still blocked deterministically: the
citation-integrity output guardrail rejects any ``paper_id`` an agent cites that
is not in the retrieved corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from athena.runtime import (
    Agent,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    RunState,
    MaxTurnsExceeded,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
    RunResult,
    SQLiteSession,
    ToolApprovalItem,
    set_tracing_disabled,
)

from .agents import ResearchTeam, build_team
from .config import get_settings
from .context import ResearchContext
from .errors import ApprovalDeferred, AthenaError
from .hooks import AthenaRunHooks
from .logging_setup import configure_logging, get_logger, log_event, new_run_id
from .schemas import (
    CitationBundle,
    GapAnalysis,
    ExperimentDesign,
    HypothesisSet,
    KnowledgeSynthesis,
    LiteratureReview,
    ResearchPlan,
    ResearchReport,
    Stage,
)

logger = get_logger("pipeline")

# How many times a resumed run may re-hit the gate before we stop re-approving.
_MAX_RESUME_REGATES = 3

# A fabricated citation is correctable: name the bad ids and re-ask.
_MAX_GUARDRAIL_RETRIES = 2

NL = "\n"

# Tools withheld in document-grounded mode. Web search would pull in papers the
# user did not supply; retrieval and memory recall are redundant when the full
# source text is in the prompt, and recall could surface findings from earlier,
# unrelated studies.
_DOCUMENT_MODE_BLOCKED_TOOLS = frozenset(
    {
        "search_literature", "get_paper_details", "commission_deep_sweep",
        "ingest_pdf", "describe_figure", "transcribe_audio",
        "index_corpus", "rag_search", "list_corpus", "recall_memory",
    }
)

DOCUMENT_MODE_RULES = """## Evidence rules for this run (they override any conflicting instruction)
This study is confined to the source document(s) the user supplied. Their full
text is reproduced below and is the ONLY evidence base.
- Every factual statement - methods, datasets, numbers, comparisons, limitations -
  must come from the source text. Quote figures exactly as printed (e.g. an AUC of
  "0.996 +/- 0.002") and say where they come from (Table 2, Section 6.1.2, Fig. 4).
- Cite the source by its paper_id in square brackets, e.g. [{paper_id}]. The
  source's own bibliography entries (e.g. "[4]") are not corpus entries: mention
  them as "ref. [4] in the source", never as a paper_id.
- Do not bring in results from outside knowledge. When the source does not report
  something, say it is not reported rather than filling the gap.
- Web search and retrieval tools are disabled for this run on purpose: the complete
  text is already in front of you, so read it directly."""

DOCUMENT_REPORT_STRUCTURE = """## Report structure for this run (replaces the default section list)
Write these sections, in this order, as a study of the question against the source:
  1. Introduction - the question, the source paper (title, authors, venue) and why
     the problem matters.
  2. Method Overview - what the proposed approach is and how it works: inputs, the
     architecture, each loss or component and what it is for, the training data.
  3. Evaluation Setup - training and test datasets, baselines, metrics, how the
     experimental conditions are defined and binned, how uncertainty is estimated.
  4. Results: Primary Evidence - the quantitative evidence that answers the
     question, organised by experimental condition. Include Markdown tables that
     reproduce the source's key numbers exactly, naming the table they came from.
  5. Results: Secondary Evidence and Trade-offs - ablations, quality/realism
     evidence, comparisons with alternative approaches, supplementary statistics.
  6. Answer to the Research Question - a direct, evidence-weighted verdict: where the
     approach helps, where it is neutral, where it is worse, and why (per the source).
  7. Research Gaps - from the gap analysis.
  8. Proposed Hypotheses - with scores and the recommendation.
  9. Proposed Experiment - including the computed sample size.
 10. Limitations - of the source's evidence and of this single-source analysis.
 11. Future Work.

Length: each section body up to 350 words (tables do not count); whole report
under 3,500 words. Every key_findings entry must carry a concrete number from the
source. The references field is the verified bibliography, verbatim."""

# An approval handler receives the pending tool call and returns
# (approved, note). The CLI implements one interactively; the UI implements one
# backed by a button; tests pass a lambda.
ApprovalHandler = Callable[[ToolApprovalItem], Awaitable[tuple[bool, str]]]


async def auto_approve(item: ToolApprovalItem) -> tuple[bool, str]:
    return True, "auto-approved (no interactive handler configured)"


@dataclass
class StageResult:
    """What happened in one stage, for the UI timeline and the audit log."""

    stage: Stage
    agent: str
    ok: bool
    detail: str
    duration_s: float


class ResearchPipeline:
    """Runs a full study, or a single targeted consult."""

    def __init__(
        self,
        *,
        session_id: str = "default",
        run_id: str | None = None,
        team: ResearchTeam | None = None,
        approval_handler: ApprovalHandler | None = None,
        verbose: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.run_id = run_id or new_run_id()
        configure_logging(self.run_id, verbose=verbose)

        self.session_id = session_id
        self.team = team or build_team()
        self.approval_handler = approval_handler or auto_approve
        self.hooks = AthenaRunHooks()
        self.stages: list[StageResult] = []

        # Conversation memory, scoped to this run. Sharing one transcript across
        # every run of a session made each run replay all its predecessors: the
        # prompt grew without bound, and agents cited paper_ids from an earlier
        # run's corpus that the integrity guardrail then - correctly - rejected
        # as fabricated. Knowledge that should outlive a run travels through
        # long-term memory (`remember_finding` / `recall_memory`) instead.
        self.session = SQLiteSession(
            f"{session_id}:{self.run_id}", str(self.settings.session_db)
        )

        self.run_config = self._build_run_config()

    # -- configuration ----------------------------------------------------

    def _build_run_config(self) -> RunConfig:
        """Point the runner at real models, or at the offline test double."""
        if self.settings.live:
            set_tracing_disabled(True)
            return RunConfig(
                workflow_name=f"Athena Research {self.run_id}",
                tracing_disabled=True,
            )

        from .mock.model import MockModelProvider

        set_tracing_disabled(True)
        log_event(
            logger,
            "running in OFFLINE mode - model calls are fixtures, tools are real",
            run_id=self.run_id,
        )
        return RunConfig(
            model_provider=MockModelProvider(),
            workflow_name=f"Athena Research {self.run_id} (offline)",
            tracing_disabled=True,
        )

    # -- the core run primitive -------------------------------------------

    async def _run_agent(
        self,
        agent: Agent[ResearchContext],
        prompt: str,
        ctx: ResearchContext,
        *,
        use_session: bool = True,
        max_turns: int | None = None,
    ) -> RunResult:
        """Run one agent, resolving any approval interruptions before returning.

        Approval handling is here rather than at the call sites because any
        agent may acquire a gated tool later, and a gate that only some call
        paths honour is not a gate.
        """
        result = await Runner.run(
            agent,
            prompt,
            context=ctx,
            hooks=self.hooks,
            run_config=self.run_config,
            session=self.session if use_session else None,
            max_turns=max_turns or self.settings.max_turns,
        )

        # The SDK suspends the run and reports pending approvals rather than
        # calling the tool. Resolve them, then resume from the saved state.
        while getattr(result, "interruptions", None):
            state = result.to_state()

            # Checkpoint *before* asking, so a user who walks away mid-decision
            # leaves behind a resumable run rather than a lost one.
            self._persist_state(state)
            self._persist_context(ctx)

            for item in result.interruptions:
                try:
                    approved, note = await self.approval_handler(item)
                except ApprovalDeferred:
                    log_event(
                        logger,
                        "approval deferred; run checkpointed",
                        tool=item.name,
                        checkpoint=str(self.state_checkpoint_path()),
                    )
                    raise
                arguments = str(getattr(item, "arguments", ""))[:500]
                if approved:
                    state.approve(item)
                    ctx.record_approval(item.name, arguments, "approved", note)
                    log_event(logger, "approval granted", tool=item.name, note=note)
                else:
                    state.reject(item, rejection_message=note or "Rejected by human.")
                    ctx.record_approval(item.name, arguments, "rejected", note)
                    log_event(logger, "approval denied", tool=item.name, note=note)

            result = await Runner.run(
                agent,
                state,
                context=ctx,
                hooks=self.hooks,
                run_config=self.run_config,
                session=self.session if use_session else None,
                max_turns=max_turns or self.settings.max_turns,
            )

        return result

    @property
    def _checkpoint_dir(self) -> Path:
        """Where run checkpoints live.

        Derived from the configured session database rather than assuming
        ``data/sessions``: those can be pointed elsewhere, and hardcoding the
        path made checkpoint writes fail silently when they were.
        """
        directory = self.settings.session_db.parent
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def state_checkpoint_path(self) -> Path:
        return self._checkpoint_dir / f"{self.run_id}_state.json"

    def context_checkpoint_path(self) -> Path:
        return self._checkpoint_dir / f"{self.run_id}_context.json"

    def _persist_context(self, ctx: ResearchContext) -> None:
        """Persist working memory beside the run state, so a resume in another
        process gets back the corpus and artefacts, not an empty context."""
        try:
            self.context_checkpoint_path().write_text(
                json.dumps(ctx.to_dict(), indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            log_event(logger, f"could not checkpoint context: {exc}")

    def _persist_state(self, state: Any) -> None:
        """Checkpoint a suspended run so a pending approval can outlive the process.

        ``RunState.to_json()`` returns a dict, not a string, so it has to be
        encoded before writing. Getting this wrong fails silently inside the
        guard below, which is exactly how it went unnoticed the first time.
        """
        try:
            payload = state.to_json() if hasattr(state, "to_json") else {}
            self.state_checkpoint_path().write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            log_event(
                logger,
                "run state checkpointed",
                path=str(self.state_checkpoint_path()),
                pending_approvals=len(getattr(state, "get_interruptions", lambda: [])()),
            )
        except Exception as exc:  # checkpointing must never break the run
            log_event(logger, f"could not checkpoint run state: {exc}")

    async def _stage(
        self,
        stage: Stage,
        agent: Agent[ResearchContext],
        prompt: str,
        ctx: ResearchContext,
        output_model: type,
        **kwargs: Any,
    ) -> Any | None:
        """Run one pipeline stage with uniform timing, logging and error capture."""
        ctx.stage = stage
        started = datetime.now(timezone.utc)
        log_event(logger, f"stage start: {stage.value}", stage=stage.value, agent=agent.name)

        spent = ctx.costs.estimated_usd()
        cap = self.settings.max_run_usd
        if self.settings.live and cap > 0 and spent >= cap:
            return self._fail_stage(
                stage, agent, ctx, started,
                f"budget reached: ${spent:.2f} spent of the ${cap:.2f} run cap; "
                f"{stage.value} was not started",
            )

        if ctx.source_mode == "documents":
            agent = self._document_agent(agent)
            # Each stage's prompt carries the full source text; replaying earlier
            # stages' transcripts would resend it once per prior stage.
            kwargs["use_session"] = False

        attempt_prompt = prompt
        for attempt in range(_MAX_GUARDRAIL_RETRIES + 1):
          try:
            result = await self._run_agent(agent, attempt_prompt, ctx, **kwargs)
            output = result.final_output_as(output_model)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            self.stages.append(
                StageResult(stage, agent.name, True, f"produced {output_model.__name__}", elapsed)
            )
            log_event(
                logger,
                f"stage ok: {stage.value}",
                stage=stage.value,
                agent=agent.name,
                duration_s=round(elapsed, 2),
            )
            return output

          except InputGuardrailTripwireTriggered as exc:
            message = f"Input guardrail blocked the run: {exc}"
            return self._fail_stage(stage, agent, ctx, started, message)
          except OutputGuardrailTripwireTriggered as exc:
            # Naming the offending ids and re-asking recovers the stage; a
            # single hallucinated id used to destroy the whole artefact and
            # cascade into "nothing produced" downstream.
            fabricated = []
            info = getattr(exc.guardrail_result, "output_info", None)
            if isinstance(info, dict):
                fabricated = info.get("fabricated") or []
            if attempt < _MAX_GUARDRAIL_RETRIES:
                log_event(
                    logger,
                    "citation guardrail tripped; re-asking",
                    stage=stage.value,
                    fabricated=fabricated,
                )
                attempt_prompt = (
                    f"{prompt}{NL}{NL}## Correction required{NL}"
                    f"Your previous attempt cited ids that are not in the corpus "
                    f"and was rejected: {fabricated}. Do not use them again. "
                    f"Cite only from this list, copying each id character for "
                    f"character:{NL}{sorted(ctx.corpus)}"
                )
                continue
            message = f"Output guardrail rejected {agent.name}'s output: {exc}"
            return self._fail_stage(stage, agent, ctx, started, message)
          except MaxTurnsExceeded as exc:
            message = f"{agent.name} exceeded the turn budget: {exc}"
            return self._fail_stage(stage, agent, ctx, started, message)
          except ApprovalDeferred:
            # Not a failure: the human chose to decide later. The run is already
            # checkpointed, so propagate and let the caller stop cleanly rather
            # than recording a bogus stage error.
            raise
          except AthenaError as exc:
            return self._fail_stage(stage, agent, ctx, started, f"{type(exc).__name__}: {exc}")
          except Exception as exc:  # unexpected: log loudly, but keep the run alive
            return self._fail_stage(
                stage, agent, ctx, started, f"unexpected {type(exc).__name__}: {exc}"
            )
        return None

    def _fail_stage(
        self,
        stage: Stage,
        agent: Agent[ResearchContext],
        ctx: ResearchContext,
        started: datetime,
        message: str,
    ) -> None:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        ctx.record_error(message)
        self.stages.append(StageResult(stage, agent.name, False, message, elapsed))
        logger.error(message, extra={"athena_extra": {"stage": stage.value, "agent": agent.name}})
        return None

    # -- the study --------------------------------------------------------

    async def run_study(
        self,
        question: str,
        *,
        attachments: list[str] | None = None,
        source_mode: str = "web",
    ) -> ResearchContext:
        """Execute all eight stages and return the populated context.

        ``source_mode="documents"`` confines the study to the attached PDFs:
        they are ingested directly (no model call), web search is disabled, and
        every stage reads their full text rather than retrieved fragments.
        """
        ctx = ResearchContext(
            run_id=self.run_id, session_id=self.session_id, research_question=question
        )
        if source_mode == "documents":
            return await self._run_document_study(ctx, question, attachments or [])

        if not self.settings.live:
            from .mock.model import set_active_context

            set_active_context(ctx)

        log_event(logger, "study start", question=question, session=self.session_id)

        # -- 1. Plan ------------------------------------------------------
        plan = await self._stage(
            Stage.PLANNING,
            self.team.principal_investigator,
            f"Plan a research programme for this question:\n\n{question}",
            ctx,
            ResearchPlan,
        )
        ctx.plan = plan
        effective_question = plan.research_question if plan else question

        # -- 2. Literature -------------------------------------------------
        attachment_note = ""
        if attachments:
            listed = "\n".join(f"- {path}" for path in attachments)
            attachment_note = (
                f"\n\nThe user attached these files. Ingest the PDFs with "
                f"`ingest_pdf`, analyse images with `describe_figure`, and "
                f"transcribe audio with `transcribe_audio` before searching:\n{listed}"
            )
            for path in attachments:
                ctx.attachments.append({"kind": "pending", "path": path})

        keywords = ", ".join(plan.search_keywords) if plan else effective_question
        literature = await self._stage(
            Stage.LITERATURE,
            self.team.literature_review,
            f"Build the evidence corpus for: {effective_question}\n\n"
            f"Suggested search terms: {keywords}\n"
            f"Run several distinct searches across all three sources."
            f"{attachment_note}",
            ctx,
            LiteratureReview,
        )
        ctx.literature = literature

        if not ctx.corpus:
            log_event(logger, "aborting: corpus is empty after literature stage")
            ctx.record_error("No papers retrieved; downstream stages cannot be grounded.")
            return ctx

        # -- 3. Synthesis --------------------------------------------------
        sub_questions = "\n".join(f"- {q}" for q in plan.sub_questions) if plan else ""
        ctx.synthesis = await self._stage(
            Stage.SYNTHESIS,
            self.team.synthesist,
            f"Synthesise the evidence on: {effective_question}\n\n"
            f"Address each sub-question with its own retrieval:\n{sub_questions}"
            + self._citable_ids(ctx),
            ctx,
            KnowledgeSynthesis,
        )

        # -- 4. Dedicated gap analysis --------------------------------------
        synthesis_digest = (
            ctx.synthesis.model_dump_json(indent=2)[:3000] if ctx.synthesis else "(none)"
        )
        ctx.gap_analysis = await self._stage(
            Stage.GAP_ANALYSIS,
            self.team.gap_analyst,
            f"Identify and prioritise the research gaps for: {effective_question}\n\n"
            f"Evidence synthesis:\n{synthesis_digest}\n\n"
            f"Probe the corpus with the field's own terminology before calling "
            f"anything a gap."
            + self._citable_ids(ctx),
            ctx,
            GapAnalysis,
        )

        # -- 5. Hypotheses ---------------------------------------------------
        if ctx.gap_analysis:
            gaps = "\n".join(
                f"- [{g.gap_id}] ({g.severity.value}/{g.gap_type}, tractability "
                f"{g.tractability}/10) {g.title}: {g.description}"
                for g in ctx.gap_analysis.gaps
            )
            gaps += f"\n\nPriority gap: {ctx.gap_analysis.priority_gap_id}"
        elif ctx.synthesis:
            gaps = "\n".join(f"- {g.gap}: {g.why_it_matters}" for g in ctx.synthesis.gaps)
        else:
            gaps = ""

        ctx.hypotheses = await self._stage(
            Stage.HYPOTHESIS,
            self.team.hypothesis_generator,
            f"Generate hypotheses for: {effective_question}\n\n"
            f"Identified gaps:\n{gaps or '(none recorded - derive your own from the corpus)'}"
            + self._citable_ids(ctx),
            ctx,
            HypothesisSet,
        )

        # -- 6. Experiment design ------------------------------------------
        if ctx.hypotheses:
            chosen = next(
                (
                    h
                    for h in ctx.hypotheses.hypotheses
                    if h.hypothesis_id == ctx.hypotheses.recommended_hypothesis_id
                ),
                ctx.hypotheses.hypotheses[0] if ctx.hypotheses.hypotheses else None,
            )
            if chosen:
                ctx.experiment = await self._stage(
                    Stage.EXPERIMENT,
                    self.team.experiment_planner,
                    f"Design an experiment to test hypothesis {chosen.hypothesis_id}:\n\n"
                    f"Statement: {chosen.statement}\n"
                    f"Null: {chosen.null_hypothesis}\n"
                    f"IVs: {', '.join(chosen.independent_variables)}\n"
                    f"DVs: {', '.join(chosen.dependent_variables)}\n\n"
                    f"Compute the sample size with the power analysis tool.",
                    ctx,
                    ExperimentDesign,
                )

        # -- 7. Citations --------------------------------------------------
        cited_ids = sorted(ctx.corpus.keys())
        ctx.citations = await self._stage(
            Stage.CITATIONS,
            self.team.citation_manager,
            f"Produce a verified APA bibliography for this study.\n\n"
            f"Corpus ids to cite ({len(cited_ids)}): {cited_ids}\n\n"
            f"Verify every id before including it."
            + self._citable_ids(ctx),
            ctx,
            CitationBundle,
        )

        # -- 8. Report ----------------------------------------------------
        ctx.report = await self._stage(
            Stage.REPORT,
            self.team.scientific_writer,
            self._build_report_brief(ctx, effective_question),
            ctx,
            ResearchReport,
            # The brief is assembled from the context on purpose, so replaying
            # every earlier stage's transcript on top of it adds nothing and
            # pushed the request past the account's per-minute token ceiling.
            use_session=False,
        )

        log_event(logger, "study complete", **ctx.summary())
        return ctx

    # -- document-grounded study -------------------------------------------

    def _document_agent(self, agent: Agent[ResearchContext]) -> Agent[ResearchContext]:
        """The same agent, minus tools that would reach outside the sources."""
        tools = [t for t in agent.tools if t.name not in _DOCUMENT_MODE_BLOCKED_TOOLS]
        return agent if len(tools) == len(agent.tools) else replace(agent, tools=tools)

    def _source_block(self, ctx: ResearchContext) -> str:
        """Evidence rules followed by the full text of every ingested source."""
        ids = sorted(ctx.source_texts)
        parts = [DOCUMENT_MODE_RULES.replace("{paper_id}", ids[0] if ids else "upload:...")]
        for pid in ids:
            paper = ctx.corpus.get(pid)
            byline = ""
            if paper is not None:
                who = ", ".join(paper.authors) or "authors not parsed"
                when = paper.year or "n.d."
                byline = f"{paper.title} - {who} ({when}), {paper.venue}"
            parts.append(
                f"## Source document [{pid}]\n{byline}\n"
                f"<<<SOURCE TEXT START>>>\n{ctx.source_texts[pid]}\n<<<SOURCE TEXT END>>>"
            )
        return "\n\n" + "\n\n".join(parts)

    async def _run_document_study(
        self, ctx: ResearchContext, question: str, attachments: list[str]
    ) -> ResearchContext:
        """The eight stages, grounded in the user's documents only."""
        from .tools.documents import ingest_pdf_file

        ctx.source_mode = "documents"
        if not self.settings.live:
            from .mock.model import set_active_context

            set_active_context(ctx)
        log_event(
            logger, "study start", question=question, session=self.session_id,
            source_mode="documents",
        )

        # -- Ingest: deterministic, free, and never skipped ---------------------
        for path in attachments:
            if not path.lower().endswith(".pdf"):
                ctx.record_error(
                    f"{Path(path).name}: only PDFs are used in document-grounded mode"
                )
                continue
            try:
                paper, chunks, body = await ingest_pdf_file(ctx, path)
                log_event(
                    logger, "source ingested", paper_id=paper.paper_id,
                    chars=len(body), chunks=chunks,
                )
            except AthenaError as exc:
                ctx.record_error(f"could not read {Path(path).name}: {exc}")

        if not ctx.source_texts:
            ctx.record_error(
                "Document-grounded mode needs at least one readable PDF; none was ingested."
            )
            return ctx

        sources = self._source_block(ctx)
        ids = sorted(ctx.source_texts)

        # -- 1. Plan --------------------------------------------------------
        ctx.plan = await self._stage(
            Stage.PLANNING,
            self.team.principal_investigator,
            f"Plan a study of this research question:\n\n{question}\n\n"
            f"Scope the plan to what the source document(s) below can answer. Make "
            f"the sub-questions specific to the source's methods, datasets, metrics "
            f"and reported results. Search keywords should be the source's own "
            f"terminology."
            + sources,
            ctx,
            ResearchPlan,
        )

        # -- 2. Literature: the corpus is exactly the supplied sources -------
        papers = [ctx.corpus[pid] for pid in ids if pid in ctx.corpus]
        pages = sum(int(a.get("pages", 0)) for a in ctx.attachments if a.get("kind") == "pdf")
        chars = sum(len(t) for t in ctx.source_texts.values())
        ctx.literature = LiteratureReview(
            query_terms_used=[],
            papers=papers,
            total_found=len(papers),
            coverage_assessment=(
                f"Document-grounded study of {len(papers)} user-supplied source(s) "
                f"({pages} pages, {chars:,} characters of full text, all of it read "
                f"by every stage). External literature search was disabled by design, "
                f"so the analysis reflects the source's own evidence and framing, "
                f"including how it characterises prior work."
            ),
            notable_gaps=[
                "No independent replication or third-party evaluation of the source's "
                "results was consulted (single-source scope)."
            ],
        )
        self.stages.append(
            StageResult(Stage.LITERATURE, "document ingestion", True,
                        f"ingested {len(papers)} source(s)", 0.0)
        )

        # -- 3. Synthesis ---------------------------------------------------
        plan = ctx.plan
        sub_questions = "\n".join(f"- {q}" for q in plan.sub_questions) if plan else ""
        plan_note = f"\nPlanner's interpretation: {plan.interpretation}\n" if plan else ""
        ctx.synthesis = await self._stage(
            Stage.SYNTHESIS,
            self.team.synthesist,
            f"Synthesise what the source establishes about:\n{question}\n{plan_note}\n"
            f"Address each sub-question with specific evidence, including the exact "
            f"numbers the source reports:\n{sub_questions}\n\n"
            f"Key claims should be precise and quantitative. Mark a claim contested "
            f"when the source's own results cut both ways (e.g. one metric or "
            f"condition improves while another degrades)."
            + sources,
            ctx,
            KnowledgeSynthesis,
        )

        # -- 4. Gaps ----------------------------------------------------------
        synthesis_digest = (
            ctx.synthesis.model_dump_json(indent=2) if ctx.synthesis else "(none)"
        )
        ctx.gap_analysis = await self._stage(
            Stage.GAP_ANALYSIS,
            self.team.gap_analyst,
            f"Identify and prioritise the research gaps for:\n{question}\n\n"
            f"A gap here is something the source leaves unestablished: limitations it "
            f"states, conditions it did not test, results that are mixed or not "
            f"statistically separable, and threats to generalisation.\n\n"
            f"Evidence synthesis:\n{synthesis_digest}"
            + sources,
            ctx,
            GapAnalysis,
        )

        # -- 5. Hypotheses ----------------------------------------------------
        gaps = ""
        if ctx.gap_analysis:
            gaps = "\n".join(
                f"- [{g.gap_id}] ({g.severity.value}/{g.gap_type}, tractability "
                f"{g.tractability}/10) {g.title}: {g.description}"
                for g in ctx.gap_analysis.gaps
            ) + f"\n\nPriority gap: {ctx.gap_analysis.priority_gap_id}"
        ctx.hypotheses = await self._stage(
            Stage.HYPOTHESIS,
            self.team.hypothesis_generator,
            f"Generate follow-up hypotheses for:\n{question}\n\n"
            f"Identified gaps:\n{gaps or '(none recorded - derive them from the source)'}\n\n"
            f"Each hypothesis must be testable with the datasets, baselines and metrics "
            f"the source describes, and its rationale must point to specific results."
            + sources,
            ctx,
            HypothesisSet,
        )

        # -- 6. Experiment ----------------------------------------------------
        if ctx.hypotheses and ctx.hypotheses.hypotheses:
            chosen = next(
                (h for h in ctx.hypotheses.hypotheses
                 if h.hypothesis_id == ctx.hypotheses.recommended_hypothesis_id),
                ctx.hypotheses.hypotheses[0],
            )
            ctx.experiment = await self._stage(
                Stage.EXPERIMENT,
                self.team.experiment_planner,
                f"Design an experiment to test hypothesis {chosen.hypothesis_id}:\n\n"
                f"Statement: {chosen.statement}\n"
                f"Null: {chosen.null_hypothesis}\n"
                f"IVs: {', '.join(chosen.independent_variables)}\n"
                f"DVs: {', '.join(chosen.dependent_variables)}\n\n"
                f"Reuse the source's datasets, baselines and evaluation protocol where "
                f"appropriate. Base the effect-size assumption on differences the source "
                f"actually reports, and compute the sample size with the power analysis "
                f"tool."
                + sources,
                ctx,
                ExperimentDesign,
            )

        # -- 7. Citations -----------------------------------------------------
        ctx.citations = await self._stage(
            Stage.CITATIONS,
            self.team.citation_manager,
            f"Produce a verified APA bibliography for this study.\n\n"
            f"Corpus ids to cite ({len(ids)}): {ids}\n\n"
            f"Verify every id once, then format them."
            + self._citable_ids(ctx),
            ctx,
            CitationBundle,
        )

        # -- 8. Report --------------------------------------------------------
        ctx.report = await self._stage(
            Stage.REPORT,
            self.team.scientific_writer,
            self._build_report_brief(ctx, question),
            ctx,
            ResearchReport,
            use_session=False,
        )
        log_event(logger, "study complete", **ctx.summary())
        return ctx

    # -- report brief ------------------------------------------------------

    def _citable_ids(self, ctx: ResearchContext) -> str:
        """The corpus ids an agent is allowed to cite, for its prompt.

        Agents that were never shown the list invented plausible-looking arXiv
        ids, which the integrity guardrail then rejected.
        """
        if not ctx.corpus:
            return ""
        return (
            f"\n\n## Citable paper_ids (the complete list)\n"
            f"Cite only these, copied character for character. Any other id - "
            f"including one you recall from elsewhere - is treated as "
            f"fabricated and the output is rejected:\n{sorted(ctx.corpus)}"
        )

    def _build_report_brief(self, ctx: ResearchContext, question: str) -> str:
        """Assemble the writer's inputs from the context, not the transcript."""
        documents = ctx.source_mode == "documents" and bool(ctx.source_texts)
        # A truncated JSON digest silently drops the tail of an artefact, so the
        # document-grounded brief, which has a bigger budget, passes them whole.
        cap = 12000 if documents else 3000
        parts = [f"Write the final research dossier for: {question}", ""]
        if documents:
            parts.append(DOCUMENT_REPORT_STRUCTURE)

        if ctx.literature:
            parts.append(f"## Corpus\n{len(ctx.corpus)} papers. "
                         f"Coverage: {ctx.literature.coverage_assessment}")
        if ctx.synthesis:
            parts.append(f"## Synthesis\n{ctx.synthesis.model_dump_json(indent=2)[:cap]}")
        if ctx.gap_analysis:
            parts.append(f"## Gap analysis\n{ctx.gap_analysis.model_dump_json(indent=2)[:cap]}")
        if ctx.hypotheses:
            parts.append(f"## Hypotheses\n{ctx.hypotheses.model_dump_json(indent=2)[:cap]}")
        if ctx.experiment:
            parts.append(f"## Experiment design\n{ctx.experiment.model_dump_json(indent=2)[:cap]}")
        if ctx.citations:
            parts.append(
                "## Verified bibliography (use this verbatim for the references "
                f"field; do not add to it)\n{ctx.citations.reference_list_markdown[:3000]}"
            )
        if ctx.corpus:
            parts.append(self._citable_ids(ctx).strip())
        if ctx.errors:
            parts.append("## Run errors (disclose these in Limitations)\n"
                         + "\n".join(f"- {e}" for e in ctx.errors[:10]))

        parts.append(
            "\nBuild the references list from the corpus. Store the durable "
            "insight with `remember_finding`, then save the dossier with "
            "`save_research_artifact` - once. Then return the ResearchReport."
        )
        brief = "\n\n".join(parts)
        if documents:
            brief += self._source_block(ctx)
        return brief

    # -- resumption --------------------------------------------------------

    def load_checkpointed_context(self) -> ResearchContext | None:
        """Rebuild the working memory saved alongside a suspended run."""
        path = self.context_checkpoint_path()
        if not path.exists():
            return None
        return ResearchContext.from_dict(json.loads(path.read_text(encoding="utf-8")))

    async def resume_from_checkpoint(
        self,
        agent: Agent[ResearchContext],
        ctx: ResearchContext | None = None,
        *,
        approve: bool = True,
        note: str = "",
    ) -> RunResult | None:
        """Resume a run that was checkpointed while awaiting human approval.

        This is what makes the approval gate usable in a real interface: the
        decision does not have to happen inside the lifetime of the process
        that asked for it. The live ``ResearchContext`` is passed back in via
        ``context_override``, so the restored run keeps the corpus and artefacts
        it had built up rather than starting from a bare deserialised copy.
        """
        path = self.state_checkpoint_path()
        if not path.exists():
            log_event(logger, "no checkpoint to resume", path=str(path))
            return None

        if ctx is None:
            ctx = self.load_checkpointed_context()
            if ctx is None:
                log_event(logger, "state checkpoint found but context checkpoint missing")
                return None

        # Claim the checkpoint before doing anything expensive. A second resume
        # of the same checkpoint - a double click, a UI rerun - used to replay
        # the whole writer turn again, each time paying for it and saving
        # another copy of the report. The rename is atomic, so exactly one
        # caller wins; the file is restored if the resume fails, so it can be
        # retried.
        claimed = path.with_suffix(".resuming.json")
        try:
            path.replace(claimed)
        except OSError:
            log_event(logger, "checkpoint already being resumed; ignoring duplicate")
            return None

        if not self.settings.live:
            from .mock.model import set_active_context

            set_active_context(ctx)

        if ctx.source_mode == "documents":
            agent = self._document_agent(agent)

        payload = json.loads(claimed.read_text(encoding="utf-8"))
        state = await RunState.from_json(
            agent, payload, context_override=RunContextWrapper(ctx)
        )

        pending = state.get_interruptions()
        log_event(logger, "resuming from checkpoint", pending_approvals=len(pending))
        for item in pending:
            arguments = str(getattr(item, "arguments", ""))[:500]
            if approve:
                state.approve(item)
                ctx.record_approval(item.name, arguments, "approved", note or "resumed")
            else:
                state.reject(item, rejection_message=note or "Rejected by human.")
                ctx.record_approval(item.name, arguments, "rejected", note or "resumed")

        # The agent may reach the gated tool again after resuming. The human has
        # already ruled on it, so apply the same decision rather than suspending
        # a second time and handing back a run with no output.
        try:
            result = await Runner.run(
                agent,
                state,
                context=ctx,
                hooks=self.hooks,
                run_config=self.run_config,
                max_turns=self.settings.max_turns,
            )
        except Exception as exc:
            # The caller here is a UI click, not a pipeline stage, so there is no
            # _stage() wrapper to record this. Fail into the context rather than
            # throwing a traceback at the user - and hand the checkpoint back, so
            # a transient failure (a dropped connection) can be retried.
            ctx.record_error(f"resume failed: {type(exc).__name__}: {exc}")
            log_event(logger, "resume failed", error=str(exc)[:300])
            if claimed.exists() and not path.exists():
                claimed.replace(path)
            return None
        for _ in range(_MAX_RESUME_REGATES):
            if not getattr(result, "interruptions", None):
                break
            state = result.to_state()
            for item in result.interruptions:
                arguments = str(getattr(item, "arguments", ""))[:500]
                if approve:
                    state.approve(item)
                    ctx.record_approval(item.name, arguments, "approved", note or "resumed")
                else:
                    state.reject(item, rejection_message=note or "Rejected by human.")
                    ctx.record_approval(item.name, arguments, "rejected", note or "resumed")
            log_event(logger, "re-applying resumed approval", tool=item.name)
            try:
                result = await Runner.run(
                    agent,
                    state,
                    context=ctx,
                    hooks=self.hooks,
                    run_config=self.run_config,
                    max_turns=self.settings.max_turns,
                )
            except Exception as exc:
                ctx.record_error(f"resume failed: {type(exc).__name__}: {exc}")
                log_event(logger, "resume failed", error=str(exc)[:300])
                return None

        # The checkpoint has been spent; leaving it would only invite a replay.
        claimed.unlink(missing_ok=True)

        # The report is the point of the resume; put it where every reader of
        # the context looks for it, instead of leaving it on the result object.
        output = getattr(result, "final_output", None)
        if isinstance(output, ResearchReport):
            ctx.report = output
            log_event(logger, "study complete", **ctx.summary())
        return result

    # -- reporting ---------------------------------------------------------

    def run_manifest(self, ctx: ResearchContext) -> dict[str, Any]:
        """Everything needed to reconstruct what this run did."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "mode": "live" if self.settings.live else "offline",
            "summary": ctx.summary(),
            "stages": [
                {
                    "stage": s.stage.value,
                    "agent": s.agent,
                    "ok": s.ok,
                    "detail": s.detail[:300],
                    "duration_s": round(s.duration_s, 2),
                }
                for s in self.stages
            ],
            "handoffs": self.hooks.handoff_chain(),
            "tool_usage": self.hooks.tool_usage(),
            "approvals": [a.__dict__ for a in ctx.approvals],
            "errors": ctx.errors,
        }

    def save_manifest(self, ctx: ResearchContext) -> Path:
        path = self.settings.artifacts_dir / f"{self.run_id}_manifest.json"
        path.write_text(json.dumps(self.run_manifest(ctx), indent=2), encoding="utf-8")
        return path
