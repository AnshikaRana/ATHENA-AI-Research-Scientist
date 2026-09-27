"""The shared run context.

Every agent and tool in a run receives the *same* ``ResearchContext`` object via
the runtime's ``RunContextWrapper``. This is how Athena manages working memory:
rather than stuffing the growing paper corpus into the conversation transcript
(which would blow the context window and cost a fortune), agents write findings
into this object and pass each other short *references* to it.

The context is never sent to the model. Tools read and mutate it locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import (
    CitationBundle,
    CritiqueReport,
    DataInterpretation,
    GapAnalysis,
    ExperimentDesign,
    HypothesisSet,
    KnowledgeSynthesis,
    LiteratureReview,
    Paper,
    ResearchPlan,
    ResearchReport,
    Stage,
)


# Per-million-token USD rates, keyed by model-name prefix. Longest match wins,
# so "gpt-4o-mini" is not billed at "gpt-4o" rates.
_MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass
class CostLedger:
    """Rolling token and request accounting, so a run can't silently overspend."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    api_calls: int = 0
    # Spend accumulated call by call at each call's own model rate. A run mixes
    # a cheap and a strong model, so pricing the token total at one rate is
    # wrong in both directions.
    usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Account one model call at that model's rates."""
        in_rate, out_rate = model_rates(model)
        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.usd += (input_tokens / 1e6) * in_rate + (output_tokens / 1e6) * out_rate

    def estimated_usd(
        self, in_rate: float | None = None, out_rate: float | None = None
    ) -> float:
        """USD spent so far; explicit rates re-price the token totals instead."""
        if in_rate is None and out_rate is None and self.usd:
            return self.usd
        if in_rate is None or out_rate is None:
            from .config import get_settings

            rates = model_rates(get_settings().reasoning_model)
            in_rate = in_rate if in_rate is not None else rates[0]
            out_rate = out_rate if out_rate is not None else rates[1]
        return (self.input_tokens / 1e6) * in_rate + (self.output_tokens / 1e6) * out_rate


def model_rates(model: str) -> tuple[float, float]:
    """Per-million-token (input, output) USD rates; unknown models price high."""
    match = max((p for p in _MODEL_RATES if model.startswith(p)), key=len, default="")
    return _MODEL_RATES.get(match, (2.50, 10.00))


@dataclass
class ApprovalRecord:
    """An audit entry for a human-in-the-loop decision."""

    tool_name: str
    arguments: str
    decision: str  # "approved" | "rejected"
    decided_at: str
    note: str = ""


@dataclass
class ResearchContext:
    """Working memory for a single research run."""

    run_id: str
    session_id: str
    research_question: str

    # --- Accumulated artefacts, one per pipeline stage ---------------------
    plan: ResearchPlan | None = None
    literature: LiteratureReview | None = None
    synthesis: KnowledgeSynthesis | None = None
    gap_analysis: GapAnalysis | None = None
    hypotheses: HypothesisSet | None = None
    experiment: ExperimentDesign | None = None
    interpretation: DataInterpretation | None = None
    critiques: list[CritiqueReport] = field(default_factory=list)
    citations: CitationBundle | None = None
    report: ResearchReport | None = None

    # --- The corpus, keyed by paper_id for O(1) citation resolution -------
    corpus: dict[str, Paper] = field(default_factory=dict)

    # --- Multi-modal inputs supplied by the user --------------------------
    attachments: list[dict[str, Any]] = field(default_factory=list)

    # Full text of each ingested document, keyed by paper_id. In
    # document-grounded mode this - not a handful of retrieved snippets - is
    # the evidence every stage reads.
    source_texts: dict[str, str] = field(default_factory=dict, repr=False)
    # "web" searches the literature APIs; "documents" confines the study to
    # the user's uploads.
    source_mode: str = "web"

    # artifact_type -> path written this run, so a repeated save overwrites.
    saved_artifacts: dict[str, str] = field(default_factory=dict)

    # --- RAG index over the corpus ----------------------------------------
    # Held here rather than in a global registry because it *is* working
    # memory: it lives and dies with the run. Typed loosely to keep this
    # module free of a dependency on the memory package.
    vector_store: Any = field(default=None, repr=False)

    # --- Bookkeeping ------------------------------------------------------
    stage: Stage = Stage.PLANNING
    reflection_round: int = 0
    costs: CostLedger = field(default_factory=CostLedger)
    approvals: list[ApprovalRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Corpus helpers ---------------------------------------------------

    def add_papers(self, papers: list[Paper], *, force: bool = False) -> int:
        """Merge papers into the corpus, deduplicating by id and title.

        ``force`` admits papers past the corpus ceiling. Material the user
        supplied themselves (an ingested PDF) must never be dropped in favour
        of search results.

        Returns the number of genuinely new papers added.
        """
        from .config import get_settings

        ceiling = float("inf") if force else get_settings().max_corpus_papers
        seen_titles = {self._norm(p.title) for p in self.corpus.values()}
        added = 0
        for paper in papers:
            if len(self.corpus) >= ceiling:
                break
            if paper.paper_id in self.corpus:
                continue
            title_key = self._norm(paper.title)
            if title_key and title_key in seen_titles:
                continue
            self.corpus[paper.paper_id] = paper
            seen_titles.add(title_key)
            added += 1
        return added

    @staticmethod
    def _norm(title: str) -> str:
        return "".join(ch for ch in title.lower() if ch.isalnum())

    def resolve_citations(self, paper_ids: list[str]) -> list[Paper]:
        return [self.corpus[pid] for pid in paper_ids if pid in self.corpus]

    def unknown_citations(self, paper_ids: list[str]) -> list[str]:
        """Citation ids that do not exist in the corpus, i.e. likely fabrications."""
        return [pid for pid in paper_ids if pid not in self.corpus]

    # --- Audit ------------------------------------------------------------

    def record_approval(
        self, tool_name: str, arguments: str, decision: str, note: str = ""
    ) -> None:
        self.approvals.append(
            ApprovalRecord(
                tool_name=tool_name,
                arguments=arguments,
                decision=decision,
                decided_at=datetime.now(timezone.utc).isoformat(),
                note=note,
            )
        )

    def record_error(self, message: str) -> None:
        self.errors.append(f"{datetime.now(timezone.utc).isoformat()} {message}")

    # --- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole working memory.

        This is what lets a run survive a process restart. The vector store is
        deliberately excluded: it is already persisted separately under
        ``data/vectors/<run_id>`` and is rebuilt with ``VectorStore.load()``,
        so duplicating its matrices here would bloat every checkpoint.
        """

        def dump(model: Any) -> Any:
            return json.loads(model.model_dump_json()) if model is not None else None

        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "research_question": self.research_question,
            "stage": self.stage.value,
            "reflection_round": self.reflection_round,
            "started_at": self.started_at,
            "plan": dump(self.plan),
            "literature": dump(self.literature),
            "synthesis": dump(self.synthesis),
            "gap_analysis": dump(self.gap_analysis),
            "hypotheses": dump(self.hypotheses),
            "experiment": dump(self.experiment),
            "interpretation": dump(self.interpretation),
            "citations": dump(self.citations),
            "report": dump(self.report),
            "critiques": [dump(c) for c in self.critiques],
            "corpus": {pid: dump(p) for pid, p in self.corpus.items()},
            "attachments": self.attachments,
            "source_texts": self.source_texts,
            "source_mode": self.source_mode,
            "saved_artifacts": self.saved_artifacts,
            "approvals": [a.__dict__ for a in self.approvals],
            "errors": self.errors,
            "costs": self.costs.__dict__,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchContext":
        """Rebuild a context from :meth:`to_dict`, restoring its RAG index."""
        ctx = cls(
            run_id=data["run_id"],
            session_id=data["session_id"],
            research_question=data["research_question"],
        )
        ctx.stage = Stage(data.get("stage", Stage.PLANNING.value))
        ctx.reflection_round = int(data.get("reflection_round", 0))
        ctx.started_at = data.get("started_at", ctx.started_at)

        def load(model_cls: Any, payload: Any) -> Any:
            return model_cls.model_validate(payload) if payload else None

        ctx.plan = load(ResearchPlan, data.get("plan"))
        ctx.literature = load(LiteratureReview, data.get("literature"))
        ctx.synthesis = load(KnowledgeSynthesis, data.get("synthesis"))
        ctx.gap_analysis = load(GapAnalysis, data.get("gap_analysis"))
        ctx.hypotheses = load(HypothesisSet, data.get("hypotheses"))
        ctx.experiment = load(ExperimentDesign, data.get("experiment"))
        ctx.interpretation = load(DataInterpretation, data.get("interpretation"))
        ctx.citations = load(CitationBundle, data.get("citations"))
        ctx.report = load(ResearchReport, data.get("report"))
        ctx.critiques = [
            CritiqueReport.model_validate(c) for c in data.get("critiques", []) if c
        ]
        ctx.corpus = {
            pid: Paper.model_validate(p)
            for pid, p in (data.get("corpus") or {}).items()
            if p
        }
        ctx.attachments = data.get("attachments", [])
        ctx.source_texts = data.get("source_texts", {}) or {}
        ctx.source_mode = data.get("source_mode", "web")
        ctx.saved_artifacts = data.get("saved_artifacts", {}) or {}
        ctx.approvals = [ApprovalRecord(**a) for a in data.get("approvals", [])]
        ctx.errors = data.get("errors", [])
        ctx.costs = CostLedger(**data.get("costs", {}))

        # Reattach the persisted RAG index if one was written for this run.
        from .memory.vector_store import VectorStore

        store = VectorStore(name=ctx.run_id)
        if store.load():
            ctx.vector_store = store
        return ctx

    def summary(self) -> dict[str, Any]:
        """A compact snapshot for logging and the UI."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "stage": self.stage.value,
            "corpus_size": len(self.corpus),
            "indexed_chunks": (
                len(self.vector_store.chunks) if self.vector_store is not None else 0
            ),
            "reflection_round": self.reflection_round,
            "has_plan": self.plan is not None,
            "has_synthesis": self.synthesis is not None,
            "has_gap_analysis": self.gap_analysis is not None,
            "has_experiment": self.experiment is not None,
            "has_interpretation": self.interpretation is not None,
            "has_citations": self.citations is not None,
            "has_report": self.report is not None,
            "approvals": len(self.approvals),
            "errors": len(self.errors),
            "tokens": self.costs.total_tokens,
            "estimated_usd": round(self.costs.estimated_usd(), 4),
        }
