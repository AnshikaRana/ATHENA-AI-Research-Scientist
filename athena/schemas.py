"""Pydantic models describing every structured hand-off payload in the system.

Every agent in Athena returns a typed object rather than free text. That gives
us three things the rubric cares about: machine-checkable hand-offs between
agents, deterministic rendering in the UI, and durable artefacts on disk.

Note on strict mode: the runtime's ``AgentOutputSchema`` closes every object
with ``additionalProperties: false`` and marks every field required, mirroring
the *strict* JSON-schema mode structured outputs run under. Every model below
therefore declares all fields as required and uses explicit nested models
instead of free-form ``dict`` values.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    """The phases of the research pipeline, used for logging and the UI."""

    PLANNING = "planning"
    LITERATURE = "literature"
    SYNTHESIS = "synthesis"
    GAP_ANALYSIS = "gap_analysis"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    INTERPRETATION = "interpretation"
    REVIEW = "review"
    CITATIONS = "citations"
    REPORT = "report"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Stage 1 - Planning (Principal Investigator)
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """One step of the Principal Investigator's plan."""

    order: int = Field(description="1-based execution order.")
    stage: Stage = Field(description="Which pipeline stage this step belongs to.")
    objective: str = Field(description="What this step must achieve, one sentence.")
    rationale: str = Field(description="Why this step is necessary for the question.")


class ResearchPlan(BaseModel):
    """The explicit plan produced before any research work happens."""

    research_question: str = Field(description="The refined, sharpened question.")
    interpretation: str = Field(
        description="How the PI understands the question, including scope boundaries."
    )
    sub_questions: list[str] = Field(
        description="2-5 decomposed sub-questions that together answer the main one."
    )
    search_keywords: list[str] = Field(
        description="4-10 precise search terms for the literature databases."
    )
    steps: list[PlanStep] = Field(description="The ordered plan of work.")
    success_criteria: list[str] = Field(
        description="Observable conditions that mean the research succeeded."
    )
    out_of_scope: list[str] = Field(
        description="Explicitly excluded topics, to stop the run drifting."
    )


# ---------------------------------------------------------------------------
# Stage 2 - Literature (Literature Review Agent)
# ---------------------------------------------------------------------------


class Paper(BaseModel):
    """A single retrieved work, normalised across arXiv / OpenAlex / Crossref."""

    paper_id: str = Field(description="Stable id, e.g. arXiv id or DOI.")
    title: str
    authors: list[str] = Field(description="Author display names, may be truncated.")
    year: int = Field(description="Publication year; 0 when unknown.")
    venue: str = Field(description="Journal or conference; empty string when unknown.")
    abstract: str = Field(description="Abstract text; empty string when unavailable.")
    url: str = Field(description="Canonical link to the work.")
    source: Literal["arxiv", "openalex", "crossref", "user_upload"] = Field(
        description="Which connector produced this record."
    )
    citation_count: int = Field(description="Citations known to the source; 0 if absent.")


class LiteratureReview(BaseModel):
    """Output of the Literature Review Agent: a curated, deduplicated corpus."""

    query_terms_used: list[str] = Field(description="Terms actually searched.")
    papers: list[Paper] = Field(description="Deduplicated, relevance-ranked papers.")
    total_found: int = Field(description="Count before deduplication and ranking.")
    coverage_assessment: str = Field(
        description="Honest note on how well the corpus covers the question."
    )
    notable_gaps: list[str] = Field(
        description="Topics the search could not find evidence for."
    )


# ---------------------------------------------------------------------------
# Stage 3 - Synthesis (Knowledge Synthesist, RAG-grounded)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """An evidence-backed statement. Every claim must cite retrieved sources."""

    statement: str = Field(description="The claim, stated precisely.")
    supporting_paper_ids: list[str] = Field(
        description="paper_id values from the corpus that support this claim."
    )
    confidence: Confidence
    contested: bool = Field(
        description="True when sources disagree; the disagreement goes in the statement."
    )


class ResearchGap(BaseModel):
    gap: str = Field(description="What the field has not established.")
    why_it_matters: str
    evidence_paper_ids: list[str] = Field(
        description="Papers whose limitations sections point at this gap."
    )


class KnowledgeSynthesis(BaseModel):
    """Grounded synthesis over the retrieved corpus."""

    summary: str = Field(description="A tight narrative of the state of the field.")
    key_claims: list[Claim]
    consensus_points: list[str]
    disagreements: list[str]
    gaps: list[ResearchGap]
    papers_cited: list[str] = Field(description="All paper_ids actually used.")


# ---------------------------------------------------------------------------
# Stage 4 - Hypotheses (Hypothesis Generator)
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    """A falsifiable, testable proposition."""

    hypothesis_id: str = Field(description="Short slug, e.g. 'H1'.")
    statement: str = Field(description="Falsifiable claim in 'If X then Y' form.")
    rationale: str = Field(description="Why the literature makes this plausible.")
    null_hypothesis: str = Field(description="The H0 that an experiment would reject.")
    independent_variables: list[str]
    dependent_variables: list[str]
    novelty_score: int = Field(description="1-10; how new relative to the corpus.")
    feasibility_score: int = Field(description="1-10; testable with normal resources.")
    impact_score: int = Field(description="1-10; how much the field would care.")
    grounded_in_paper_ids: list[str]
    addresses_gap: str = Field(description="Which identified gap this attacks.")


class HypothesisSet(BaseModel):
    hypotheses: list[Hypothesis]
    recommended_hypothesis_id: str = Field(
        description="The single best candidate to design an experiment for."
    )
    selection_rationale: str


# ---------------------------------------------------------------------------
# Stage 5 - Experiment design (Experiment Planner)
# ---------------------------------------------------------------------------


class PowerAnalysis(BaseModel):
    """Sample-size justification, computed with the analysis tool, not guessed."""

    effect_size: float = Field(description="Assumed Cohen's d or equivalent.")
    alpha: float = Field(description="Significance threshold, typically 0.05.")
    power: float = Field(description="Target statistical power, typically 0.8.")
    required_sample_size: int = Field(description="Per-group n from the calculation.")
    justification: str = Field(description="Where the effect-size assumption came from.")


class ExperimentDesign(BaseModel):
    """A protocol detailed enough for another researcher to execute."""

    hypothesis_id: str
    title: str
    methodology: str = Field(description="Design type and full procedure.")
    independent_variables: list[str]
    dependent_variables: list[str]
    controls: list[str] = Field(description="Control conditions and confounds handled.")
    datasets_or_materials: list[str]
    procedure_steps: list[str] = Field(description="Ordered, executable steps.")
    power_analysis: PowerAnalysis
    statistical_tests: list[str] = Field(description="Planned analyses, chosen in advance.")
    threats_to_validity: list[str]
    ethical_considerations: list[str]
    estimated_duration: str
    expected_outcomes: str = Field(description="What supports vs. refutes the hypothesis.")


# ---------------------------------------------------------------------------
# Stage 6 - Peer review (Critic) -> drives the reflection loop
# ---------------------------------------------------------------------------


class ReviewFinding(BaseModel):
    severity: Literal["blocking", "major", "minor"]
    section: str = Field(description="Which artefact/section the issue is in.")
    issue: str
    required_fix: str = Field(description="Concrete instruction for the revision.")


class CritiqueReport(BaseModel):
    """The self-review that gates whether the pipeline may proceed."""

    approved: bool = Field(
        description="True only when no blocking findings remain."
    )
    overall_assessment: str
    rigor_score: int = Field(description="1-10 methodological soundness.")
    novelty_score: int = Field(description="1-10 contribution beyond the corpus.")
    evidence_grounding_score: int = Field(
        description="1-10; are claims actually supported by cited papers."
    )
    findings: list[ReviewFinding]
    revision_instructions: str = Field(
        description="A single actionable brief for the next revision round; "
        "empty string when approved."
    )


# ---------------------------------------------------------------------------
# Stage 7 - Final report (Scientific Writer)
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    heading: str
    body_markdown: str


class ResearchReport(BaseModel):
    """The deliverable dossier."""

    title: str
    research_question: str
    executive_summary: str = Field(description="150-250 words for a non-specialist.")
    sections: list[ReportSection] = Field(
        description="Introduction, Related Work, Gaps, Hypotheses, Proposed "
        "Experiment, Limitations, Future Work."
    )
    key_findings: list[str]
    limitations: list[str]
    references: list[str] = Field(description="Formatted citation strings.")
    confidence: Confidence
    word_count: int


# ---------------------------------------------------------------------------
# Guardrail payloads
# ---------------------------------------------------------------------------


class RelevanceCheck(BaseModel):
    """Input guardrail: is this actually a research question we should spend on?"""

    is_research_question: bool
    reasoning: str
    suggested_rewrite: str = Field(
        description="A better-formed question, or empty string if already fine."
    )


class IntegrityCheck(BaseModel):
    """Output guardrail: catch unsupported claims before they reach the user."""

    has_unsupported_claims: bool
    flagged_statements: list[str]
    reasoning: str

# ---------------------------------------------------------------------------
# Stage 3b - Research gap analysis (Research Gap Analyst)
# ---------------------------------------------------------------------------


class GapSeverity(str, Enum):
    """How consequential a gap is for the field."""

    FOUNDATIONAL = "foundational"   # blocks progress across the whole area
    SIGNIFICANT = "significant"     # limits a major line of work
    INCREMENTAL = "incremental"     # a useful but bounded contribution


class AnalysedGap(BaseModel):
    """A research gap characterised well enough to act on."""

    gap_id: str = Field(description="Short slug, e.g. 'G1'.")
    title: str = Field(description="The gap in one line.")
    description: str = Field(description="What specifically the field has not established.")
    severity: GapSeverity
    gap_type: Literal[
        "empirical", "theoretical", "methodological", "population", "replication"
    ] = Field(description="What kind of missing knowledge this is.")
    evidence_paper_ids: list[str] = Field(
        description="Papers whose limitations or future-work sections show this gap."
    )
    why_unaddressed: str = Field(
        description="Why the field has not closed it: hard, unfashionable, or newly possible."
    )
    tractability: int = Field(description="1-10; how addressable with current methods.")
    potential_impact: str = Field(description="What closing it would enable.")


class GapAnalysis(BaseModel):
    """Output of the dedicated gap-analysis stage."""

    gaps: list[AnalysedGap]
    priority_gap_id: str = Field(description="The gap most worth attacking now.")
    prioritisation_rationale: str = Field(
        description="Why that gap beats the others on impact x tractability."
    )
    saturated_areas: list[str] = Field(
        description="Sub-topics the corpus shows are already well covered."
    )
    corpus_limitations: str = Field(
        description="How the corpus itself constrains this analysis - a gap in the "
        "search is not a gap in the field."
    )


# ---------------------------------------------------------------------------
# Stage 7 - Data interpretation (Data Interpreter)
# ---------------------------------------------------------------------------


class StatisticalResult(BaseModel):
    """One computed result, reported with the context needed to judge it."""

    metric: str = Field(description="What was measured, e.g. 'mean accuracy'.")
    value: float
    comparison: str = Field(
        description="What it is being compared against; empty string if standalone."
    )
    p_value: float = Field(description="p-value if a test was run; -1.0 when not applicable.")
    effect_size: float = Field(description="Effect size if computed; -1.0 when not applicable.")
    interpretation: str = Field(description="What this means, hedged appropriately.")


class DataInterpretation(BaseModel):
    """Analysis of supplied data, or of quantitative findings across the corpus."""

    data_source: str = Field(description="The dataset analysed, or 'corpus findings'.")
    approach: str = Field(description="What analysis was performed and why.")
    results: list[StatisticalResult]
    patterns: list[str] = Field(description="Trends and relationships actually observed.")
    anomalies: list[str] = Field(description="Outliers, inconsistencies, suspicious values.")
    statistical_caveats: list[str] = Field(
        description="Assumptions made, tests whose preconditions may not hold, "
        "multiple-comparison exposure, sample-size limits."
    )
    supports_hypothesis: Literal["supports", "refutes", "inconclusive", "not_applicable"] = (
        Field(description="What the data says about the hypothesis under test.")
    )
    confidence: Confidence
    interpretation_summary: str = Field(description="The bottom line, stated carefully.")


# ---------------------------------------------------------------------------
# Stage 9 - Citations (Citation Manager)
# ---------------------------------------------------------------------------


class FormattedCitation(BaseModel):
    """One reference, rendered in the requested style."""

    paper_id: str = Field(description="Corpus id this citation was built from.")
    in_text: str = Field(description="In-text form, e.g. '(Scarselli et al., 2008)'.")
    full_reference: str = Field(description="Full reference-list entry in the target style.")
    bibtex_key: str = Field(description="Citation key, e.g. 'scarselli2008graph'.")
    bibtex_entry: str = Field(description="Complete BibTeX record.")


class CitationBundle(BaseModel):
    """A verified, style-consistent bibliography."""

    style: Literal["APA", "IEEE", "MLA", "Nature"] = Field(description="Style applied.")
    citations: list[FormattedCitation]
    reference_list_markdown: str = Field(description="The rendered reference list.")
    bibtex_file: str = Field(description="All BibTeX entries concatenated, ready to save.")
    unverified_ids: list[str] = Field(
        description="Cited ids absent from the corpus. Must be empty in a clean run."
    )
    integrity_note: str = Field(
        description="Statement of what was verified and what could not be."
    )
