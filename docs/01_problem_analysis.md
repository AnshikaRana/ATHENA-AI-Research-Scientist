# 1. Problem Analysis

## 1.1 Business context

Scientific output has outgrown the human capacity to read it. Roughly **three
million peer-reviewed articles** are published each year, on top of a preprint
literature growing faster still. The consequences are measurable:

- A systematic literature review takes a research team **6–12 months**, and by
  the time it is published a meaningful fraction of it is out of date.
- Surveyed researchers report spending **~23% of their working time** on
  literature search and reading rather than on original work.
- An estimated **$28 billion per year** of preclinical research spend in the US
  alone is attributed to work that cannot be reproduced — much of it traceable
  to underpowered designs and analyses that were not specified in advance.
- Cross-disciplinary work is disproportionately penalised: a machine-learning
  researcher entering computational biology has no efficient way to find what
  that field already knows.

The pressure is worst in exactly the places where research matters most —
early-stage academic groups, small biotech, and industrial R&D teams that must
decide *what to work on* without a large team to survey the ground first.

There is a second, newer problem. Generative AI has made it trivial to produce
text that *looks* like a literature review. It is equally trivial to produce a
fabricated citation: a well-formed reference, correctly formatted, to a paper
that does not exist. Studies of general-purpose chatbots used for reference
generation have found fabrication rates between 30% and 90% depending on domain.
A research assistant that cannot guarantee its citations is worse than no
assistant, because its errors are expensive to detect and it is trusted by
default.

**So the opportunity is not "summarise papers with an LLM".** That is a solved
and largely commoditised capability. The opportunity is a system that carries a
research question through the *entire* early-stage workflow — search, synthesis,
gap analysis, hypothesis, experimental design, interpretation, citation, write-up —
and that is *architecturally* incapable of citing a paper it never retrieved.

## 1.2 Stakeholders

| Stakeholder | What they need | How this system serves it | What would make them reject it |
|---|---|---|---|
| **Principal Investigator / academic researcher** | To know what is already established and where the genuine openings are, before committing a year of a student's time | Full pipeline from question to dossier, with gaps explicitly classified and prioritised | A single fabricated citation, or a "gap" that is really a gap in the search |
| **PhD student / early-career researcher** | Fast orientation in an unfamiliar sub-field; a defensible experimental design | Literature corpus with citation counts, plus a protocol with a *computed* sample size | Advice that would not survive their supervisor's first question |
| **R&D lead (industry)** | Defensible go/no-go input on a research direction, at low cost | Structured, auditable dossier with an explicit confidence rating and stated limitations | Opaque reasoning; no audit trail for a decision worth millions |
| **Systematic reviewer / evidence synthesist** | Reproducible, transparent search and screening | Full JSONL trace of every query, tool call and agent decision; run manifest | Any step that cannot be reproduced or inspected |
| **Research integrity officer / journal editor** | Assurance that AI-assisted work is honest about what it did | Deterministic citation verification; provenance sidecar on every artefact; unresolved peer-review findings carried into Limitations | A system that hides its own failures |
| **Grant reviewer** | To judge whether a proposal's premise is sound | Gap analysis distinguishing genuine openings from saturated areas | Novelty claims with no basis in the retrieved corpus |
| **The operator running the system** | Cost control and a stop button | Token/cost ledger, turn budgets, human approval before any artefact is written | Unbounded spend; irreversible actions taken without asking |

## 1.3 Problem statement

> Early-stage research requires a sequence of interdependent expert judgements —
> finding the relevant literature, establishing what it collectively shows,
> identifying what it has *not* established, proposing a falsifiable hypothesis,
> designing an adequately powered experiment, interpreting results, and
> documenting all of it with verifiable citations.
>
> A single language model performs every one of these tasks *plausibly* and none
> of them *reliably*. It conflates summarisation with synthesis, presents a gap
> in its search as a gap in the field, asserts sample sizes it did not compute,
> and fabricates references at a rate that makes its output unusable without
> full manual verification — at which point the assistance has cost more time
> than it saved.
>
> **The problem is not capability but accountability**: no single-agent system
> can specialise, cross-examine itself, or prove that a claim it makes is
> grounded in a source it actually retrieved.

### Why a multi-agent architecture is the right answer

This is a genuine design decision, not a formality, and it rests on four
properties a monolithic prompt cannot provide:

1. **Conflicting objectives need separate agents.** Hypothesis generation is
   rewarded for divergence (it runs at temperature 0.8); synthesis and review are
   rewarded for fidelity (temperature 0.2). One agent cannot be optimised for
   both, and asking it to be produces work that is timid *and* unfaithful.
2. **Self-review requires a genuine second party.** An agent asked to critique
   its own output in the same context is systematically lenient. A separate Peer
   Reviewer agent, given the corpus tools and instructed to verify rather than
   trust, produces blocking findings that actually change the artefact.
3. **Grounding must be enforced structurally, not requested politely.** Because
   the Knowledge Synthesist can only cite what the Literature Review Agent put in
   the shared corpus, and because an output guardrail rejects any `paper_id` not
   in that corpus, fabrication is *impossible by construction* rather than
   *discouraged by instruction*.
4. **Different stages need different tools and different costs.** Routing runs
   on a cheap model with two tools; experiment design runs on a reasoning model
   with a statistics sandbox. A single agent holding all twenty tools chooses
   worse and costs more.

## 1.4 Objectives

### Functional objectives

| # | Objective | How it is met | How it is verified |
|---|---|---|---|
| F1 | Conduct a literature review across multiple independent databases | Concurrent search over arXiv, OpenAlex and Crossref, merged and deduplicated by id and normalised title | `test_dedup_by_normalised_title`; live run retrieves and merges from all three |
| F2 | Identify and prioritise research gaps | Dedicated Research Gap Analyst producing typed, severity-rated, tractability-scored gaps | `GapAnalysis` schema; gap stage in pipeline test |
| F3 | Design executable experiments | Experiment Planner required to obtain sample size from the power-analysis tool | `test_power_analysis_matches_cohen_tables` (matches Cohen's published tables) |
| F4 | Analyse and interpret findings | Data Interpreter over user datasets or corpus results, with statistics computed by tools | `test_group_comparison_detects_a_real_effect`; p-values validated against t-tables |
| F5 | Generate verifiable citations | Deterministic formatting from stored metadata in APA/IEEE/MLA/Nature plus BibTeX | `test_citation_styles_render_distinctly`; `test_format_citations_refuses_unknown_ids` |
| F6 | Produce a publication-ready report | Scientific Writer assembling all artefacts, with unresolved criticisms carried into Limitations | `test_full_study_produces_every_artefact` |

### Quality objectives

| # | Objective | Mechanism |
|---|---|---|
| Q1 | **Zero fabricated citations** | Deterministic output guardrail rejects any `paper_id` absent from the retrieved corpus. Not a model judgement. |
| Q2 | **No unaudited claims** | Every agent returns a typed Pydantic object; claims carry explicit `supporting_paper_ids` and a confidence level. |
| Q3 | **Self-correction before delivery** | Four artefacts reviewed concurrently; blocking findings return the work to its producer for up to two revision rounds. |
| Q4 | **Full auditability** | JSONL trace of every agent start/end, handoff, tool call and guardrail decision; run manifest; provenance sidecar per artefact. |
| Q5 | **Honest failure** | A dead database degrades the corpus and is disclosed, rather than failing the run. An empty corpus aborts before synthesis rather than inventing findings. |

### Operational objectives

| # | Objective | Mechanism |
|---|---|---|
| O1 | **Human control of consequential actions** | Writing any artefact requires explicit approval; expensive searches are gated dynamically by cost. |
| O2 | **Bounded cost** | Two-tier model routing, turn budgets, capped reflection rounds, and a running token/cost ledger. |
| O3 | **Resumability** | A deferred approval checkpoints the run state *and* the working memory, so the decision can be made in another process or another day. |
| O4 | **Runs without an API key** | Offline mode substitutes fixtures for model calls only; literature search, RAG, PDF parsing, statistics and citation generation remain real. |
| O5 | **Runs standalone** | Athena depends only on public scholarly APIs (arXiv, OpenAlex, Crossref) and the OpenAI API; no external orchestration infrastructure is required. |

### Explicit non-objectives

Stating these matters, because a system that implies them would be misleading:

- **Athena does not run experiments.** It designs them. No result it reports is empirical.
- **Athena does not replace peer review.** Its Peer Reviewer catches internal inconsistency and ungrounded claims, not domain error.
- **Athena does not read full texts at scale.** It indexes abstracts, plus the full text of PDFs the user supplies. Conclusions drawn from abstracts alone are labelled as such.
- **Athena does not assess whether a hypothesis is *true*.** Novelty and impact scores are judgements about the retrieved corpus, not about reality.
