# 2. Multi-Agent Design

## 2.1 Agent architecture

Athena runs **eight specialised agents** in a deterministic pipeline: the stage
order is fixed by dependency (you cannot synthesise evidence you have not
retrieved), and the judgement inside each stage is the model's.

> **Note (cost-reduced build).** An earlier version had eleven agents including a
> Research Director (LLM routing via handoffs), a Data Interpreter, and a Peer
> Reviewer with a concurrent reflection loop. Those three were removed to cut
> per-run token cost; sections below that describe them are retained for context
> but no longer reflect the running system. Fabricated citations are still
> blocked deterministically by the citation-integrity guardrail.

Six agents map directly onto the roles named in the project brief. Five more
exist because the pipeline does not work without them.

| Brief's role | Agent | Why it exists |
|---|---|---|
| Literature Review | **Literature Review Agent** | Acquires evidence. Explicitly forbidden from interpreting it. |
| Research Gap Analysis | **Research Gap Analyst** | Finds what the field has *not* established, typed and prioritised. |
| Experiment Planning | **Experiment Planner** | Turns a hypothesis into an executable, adequately powered protocol. |
| Data Interpretation | **Data Interpreter** | Analyses datasets or corpus results; all statistics come from tools. |
| Citation Manager | **Citation Manager** | Generates and *verifies* the bibliography. Last line of defence. |
| Scientific Writer | **Scientific Writer** | Assembles the dossier; owns the approval-gated save. |
| *(additional)* | **Research Director** | Routes ad-hoc requests via runtime handoffs. |
| *(additional)* | **Principal Investigator** | Sharpens the question and plans the programme. |
| *(additional)* | **Knowledge Synthesist** | RAG-grounded synthesis — the agent whose output everything downstream depends on. |
| *(additional)* | **Hypothesis Generator** | The one divergent step in an otherwise conservative system. |
| *(additional)* | **Peer Reviewer** | Adversarial self-review; gates progression. |

### The central architectural decision

**Orchestration is deterministic where dependencies are hard, and LLM-driven
where routing is a real choice.**

Research stages have strict dependencies: you cannot synthesise evidence you
have not retrieved, or design an experiment for a hypothesis you have not
selected. Letting a model rediscover that ordering on every run costs tokens,
adds failure modes, and makes the system behave differently each time it is
demonstrated. So `pipeline.py` fixes the *sequence* in code.

What remains genuinely uncertain is which specialist should handle a one-off
request — "find me papers on X", "critique this design", "is this testable?".
There the right agent is not knowable in advance, so the **Research Director**
transfers control with a real `handoff()`.

This is why handoffs appear in consult mode and not in study mode, and that is a
deliberate design position rather than an omission.

## 2.2 Roles of each agent

### 1. Research Director — routing
- **Output**: plain text (it hands off rather than answering)
- **Tools**: `recall_memory`, `remember_finding`
- **Handoffs**: all eight specialists
- **Guardrail**: input relevance screening
- **Model**: fast tier, temperature 0.1 — routing should be boring and repeatable

### 2. Principal Investigator — planning
- **Output**: `ResearchPlan`
- Sharpens a vague question into an answerable one, decomposes it into 2–5
  sub-questions, and defines success criteria *and* explicit out-of-scope
  boundaries — which is what stops a broad question becoming an unbounded search.
- Checks long-term memory first: prior work on an adjacent topic changes what is
  worth doing now.
- **Model**: reasoning tier, temperature 0.3

### 3. Literature Review Agent — evidence acquisition
- **Output**: `LiteratureReview`
- Runs several *distinct* searches varying phenomenon, method, application and
  historical terminology. Queries all three databases because their coverage
  differs: arXiv has preprints, OpenAlex has citation counts and broad indexing,
  Crossref has the publisher record.
- **Constraint**: forbidden from interpreting the papers. Its `coverage_assessment`
  must be candid about a thin corpus, because a downstream agent that knows the
  evidence is weak will hedge correctly and one that does not will not.
- **Model**: fast tier, temperature 0.2

### 4. Knowledge Synthesist — RAG-grounded synthesis
- **Output**: `KnowledgeSynthesis` · **Guardrail**: citation integrity
- Must call `index_corpus`, then `rag_search` **per sub-question**, and may write
  only what the retrieved passages support.
- **The defining constraint**: it is instructed not to use its own parametric
  knowledge, because such a claim is unattributable and undefendable. The
  guardrail enforces this structurally.
- **Model**: reasoning tier, temperature 0.2

### 5. Research Gap Analyst — gap identification
- **Output**: `GapAnalysis` · **Guardrail**: citation integrity
- Classifies each gap as empirical, theoretical, methodological, population or
  replication; rates severity and tractability separately, because a
  foundational gap nobody can attack is worse guidance than a modest one that is
  tractable now.
- **The failure it is built to avoid**: mistaking a hole in the *search* for a
  hole in the *field*. It must re-probe with the field's own terminology before
  declaring a gap, and record residual doubt in `corpus_limitations`.
- **Model**: reasoning tier, temperature 0.4

### 6. Hypothesis Generator — divergence
- **Output**: `HypothesisSet` · **Guardrail**: citation integrity
- Produces 3–5 falsifiable "If X then Y" propositions attacking identified gaps,
  each with an explicit null hypothesis.
- **Runs at temperature 0.8** — the only agent in the system tuned for
  divergence. Conservatism is the failure mode here; the Peer Reviewer supplies
  the corresponding convergence.
- Scores novelty, feasibility and impact, and is explicitly instructed not to
  inflate them: a set where everything scores 8–10 conveys nothing.

### 7. Experiment Planner — protocol design
- **Output**: `ExperimentDesign`
- **Hard constraint**: sample size must come from `statistical_power_analysis`,
  never estimation, and the assumed effect size must be justified from a cited
  paper or as the smallest practically meaningful effect.
- Requires pre-specified statistical tests, named confounds with controls, and a
  refuting outcome — a design with no refuting outcome tests nothing.

### 8. Data Interpreter — analysing findings
- **Output**: `DataInterpretation`
- Analyses a supplied dataset, or the quantitative results reported across the
  corpus when none is supplied.
- **Computes nothing itself.** Every statistic comes from `analyze_dataset`,
  `compare_groups` or the sandbox. Its instructions are mostly restraint:
  correlation is not causation, a p-value is not an effect size, non-significant
  is not "no effect", and "inconclusive" is a legitimate verdict.

### 9. Peer Reviewer — reflection
- **Output**: `CritiqueReport`
- Given the corpus tools so it can *verify* citations rather than trust them.
- Reviews in severity order: fabricated citations (always blocking), overclaiming,
  unfalsifiable hypotheses, unjustified sample sizes, uncontrolled confounds,
  post-hoc-able analyses, novelty inflation.
- `approved=False` with blocking findings returns the artefact to its producer.
- Instructed to use the full 1–10 range, because a reviewer who awards 8s
  regardless is decorative.

### 10. Citation Manager — bibliography
- **Output**: `CitationBundle` · **Guardrail**: citation integrity
- Verifies first, formats second. Explicitly instructed never to write a citation
  itself: *"You know what a plausible reference looks like, which is precisely
  the problem."*
- **Model**: fast tier, temperature 0.1 — formatting should be deterministic

### 11. Scientific Writer — the dossier
- **Output**: `ResearchReport` · **Guardrail**: citation integrity
- Assembles, compresses and connects. May not add findings, and may not drop the
  reviewer's unresolved criticisms because they spoil the narrative.
- Owns `save_research_artifact`, the approval-gated tool.

## 2.3 Agent interaction and handoff flow

### Study mode — deterministic, ten stages

```
   Principal Investigator ──▶ ResearchPlan
             │
   Literature Review Agent ─▶ LiteratureReview      [3 databases in parallel]
             │
     Knowledge Synthesist ──▶ KnowledgeSynthesis    [RAG per sub-question]
             │
     Research Gap Analyst ──▶ GapAnalysis
             │
     Hypothesis Generator ──▶ HypothesisSet
             │
      Experiment Planner ───▶ ExperimentDesign      [power analysis computed]
             │
       Data Interpreter ────▶ DataInterpretation
             │
   ┌─────────┴─────────────────────────────────┐
   │  Peer Reviewer × 4, CONCURRENT            │
   │  synthesis │ gaps │ hypotheses │ design   │
   └─────────┬─────────────────────────────────┘
             │  blocking findings?
             ├── yes ──▶ revise concurrently, re-review (max 2 rounds)
             └── no ───▶ continue
             │
       Citation Manager ────▶ CitationBundle        [verified against corpus]
             │
      Scientific Writer ────▶ ResearchReport
             │
        ⏸ HUMAN APPROVAL ──▶ artefact written / rejected / deferred
```

### Consult mode — LLM-driven handoff

```
   User request ──▶ Research Director ──handoff()──▶ one specialist ──▶ typed output
```

The handoff carries three things worth noting:

- **`on_handoff` callback** — updates `ResearchContext.stage` and writes the
  transfer to the audit log, so a handoff is observable, not implicit.
- **`input_filter=remove_all_tools`** — strips the Director's own tool chatter
  from the transferred history. The specialist needs the request and the routing
  decision, not a replay of memory lookups it cannot act on.
- **Input guardrail** — the relevance screener runs before any spend.

### How agents actually exchange information

**Not through the transcript.** Agents write into the shared `ResearchContext`
and pass each other short references to it.

This matters more than it sounds. A 40-paper corpus serialised into the
conversation would consume tens of thousands of tokens *per agent turn*. Instead
`search_literature` writes full records into `ctx.corpus` and returns a ~300-token
digest. The corpus is available to every agent through `RunContextWrapper`, and
the model never pays for it.

Three memory tiers, deliberately distinct:

| Tier | Implementation | Lifetime | Purpose |
|---|---|---|---|
| **Conversation** | `SQLiteSession` | Session | Follow-up questions know what was said |
| **Working** | `ResearchContext` | One run | Corpus, stage artefacts, cost ledger, approvals |
| **Long-term** | SQLite + embeddings | Permanent | Findings and preferences recalled across sessions |

### The reflection loop

Review is not advisory. `_review_and_revise` fans out four concurrent Peer
Reviewer runs — one per artefact — via `asyncio.gather`. Each runs **without the
shared session**, deliberately: three reviewers writing into one transcript
simultaneously would read each other's half-written turns, which is both a
correctness bug and a source of review contamination.

Artefacts with blocking findings are returned to their producing agent with the
reviewer's brief, revised concurrently, and re-reviewed. After
`max_reflection_rounds` the loop terminates and unresolved criticisms are
recorded as errors — and the Scientific Writer is explicitly required to carry
them into the report's Limitations section rather than quietly dropping them.

## 2.4 Tool integration overview

**Twenty tools across nine groups.** Each agent receives only the subset its role
needs; a Scientific Writer holding a literature search tool would wander off and
search instead of writing.

| Group | Tools | External dependency | Notes |
|---|---|---|---|
| **Literature** | `search_literature`, `commission_deep_sweep`, `get_paper_details`, `list_corpus` | arXiv, OpenAlex, Crossref | Three APIs queried concurrently; one failure degrades rather than aborts |
| **RAG** | `index_corpus`, `rag_search` | — (hashed lexical embedding) | Chunk → embed → cosine retrieval, persisted to disk |
| **Documents** | `ingest_pdf`, `describe_figure`, `transcribe_audio` | pypdf, GPT-4o vision, Whisper | The multi-modal surface |
| **Analysis** | `statistical_power_analysis`, `run_python_analysis`, `analyze_dataset`, `compare_groups` | pandas, numpy | Sandboxed execution; power and p-values computed, never guessed |
| **Citations** | `format_citations`, `verify_citations` | — | Deterministic; four styles plus BibTeX |
| **Memory** | `remember_finding`, `recall_memory` | SQLite + embeddings | Cross-session recall |
| **Artefacts** | `save_research_artifact` | — | **Static approval gate** |
| *(dynamic gate)* | `commission_deep_sweep` | — | **Approval only above a cost threshold** |

### Two approval styles, on purpose

- **Static** — `save_research_artifact` always asks. Writing a research dossier
  under the user's name is consequential and outward-facing.
- **Dynamic** — `commission_deep_sweep` is a function of its arguments: a narrow
  sweep proceeds freely; one requesting more than 30 results stops for
  authorisation. The gate is on *cost*, not on the tool.

### The sandbox

`run_python_analysis` executes model-written code, so it is defended twice:

1. **AST validation before execution** — imports restricted to an allowlist;
   `open`/`exec`/`eval`/`__import__` rejected; dunder attribute access blocked
   (the classic `().__class__.__bases__[0].__subclasses__()` escape).
2. **A separate short-lived process** in isolated mode with a 15-second hard
   timeout, so a runaway snippet cannot take the run down.

Runtime errors are returned to the agent as text rather than raised — a
traceback is information it can use to fix its own code on the next turn.
