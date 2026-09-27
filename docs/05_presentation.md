---
marp: true
theme: default
paginate: true
---

<!--
Renders as slides with Marp (`marp docs/05_presentation.md --pdf`) or reveal-md,
and reads as a normal document on GitHub.
Speaker notes are in HTML comments under each slide.
-->

# 🔬 Athena
## An AI Research Scientist

**11 agents · 20 tools · 5 external APIs · 10-stage pipeline**

Built on the OpenAI API + a small in-house agent runtime

Kanan Mittal — Multi-Agent Capstone

<!-- Open by naming the one thing that makes this different: it cannot fabricate
a citation. Everything else follows from that design choice. -->

---

## The problem is not capability. It's accountability.

Ask any chatbot for a literature review. You get fluent prose — and references
that **do not exist**.

- Published fabrication rates for LLM citations: **30–90%** depending on domain
- A systematic review takes a team **6–12 months**
- Researchers spend **~23% of working time** on literature search
- **$28B/year** in US preclinical spend attributed to irreproducible work

> An assistant you must fact-check line by line has cost you more time than it
> saved.

<!-- The key framing: this isn't "LLMs are bad at summarising" — they're good at
it. It's that you cannot trust the output without verifying it, and verification
is the expensive part. -->

---

## Why one agent cannot do this

Four properties a monolithic prompt cannot provide:

| | |
|---|---|
| **Conflicting objectives** | Hypothesis generation needs divergence (temp **0.8**). Synthesis and review need fidelity (temp **0.2**). One agent can't be both. |
| **Genuine self-review** | An agent critiquing its own output in the same context is systematically lenient. |
| **Structural grounding** | Separating *who retrieves* from *who cites* is what makes verification possible. |
| **Cost and tool routing** | Routing needs 2 tools and a cheap model. Design needs a statistics sandbox and a reasoning model. |

<!-- This is the slide that justifies the whole architecture. If asked "why not
just one good prompt?", this is the answer. -->

---

## Architecture

```
Principal Investigator ─▶ Literature Review ─▶ Synthesist ─▶ Gap Analyst
      ─▶ Hypothesis Generator ─▶ Experiment Planner ─▶ Data Interpreter
                              │
              ┌───────────────┴────────────────┐
              │  Peer Reviewer × 4  CONCURRENT │──┐ blocking
              └───────────────┬────────────────┘  │ findings
                              │◀──────────────────┘
              Citation Manager ─▶ Scientific Writer ─▶ ⏸ HUMAN APPROVAL
```

**Three memory tiers** · **Deterministic pipeline + LLM handoffs** · **Full audit trail**

<!-- Diagram is generated from live code by scripts/generate_diagram.py, so it
cannot drift from the implementation. Worth mentioning. -->

---

## The eleven agents

| Agent | Output | Distinguishing constraint |
|---|---|---|
| Research Director | *routes* | Real runtime handoffs |
| Principal Investigator | `ResearchPlan` | Defines out-of-scope bounds |
| Literature Review | `LiteratureReview` | **Forbidden from interpreting** |
| Knowledge Synthesist | `KnowledgeSynthesis` | **May not use parametric knowledge** |
| Research Gap Analyst | `GapAnalysis` | Must distinguish search gap from field gap |
| Hypothesis Generator | `HypothesisSet` | The one divergent agent |
| Experiment Planner | `ExperimentDesign` | Sample size **computed, never guessed** |
| Data Interpreter | `DataInterpretation` | **Computes nothing itself** |
| Peer Reviewer | `CritiqueReport` | Verifies citations, doesn't trust them |
| Citation Manager | `CitationBundle` | **Never writes a citation itself** |
| Scientific Writer | `ResearchReport` | May not drop unresolved criticism |

<!-- Every agent returns a typed Pydantic object — that's the structured-output
requirement, and it's what makes handoffs machine-checkable. -->

---

## Orchestration: deterministic *and* LLM-driven

**Deterministic where dependencies are hard.**
You cannot synthesise evidence you have not retrieved. Letting a model
rediscover that ordering every run costs tokens and makes demos non-reproducible.
→ *the sequence is code.*

**LLM handoffs where routing is a genuine choice.**
*"Critique this design"* — which specialist? Not knowable in advance.
→ *`handoff()` with `on_handoff` logging and `input_filter=remove_all_tools`.*

```
Research Director ──handoff()──▶ one of 8 specialists
```

Both, on purpose. This is a design position, not a missing feature.

<!-- If a reviewer asks "where are the handoffs?" — they're in consult mode, and
this slide explains why they are deliberately not in study mode. -->

---

## How fabrication is made impossible

Not a prompt. A **set lookup**.

```python
# athena/guardrails.py
fabricated = sorted(cid for cid in cited if cid not in ctx.corpus)
return GuardrailFunctionOutput(
    output_info=info, tripwire_triggered=bool(fabricated)
)
```

- The Synthesist can only cite what the Literature Review Agent **actually retrieved**
- An output guardrail rejects any `paper_id` absent from the corpus
- The Citation Manager **verifies before formatting**
- Citations are rendered **deterministically from stored metadata**

> Fabrication is impossible by construction, not discouraged by instruction.

<!-- This is the headline technical claim. Demo it live: ask for a citation to a
fake ID and watch the guardrail trip. -->

---

## Reflection: review that changes the artefact

Four Peer Reviewer runs execute **concurrently** via `asyncio.gather`.

```
Round 1:  synthesis ✗  gaps ✗  hypotheses ✗  design ✗   → revise concurrently
Round 2:  synthesis ✓  gaps ✓  hypotheses ✓  design ✓   → proceed
```

- Blocking findings return each artefact **to the agent that produced it**
- Bounded at 2 rounds — then unresolved criticism is **carried into Limitations**,
  not quietly dropped
- Reviewers run **sessionless**: sharing a transcript would let them read each
  other's half-written turns

*Verified by tests for both the revise-then-approve path and the
never-terminates path.*

<!-- The bounded-loop test matters: an unbounded reflection loop is a real
failure mode, and we prove it terminates. -->

---

## Numbers come from tools, not the model

| Claim | Source | Validation |
|---|---|---|
| Sample size | `statistical_power_analysis` | Matches **Cohen's published tables** (d=0.5 → n=64) |
| p-value | Hand-implemented incomplete beta | Matches **standard t-tables** to 4 decimals |
| Group difference | Welch's t-test + Cohen's d | Detects seeded effect, reports magnitude |
| Citations | Deterministic from metadata | 4 styles + BibTeX |

**Sandboxed execution**: AST validation (import allowlist, no `eval`/`open`/dunder
access) → separate isolated process → 15s hard timeout.
*Six escape attempts tested. All blocked.*

<!-- An LLM that estimates a sample size has produced a design that looks
rigorous and isn't. This slide is the answer to "how do you know it's right?" -->

---

## Memory, multi-modal input, and human control

**Three memory tiers** — conversation (`SQLiteSession`), working
(`ResearchContext`), long-term (SQLite + embeddings, cross-session).

**Context efficiency**: agents exchange *references*, not payloads. A 40-paper
corpus costs a ~300-token digest instead of tens of thousands per turn.

**Multi-modal**: PDF · dataset (CSV/Excel) · figure image (GPT-4o vision) · audio (Whisper)

**Human approval**, two styles:
- *Static* — writing an artefact always asks
- *Dynamic* — a search only asks when it gets expensive

**Deferred approval survives a process restart.** Verified across two separate
processes: 9 papers, plan, synthesis, 13 RAG chunks restored → approved → completed.

<!-- The cross-process resume is the piece people don't expect. A web request
can't block on a human, so the checkpoint is what makes the UI approval real. -->

---

## Results, and what it does not do

**Working today** — 8 stages · 8 agents · 18 tools · **73 tests in ~8s** ·
runs with **no API key** (only the model is faked; search, RAG, PDF, statistics
and approvals are real)

**Honest limitations, stated in the product itself:**
- Reads **abstracts**, not full texts (except supplied PDFs)
- Offline embeddings are **lexical, not semantic**
- Novelty scores are **corpus-relative**
- Peer Reviewer shares the base model → inherits its blind spots
- **No experiments are run.** Designs are proposed, never executed

> A system that discloses its own failure modes is the only kind worth trusting
> with a research question.

<!-- Close on the limitations slide deliberately. It's the most credible thing in
the deck, and it's the behaviour the system itself is built around. -->
