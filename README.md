# 🔬 Athena — AI Research Scientist

**A multi-agent research system that carries a question from "I wonder…" to a
cited, publication-ready dossier — and cannot cite a paper it never retrieved.**

Built on the **OpenAI API** (`openai` SDK, Chat Completions) with a small
in-house agent runtime under `athena/runtime/` — no agent framework.
8 specialised agents · 18 tools · 4 external APIs · 8-stage pipeline · 73 tests.

<!-- Replace with your own repo/CI badges once pushed -->
![Python](https://img.shields.io/badge/python-3.12-blue)
![OpenAI SDK](https://img.shields.io/badge/openai-SDK-black)
![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)

---

## The problem this solves

Ask a general-purpose chatbot for a literature review and you will get fluent
prose containing references that do not exist. Published fabrication rates for
LLM-generated citations run from 30% to 90% depending on domain. A research
assistant you must fact-check line by line has cost you more time than it saved.

Athena's answer is architectural rather than persuasive. The Knowledge Synthesist
can only cite what the Literature Review Agent actually put in the shared corpus,
and a **deterministic output guardrail** rejects any `paper_id` that is not in
it. Fabrication is not discouraged by instruction — it is impossible by
construction.

```python
# athena/guardrails.py — this is a set lookup, not a model judgement
fabricated = sorted(cid for cid in cited if cid not in ctx.corpus)
return GuardrailFunctionOutput(output_info=info, tripwire_triggered=bool(fabricated))
```

---

## Quick start

```bash
pip install -r requirements.txt
python -m athena.cli study "What limits the expressive power of message passing GNNs?"
```

**No API key required.** Without an `OPENAI_API_KEY_2`, Athena runs offline:
model reasoning is replaced by clearly-labelled fixtures, while literature
search, PDF parsing, RAG indexing, audio transcription, statistics, citations
and the human-approval gate all remain **real**.

```bash
streamlit run app.py          # web UI with live agent trace + approval buttons
pytest                        # 90 tests, no key, no network
```

**Studying a paper you already have.** Attach a PDF in the web UI and choose
*Uploaded documents only*. The PDF is parsed locally, web search is switched
off, and every agent reads the paper's full text, so each claim and number in
the report traces back to it. `ATHENA_MAX_RUN_USD` caps what a single run may
spend.

---

## Architecture

![agents](https://img.shields.io/badge/agents-8-blue) ![tools](https://img.shields.io/badge/tools-20-blue) ![stages](https://img.shields.io/badge/pipeline%20stages-8-blue)

```
Principal Investigator ─▶ Literature Review ─▶ Knowledge Synthesist ─▶ Gap Analyst
        ─▶ Hypothesis Generator ─▶ Experiment Planner ─▶ Citation Manager
                    ─▶ Scientific Writer ─▶ ⏸ HUMAN APPROVAL
```

Full generated diagram: **[docs/03_architecture_diagram.md](docs/03_architecture_diagram.md)**

### The eight agents

| Agent | Output | Why it exists |
|---|---|---|
| **Principal Investigator** | `ResearchPlan` | Sharpens the question; sets out-of-scope bounds so a broad question can't become an unbounded search |
| **Literature Review Agent** | `LiteratureReview` | Acquires evidence from 3 databases. Forbidden from interpreting it |
| **Knowledge Synthesist** | `KnowledgeSynthesis` | RAG-grounded synthesis. May not use its own parametric knowledge |
| **Research Gap Analyst** | `GapAnalysis` | Typed, severity- and tractability-rated gaps |
| **Hypothesis Generator** | `HypothesisSet` | Falsifiable proposals. The one divergent agent (temp 0.8) |
| **Experiment Planner** | `ExperimentDesign` | Protocols whose sample size is *computed*, never guessed |
| **Citation Manager** | `CitationBundle` | Verifies, then formats. Never writes a citation itself |
| **Scientific Writer** | `ResearchReport` | Assembles the dossier; owns the approval-gated save |

---

## What makes it more than a prompt chain

**Deterministic where dependencies are hard.** You cannot synthesise evidence
you have not retrieved, or design an experiment for a hypothesis you have not
chosen, so the pipeline sequence is fixed in code. The judgement inside each
stage is the model's.

**Numbers come from tools, not from the model.** Sample sizes match Cohen's
published tables; p-values are computed from a hand-implemented incomplete beta
function and match standard t-tables to four decimals. An LLM that estimates a
sample size has produced a design that looks rigorous and isn't.

**Fabrication is blocked deterministically.** A cited `paper_id` that is not in
the retrieved corpus trips the citation-integrity output guardrail and the run
fails — it is a set lookup, not a reviewer's judgement.

**Approvals that outlive the process.** Deferring an approval checkpoints both
the SDK `RunState` *and* the working memory. Verified across two separate
processes: 9 papers, plan, synthesis and 13 RAG chunks restored, then approved
and completed.

**Context that doesn't blow up.** Agents exchange references, not payloads. A
40-paper corpus lives in `ResearchContext` and costs a ~300-token digest in the
transcript instead of tens of thousands of tokens per turn.

**A sandbox that's actually a sandbox.** Model-written code passes AST validation
(import allowlist, no `eval`/`open`/`__import__`, dunder access blocked) and then
runs in a separate isolated process with a hard timeout. Six escape attempts are
tested; all are blocked.

---

## Tools (18)

| Group | Tools | Backed by |
|---|---|---|
| Literature | `search_literature`, `commission_deep_sweep`, `get_paper_details`, `list_corpus` | arXiv · OpenAlex · Crossref |
| RAG | `index_corpus`, `rag_search` | hashed lexical embedding + cosine |
| Documents | `ingest_pdf`, `describe_figure`, `transcribe_audio` | pypdf · GPT-4o vision · Whisper |
| Analysis | `statistical_power_analysis`, `run_python_analysis`, `analyze_dataset`, `compare_groups` | pandas · numpy · sandbox |
| Citations | `format_citations`, `verify_citations` | Deterministic — APA/IEEE/MLA/Nature + BibTeX |
| Memory | `remember_finding`, `recall_memory` | SQLite + embeddings |
| Artefacts | `save_research_artifact` | **Human-approval gated** |

---

## Documentation

| Document | Contents |
|---|---|
| [01 — Problem Analysis](docs/01_problem_analysis.md) | Business context, stakeholders, problem statement, objectives |
| [02 — Multi-Agent Design](docs/02_multi_agent_design.md) | Architecture, agent roles, pipeline flow, tool integration |
| [03 — Architecture Diagram](docs/03_architecture_diagram.md) | Generated from live code, so it can't drift |
| [04 — Project Documentation](docs/04_documentation.md) | Setup, module map, advanced features, testing, limitations |
| [05 — Presentation](docs/05_presentation.md) | 12 slides |

---

## Honest limitations

- Reads **abstracts**, not full texts (except PDFs you supply).
- RAG retrieval is **lexical, not semantic** by design — a hashed bag-of-words
  embedding is used in every mode, so paraphrases with no shared vocabulary
  won't retrieve.
- Novelty scores are **corpus-relative**: a 9 may be well known in literature the
  search didn't reach.
- **No adversarial peer-review pass.** Fabricated citations are still blocked
  deterministically, but nothing catches overclaiming, an unjustified sample
  size, or an unfalsifiable hypothesis — that judgement is left to you.
- **No experiments are run.** Designs are proposed, never executed.

---

## Tech stack

`openai` · `pydantic 2` · `httpx` · `numpy` · `pandas` · `pypdf` ·
`feedparser` · `streamlit` · `rich` · `pytest`

The agent runtime (`Agent`, `Runner`, tool schema generation, guardrails,
`SQLiteSession`, the serialisable approval interrupt) lives in `athena/runtime/`
— a small, self-contained layer over the OpenAI Chat Completions API, not a
third-party agent framework.

## License

MIT
