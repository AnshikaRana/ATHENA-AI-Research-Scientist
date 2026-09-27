# 4. Project Documentation

## 4.1 Quick start

```bash
git clone <your-repo-url> && cd agentic-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # optional — the system runs without a key
```

```bash
# Full eight-stage study
python -m athena.cli study "What limits the expressive power of message passing GNNs?"

# Attach source material (multi-modal)
python -m athena.cli study "Do ViTs beat CNNs on small datasets?" \
    --attach paper.pdf --attach figure.png

# Resume a run that was paused for approval
python -m athena.cli resume run_20260908_183518_a82f7b

# Inspect the team and long-term memory
python -m athena.cli agents
python -m athena.cli memory --query "graph neural networks"

# Web interface
streamlit run app.py
```

### Offline mode

With no `OPENAI_API_KEY_2`, Athena runs offline. **Only the model layer is faked.**
Literature search hits arXiv, OpenAlex and Crossref for real; PDFs are really
parsed; the RAG index is really built; power analyses, t-tests and citations are
really computed; the approval gate really fires. Model *reasoning* is replaced by
schema-valid fixtures that are clearly labelled `[OFFLINE FIXTURE]`.

This exists so the system is demonstrable without spend, and so the test suite is
hermetic. It is a test double, not a simulation of intelligence, and the fixtures
say so in their own text.

## 4.2 Module map

```
athena/
├── config.py            Settings; the only module that reads the environment
├── schemas.py           15 Pydantic models — every structured output
├── context.py           ResearchContext: working memory, corpus, ledger, serialisation
├── pipeline.py          Orchestration: the eight stages, approvals, checkpoints
├── guardrails.py        Input relevance + deterministic citation integrity
├── hooks.py             Lifecycle hooks: trace, handoff log, cost accounting
├── logging_setup.py     Rich console + JSONL audit sink
├── errors.py            Typed exceptions
├── cli.py               Command-line interface
├── runtime/             In-house agent runtime over the OpenAI Chat Completions API
│                        (Agent, Runner, tool schemas, handoffs, guardrails,
│                        SQLiteSession, the serialisable approval interrupt)
├── agents/              11 agents, one module each
├── tools/               20 tools across 9 capability groups
├── memory/
│   ├── vector_store.py  Chunking, hashed lexical embedding, cosine retrieval, persistence
│   └── long_term.py     Cross-session semantic memory (SQLite + hashed embedding)
└── mock/model.py        Offline test double for the model layer

app.py                   Streamlit UI
scripts/generate_diagram.py   Architecture diagram, generated from live code
tests/                   82 tests
```

## 4.3 Advanced features

### Planning and reasoning
The Principal Investigator emits an explicit `ResearchPlan` before any work
happens: a sharpened question, 2–5 decomposed sub-questions, search keywords
including field jargon and historical synonyms, ordered steps respecting
dependencies, observable success criteria, and explicit out-of-scope boundaries.
Downstream stages consume the plan — the Synthesist addresses each sub-question
with its own retrieval, and the Literature Review Agent searches the keywords.

Reasoning depth is tuned per role: temperature 0.1 for routing and citation
formatting, 0.2 for synthesis, review and interpretation, 0.4 for gap analysis,
0.8 for hypothesis generation, reinforced by a "divergence discipline" block in
the Hypothesis Generator's instructions.

### RAG / knowledge retrieval
`athena/memory/vector_store.py`. Sentence-boundary chunking at ~900 characters
with 150-character overlap (a claim straddling a boundary would otherwise be
retrievable from neither chunk); L2-normalised cosine similarity; persisted as
`.npy` + JSON per run.

Retrieval is kept lexical by design: every run — live or offline — embeds with
signed feature hashing over unigrams **and bigrams** with sublinear TF. The bigrams are not decoration: without them
"expressive power" and "power analysis" collide on the shared token `power` and
the wrong document ranks first. With them, ranking separates cleanly
(0.348 vs 0.073 on the test case). The fallback is genuinely lexical, not
semantic, and the code says so.

### Long-term memory
Three tiers, kept distinct (see §2.3). The long-term tier stores findings,
questions, preferences and critiques as embedded rows in SQLite, searchable
semantically across sessions. The Principal Investigator queries it first on
every run; the Scientific Writer writes the durable insight at the end.

### Reflection / self-review
The Peer Reviewer's `approved` flag gates progression. Four artefacts are
reviewed **concurrently**; blocking findings return each artefact to its
producing agent with an actionable brief, revisions run concurrently, and the
work is re-reviewed. Bounded at `max_reflection_rounds` (default 2), after which
unresolved criticisms are recorded and carried into the report's Limitations.

Verified by `test_reflection_loop_revises_then_approves` (rejects round 1,
approves round 2) and `test_reflection_budget_is_bounded` (perpetual rejection
terminates rather than looping).

### Parallel agent execution
Two genuine sites:
- **Literature search** — three databases via `asyncio.gather`, with
  `return_exceptions=True` so one outage degrades rather than aborts. Measured
  wall time ~2.2 s for three sources.
- **Peer review** — four concurrent Peer Reviewer runs, then concurrent
  revisions. Each runs sessionless: sharing a `SQLiteSession` across parallel
  runs would let reviewers read each other's half-written turns.

### Error handling and logging
- Typed exception hierarchy (`AthenaError` and descendants).
- HTTP layer with bounded retries and **exponential backoff plus jitter** — the
  jitter matters because three parallel searches would otherwise synchronise and
  hammer a rate-limited endpoint together.
- Per-stage error capture: a failed stage records the error and the pipeline
  continues, so one bad stage does not lose the run.
- `failure_error_function` on tools returns failures to the agent as text it can
  act on rather than raising.
- Dual logging: Rich console plus a JSONL audit file per run, replayed by the UI
  as an agent trace.
- Honest degradation: a dead database is disclosed in the report's Limitations;
  an empty corpus **aborts before synthesis** rather than inventing findings.

### Multi-modal inputs
| Input | Tool | Live | Offline |
|---|---|---|---|
| PDF | `ingest_pdf` | ✅ | ✅ (fully local) |
| Dataset (CSV/TSV/JSON/Excel) | `analyze_dataset`, `compare_groups` | ✅ | ✅ (fully local) |
| Figure image | `describe_figure` | ✅ GPT-4o vision | Registered, explicitly not analysed |
| Audio | `transcribe_audio` | ✅ Whisper | Registered, explicitly not transcribed |

Offline, the image and audio tools return a labelled placeholder that instructs
the agent not to invent findings — a silent no-op would be worse than useless.

### Session persistence
- `SQLiteSession` for conversation memory across runs sharing a session id.
- **Deferred approval** checkpoints both `RunState` *and* the serialised
  `ResearchContext`, so a pending decision survives a process restart. Verified
  end-to-end across two genuinely separate processes: 9 papers, plan, synthesis
  and 13 RAG chunks restored, then approved and completed.
- The vector index is persisted separately per run and reattached on restore.

### Human approval
Static gate on artefact writes; dynamic, cost-based gate on wide searches. The
CLI offers approve / reject / **defer**; the Streamlit UI uses defer-and-resume,
because a web request cannot block on a human. Every decision is written to the
audit trail and to a provenance sidecar beside the artefact.

## 4.4 Testing

```bash
pytest                    # 82 tests, ~5 seconds, no API key, no network
```

Tests are hermetic: outbound HTTP is stubbed and the model is already a fixture
offline. Coverage of note:

- **Numerical correctness** — power analysis validated against Cohen's published
  tables; p-values against standard t-tables to 4 decimal places.
- **Sandbox security** — six escape attempts, all blocked.
- **Citation integrity** — fabricated ids trip the guardrail; real ids pass; the
  guardrail stays quiet before the corpus exists (otherwise it would block the
  planning stage every run).
- **Orchestration** — all ten stages in order; reflection loop revises and
  terminates; approval fires, is audited, and survives a process restart.
- **Resilience** — a failing database degrades the corpus; an empty corpus aborts
  before ungrounded synthesis.

## 4.5 Known limitations

Stated plainly, because the system is built to disclose its own failures:

1. **Abstracts, not full texts.** Only user-supplied PDFs are read in full.
   Conclusions from abstracts alone are weaker, and the report says so.
2. **The offline fallback is lexical.** Hashed bigram embeddings match vocabulary,
   not meaning. Paraphrases with no shared words will not retrieve.
3. **Novelty scores are corpus-relative.** A hypothesis scored 9 may be
   well-known in literature the search did not reach.
4. **OpenAlex is intermittently unavailable.** Observed failing in several live
   runs. The system degrades and discloses; it does not hide it.
5. **The Peer Reviewer shares the base model** with the agents it reviews, so it
   inherits their blind spots. It catches ungrounded claims and internal
   inconsistency, not domain error.
6. **No experiments are run.** Designs are proposed, never executed.
