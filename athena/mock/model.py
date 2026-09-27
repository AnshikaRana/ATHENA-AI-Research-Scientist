"""An offline test double for the model layer.

This exists so the entire system - agents, handoffs, tools, guardrails,
reflection loop, approvals, persistence - can be exercised with no API key and
no spend. It is a *test double*, not a simulation of intelligence: it returns
schema-valid fixtures, not reasoning.

Two details make the offline runs genuinely useful rather than a hollow demo:

1. The literature, RAG, PDF, sandbox and memory tools are all still real. Only
   the model is faked, so offline runs hit arXiv, OpenAlex and Crossref for
   actual papers.
2. Fixtures cite ``paper_id`` values pulled from the live corpus, so the
   citation-integrity guardrail does real work offline instead of trivially
   passing.

It implements :class:`athena.runtime.Model` - the same interface the live
OpenAI backend implements - so the runner cannot tell the difference.
"""

from __future__ import annotations

import json
from typing import Any

from ..logging_setup import get_logger, log_event
from ..runtime import ModelResponse, Usage

logger = get_logger("mock.model")

# The mock has no access to the run context through the Model interface, so the
# pipeline registers the active context here before running. Deliberately
# module-level: this is scaffolding for offline mode, not production plumbing.
_ACTIVE_CONTEXT: Any = None


def set_active_context(context: Any) -> None:
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context


def _corpus_ids(limit: int = 4) -> list[str]:
    """Real paper ids from the live corpus, so citations validate."""
    if _ACTIVE_CONTEXT is None or not getattr(_ACTIVE_CONTEXT, "corpus", None):
        return []
    return list(_ACTIVE_CONTEXT.corpus.keys())[:limit]


def _run_id() -> str:
    """Current run id, used to scope the mock's tool-script bookkeeping.

    Sessions persist across runs, so a replayed transcript from an earlier run
    would otherwise be counted as progress through *this* run's script, and the
    agent would skip straight past its tools.
    """
    if _ACTIVE_CONTEXT is None:
        return "norun"
    return getattr(_ACTIVE_CONTEXT, "run_id", "norun")


def _question() -> str:
    if _ACTIVE_CONTEXT is None:
        return "the research question"
    return getattr(_ACTIVE_CONTEXT, "research_question", "the research question")


# ---------------------------------------------------------------------------
# Fixtures, one per structured output type
# ---------------------------------------------------------------------------


def _fixture(schema_name: str) -> dict[str, Any]:
    question = _question()
    ids = _corpus_ids()
    note = "[OFFLINE FIXTURE] Generated without a model call."

    if schema_name == "ResearchPlan":
        return {
            "research_question": question,
            "interpretation": f"{note} Scoped to peer-reviewed and preprint literature.",
            "sub_questions": [
                f"What does existing work establish about {question}?",
                "Which methodological approaches dominate, and what do they assume?",
                "What does the field explicitly identify as unresolved?",
            ],
            "search_keywords": [w for w in question.split() if len(w) > 4][:6]
            or ["research"],
            "steps": [
                {"order": 1, "stage": "literature", "objective": "Build the corpus",
                 "rationale": "Nothing downstream is grounded without sources."},
                {"order": 2, "stage": "synthesis", "objective": "Establish what is known",
                 "rationale": "Gaps are only visible against established knowledge."},
                {"order": 3, "stage": "hypothesis", "objective": "Propose testable claims",
                 "rationale": "Gaps become research only when made falsifiable."},
                {"order": 4, "stage": "experiment", "objective": "Design a test",
                 "rationale": "A hypothesis without a protocol cannot be acted on."},
            ],
            "success_criteria": [
                "At least 8 relevant sources retrieved",
                "Every claim traced to a retrieved paper",
                "Sample size computed rather than asserted",
            ],
            "out_of_scope": ["Running the experiment", "Proprietary or unpublished data"],
        }

    if schema_name == "LiteratureReview":
        papers = []
        if _ACTIVE_CONTEXT is not None:
            for paper in list(_ACTIVE_CONTEXT.corpus.values())[:12]:
                papers.append(json.loads(paper.model_dump_json()))
        return {
            "query_terms_used": [question],
            "papers": papers,
            "total_found": len(papers),
            "coverage_assessment": (
                f"{note} Corpus assembled from live arXiv/OpenAlex/Crossref queries; "
                "relevance ranking was not performed by a model."
            ),
            "notable_gaps": ["Offline mode performed no model-driven query expansion."],
        }

    if schema_name == "KnowledgeSynthesis":
        return {
            "summary": f"{note} Synthesis over {len(ids)} retrieved sources on {question}.",
            "key_claims": [
                {
                    "statement": "The retrieved corpus addresses the question but was "
                                 "not interpreted by a model in offline mode.",
                    "supporting_paper_ids": ids[:2],
                    "confidence": "low",
                    "contested": False,
                }
            ],
            "consensus_points": ["Not established offline."],
            "disagreements": ["Not established offline."],
            "gaps": [
                {
                    "gap": "Offline mode cannot identify substantive research gaps.",
                    "why_it_matters": "Gap analysis requires reading the sources.",
                    "evidence_paper_ids": ids[:1],
                }
            ],
            "papers_cited": ids[:2],
        }

    if schema_name == "GapAnalysis":
        return {
            "gaps": [
                {
                    "gap_id": "G1",
                    "title": f"{note} Placeholder gap for {question}.",
                    "description": "Offline mode cannot analyse the corpus for real gaps.",
                    "severity": "significant",
                    "gap_type": "empirical",
                    "evidence_paper_ids": ids[:2],
                    "why_unaddressed": "Not determined offline.",
                    "tractability": 5,
                    "potential_impact": "Not determined offline.",
                }
            ],
            "priority_gap_id": "G1",
            "prioritisation_rationale": f"{note} Only one placeholder gap exists.",
            "saturated_areas": ["Not determined offline."],
            "corpus_limitations": (
                f"Corpus of {len(_corpus_ids(999))} papers was retrieved live, but no "
                "model reasoning was applied to it."
            ),
        }

    if schema_name == "DataInterpretation":
        return {
            "data_source": "corpus findings",
            "approach": f"{note} No analysis was performed.",
            "results": [
                {
                    "metric": "placeholder",
                    "value": 0.0,
                    "comparison": "",
                    "p_value": -1.0,
                    "effect_size": -1.0,
                    "interpretation": "Offline fixture; no data was analysed.",
                }
            ],
            "patterns": ["None identified offline."],
            "anomalies": [],
            "statistical_caveats": ["No statistical analysis was performed."],
            "supports_hypothesis": "not_applicable",
            "confidence": "low",
            "interpretation_summary": f"{note} No interpretation available offline.",
        }

    if schema_name == "CitationBundle":
        citations = []
        entries = []
        if _ACTIVE_CONTEXT is not None:
            from ..tools.citations import _bibtex_entry, _bibtex_key, _in_text, render_reference

            for paper in list(_ACTIVE_CONTEXT.corpus.values())[:12]:
                key = _bibtex_key(paper)
                entry = _bibtex_entry(paper, key)
                entries.append(entry)
                citations.append(
                    {
                        "paper_id": paper.paper_id,
                        "in_text": _in_text(paper.authors, paper.year),
                        "full_reference": render_reference(paper, "APA"),
                        "bibtex_key": key,
                        "bibtex_entry": entry,
                    }
                )
        return {
            "style": "APA",
            "citations": citations,
            "reference_list_markdown": "\n".join(
                f"- {c['full_reference']}" for c in citations
            ),
            "bibtex_file": "\n\n".join(entries),
            "unverified_ids": [],
            "integrity_note": (
                f"{note} Citations were generated deterministically from corpus "
                "metadata by the citation tool, so they are real even offline."
            ),
        }

    if schema_name == "HypothesisSet":
        return {
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "statement": f"{note} Placeholder hypothesis regarding {question}.",
                    "rationale": "Offline fixture; not derived from the literature.",
                    "null_hypothesis": "No effect is observed.",
                    "independent_variables": ["method"],
                    "dependent_variables": ["outcome metric"],
                    "novelty_score": 5,
                    "feasibility_score": 5,
                    "impact_score": 5,
                    "grounded_in_paper_ids": ids[:2],
                    "addresses_gap": "Offline placeholder gap.",
                }
            ],
            "recommended_hypothesis_id": "H1",
            "selection_rationale": f"{note} Only one candidate exists offline.",
        }

    if schema_name == "ExperimentDesign":
        return {
            "hypothesis_id": "H1",
            "title": f"[OFFLINE] Proposed test of {question}",
            "methodology": f"{note} Two-group comparison with random assignment.",
            "independent_variables": ["method"],
            "dependent_variables": ["outcome metric"],
            "controls": ["Matched baseline condition"],
            "datasets_or_materials": ["Public benchmark dataset"],
            "procedure_steps": [
                "Assign units to control and treatment",
                "Apply the manipulation",
                "Measure the outcome across repeated seeds",
            ],
            "power_analysis": {
                "effect_size": 0.5,
                "alpha": 0.05,
                "power": 0.8,
                "required_sample_size": 65,
                "justification": "Computed by the power-analysis tool for a medium effect.",
            },
            "statistical_tests": ["Two-sample t-test with Welch correction"],
            "threats_to_validity": ["Offline fixture has not been reviewed for confounds."],
            "ethical_considerations": ["No human subjects in this placeholder design."],
            "estimated_duration": "6 weeks",
            "expected_outcomes": "A significant group difference would support H1.",
        }

    if schema_name == "CritiqueReport":
        return {
            "approved": True,
            "overall_assessment": (
                f"{note} No model-based review was performed; approving to let the "
                "offline pipeline complete. Do not read this as a quality signal."
            ),
            "rigor_score": 5,
            "novelty_score": 5,
            "evidence_grounding_score": 5,
            "findings": [
                {
                    "severity": "major",
                    "section": "whole run",
                    "issue": "Produced in offline mode without model reasoning.",
                    "required_fix": "Re-run with an OPENAI_API_KEY_2 configured.",
                }
            ],
            "revision_instructions": "",
        }

    if schema_name == "ResearchReport":
        refs = []
        if _ACTIVE_CONTEXT is not None:
            for paper in list(_ACTIVE_CONTEXT.corpus.values())[:10]:
                authors = paper.authors[0] if paper.authors else "Unknown"
                refs.append(f"{authors} et al. ({paper.year}). {paper.title}. {paper.url}")
        return {
            "title": f"[OFFLINE] Research Dossier: {question}",
            "research_question": question,
            "executive_summary": (
                f"{note} This dossier was assembled without model reasoning. The "
                f"corpus of {len(_corpus_ids(999))} papers was retrieved from live "
                "academic APIs, but no synthesis, hypothesis generation or review "
                "was performed. Configure an OPENAI_API_KEY_2 for a substantive report."
            ),
            "sections": [
                {"heading": "Introduction", "body_markdown": f"Question: {question}"},
                {"heading": "Related Work",
                 "body_markdown": f"Corpus of {len(refs)} retrieved works."},
                {"heading": "Limitations",
                 "body_markdown": "Produced in offline mode. No claims are model-derived."},
            ],
            "key_findings": ["Offline mode produces no findings."],
            "limitations": [
                "No model reasoning was performed.",
                "Claims and scores are fixtures, not analysis.",
            ],
            "references": refs,
            "confidence": "low",
            "word_count": 120,
        }

    if schema_name == "RelevanceCheck":
        return {
            "is_research_question": True,
            "reasoning": f"{note} Offline screening approves by default.",
            "suggested_rewrite": "",
        }

    if schema_name == "IntegrityCheck":
        return {
            "has_unsupported_claims": False,
            "flagged_statements": [],
            "reasoning": f"{note} No model-based integrity check offline.",
        }

    return {}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _text_item(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": text}


def _call_item(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


class MockModel:
    """Returns schema-valid fixtures, and drives the real tool layer.

    The tool script below is what makes offline mode worth having. Each agent
    is walked through the tools it would really call - literature search against
    live APIs, corpus indexing, retrieval, power analysis, the approval-gated
    save - and only then handed its fixture. So an offline run genuinely
    exercises the tools, the guardrails and the human-approval interrupt; only
    the reasoning is faked.
    """

    # schema name -> ordered (tool_name, arguments) the mock will call.
    TOOL_SCRIPT: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "ResearchPlan": [
            ("recall_memory", {"query": "prior related work", "limit": 3}),
        ],
        "LiteratureReview": [
            ("search_literature", {"sources": ["arxiv", "openalex", "crossref"],
                                   "max_results_per_source": 5}),
            ("list_corpus", {}),
        ],
        "KnowledgeSynthesis": [
            ("index_corpus", {}),
            ("rag_search", {"top_k": 5}),
        ],
        "GapAnalysis": [
            ("rag_search", {"top_k": 5}),
        ],
        "CitationBundle": [
            ("verify_citations", {}),
            ("format_citations", {"style": "APA"}),
        ],
        "HypothesisSet": [
            ("rag_search", {"top_k": 4}),
        ],
        "ExperimentDesign": [
            ("statistical_power_analysis", {"effect_size": 0.5, "alpha": 0.05, "power": 0.8}),
        ],
        "ResearchReport": [
            ("remember_finding", {"kind": "finding", "topic": "offline run",
                                  "content": "Offline fixture run completed."}),
            # Triggers the human-approval interrupt, offline.
            ("save_research_artifact", {"artifact_type": "report"}),
        ],
    }

    def __init__(self, name: str = "mock") -> None:
        self.name = name

    def _next_tool_call(
        self, schema_name: str, available: set[str], calls_made: int
    ) -> tuple[str, dict[str, Any]] | None:
        """The next scripted call this agent can actually make, if any."""
        script = [
            step for step in self.TOOL_SCRIPT.get(schema_name, []) if step[0] in available
        ]
        return script[calls_made] if calls_made < len(script) else None

    def _arguments_for(self, tool_name: str, base: dict[str, Any]) -> str:
        """Fill in arguments that depend on the live run context."""
        args = dict(base)
        question = _question()
        if tool_name == "search_literature":
            args["query"] = question
        elif tool_name == "rag_search":
            args["question"] = question
        elif tool_name == "recall_memory":
            args["query"] = question
        elif tool_name in {"verify_citations", "format_citations"}:
            args["paper_ids"] = _corpus_ids(12)
        elif tool_name == "save_research_artifact":
            args["filename"] = "offline_research_dossier"
            args["content"] = (
                f"# [OFFLINE] Research Dossier\n\n## Question\n{question}\n\n"
                f"## Corpus\n{len(_corpus_ids(999))} papers retrieved from live "
                f"academic APIs.\n\n## Note\nGenerated in offline mode without "
                f"model reasoning.\n"
            )
        return json.dumps(args)

    async def get_response(
        self,
        *,
        system_instructions: str | None,
        input: str | list[dict[str, Any]],
        model_settings: Any,
        model_name: str,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        force_output: bool = False,
    ) -> ModelResponse:
        usage = Usage(requests=1, input_tokens=0, output_tokens=0, total_tokens=0)
        available = {getattr(t, "name", "") for t in tools}
        is_plain_text = output_schema is None or output_schema.is_plain_text()
        schema_name = "PlainText" if is_plain_text else output_schema.name()

        call_prefix = f"mock_call_{_run_id()}_{schema_name}_"
        calls_made = _count_tool_outputs(input, call_prefix)

        # A plain-text agent with handoffs is the Director: transfer, so the
        # handoff machinery is exercised offline too.
        if is_plain_text and handoffs:
            target = handoffs[0]
            log_event(logger, "mock handoff", to=target.tool_name)
            return ModelResponse(
                output=[_call_item("mock_handoff_call", target.tool_name, "{}")],
                usage=usage,
                response_id="mock_handoff_response",
            )

        if is_plain_text:
            return ModelResponse(
                output=[_text_item("[OFFLINE FIXTURE] No model reasoning available.")],
                usage=usage,
                response_id="mock_text_response",
            )

        # Walk the tool script before producing the final structured output.
        # On the repair path the caller wants the artefact itself, not a tool call.
        step = None if force_output else self._next_tool_call(
            schema_name, available, calls_made
        )
        if step is not None:
            tool_name, base_args = step
            arguments = self._arguments_for(tool_name, base_args)
            log_event(logger, "mock tool call", tool=tool_name, schema=schema_name)
            return ModelResponse(
                output=[_call_item(f"{call_prefix}{calls_made}", tool_name, arguments)],
                usage=usage,
                response_id=f"mock_tool_{tool_name}",
            )

        payload = _fixture(schema_name)
        log_event(logger, "mock structured output", schema=schema_name)
        return ModelResponse(
            output=[_text_item(json.dumps(payload))],
            usage=usage,
            response_id=f"mock_{schema_name}",
        )


def _count_tool_outputs(
    input: str | list[dict[str, Any]], call_id_prefix: str
) -> int:
    """Count completed tool calls *this agent* made.

    Counting every ``function_call_output`` in the input would be wrong: agents
    share a SQLiteSession, so the transcript handed to the Literature Review Agent
    already contains the Principal Investigator's tool results. That inflated
    count made each agent skip to the wrong step of its script. Matching on the
    call-id prefix scopes the count to the current agent's own calls.
    """
    if isinstance(input, str):
        return 0
    count = 0
    for item in input:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call_output":
            continue
        if str(item.get("call_id", "")).startswith(call_id_prefix):
            count += 1
    return count


class MockModelProvider:
    """Hands every agent the same mock model."""

    def __init__(self) -> None:
        self._model = MockModel()

    def get_model(self, model_name: str | None) -> MockModel:
        return self._model
