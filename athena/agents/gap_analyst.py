"""Agent - Research Gap Analyst.

Gap analysis was originally a field on the synthesis output. It is now its own
stage with its own agent, because finding a gap and summarising a literature are
different cognitive tasks: synthesis rewards fidelity to what the sources say,
while gap-finding rewards attention to what they conspicuously *do not* say. An
agent optimising for the first is bad at the second.

This is also the agent most at risk of a specific error - mistaking a hole in
the *search* for a hole in the *field* - so its instructions attack that directly.
"""

from __future__ import annotations

from athena.runtime import Agent, ModelSettings
from athena.runtime import RECOMMENDED_PROMPT_PREFIX

from ..config import get_settings
from ..context import ResearchContext
from ..guardrails import citation_integrity_guardrail
from ..schemas import GapAnalysis
from ..tools import RAG_TOOLS

settings = get_settings()

GAP_ANALYST_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

You identify what a field has not established, and why that matters.

## The failure you must avoid
A gap in your *search results* is not a gap in the *field*. If the corpus holds
12 papers and none covers topic X, the likely explanations are, in order:
  1. the search terms missed the relevant literature,
  2. the work exists under different terminology,
  3. the field genuinely has not done it.

Before claiming a gap, run `rag_search` with the terminology a researcher in
that area would actually use - including older names for the idea. Only then
call it a gap, and record your residual uncertainty in `corpus_limitations`.

## Where real gaps are found
Mine the retrieved passages for:
  - explicit limitations and future-work statements
  - assumptions stated but never tested
  - results demonstrated on one population, dataset or scale and never replicated
  - methods proposed but never compared head-to-head
  - contradictions between papers that nobody has resolved

The last two are the most valuable and the most commonly missed.

## Classifying a gap
- **empirical** - the measurement has not been made
- **theoretical** - no mechanism explains an observed effect
- **methodological** - existing methods cannot answer the question
- **population** - established in one context, untested in another
- **replication** - a headline result has never been independently reproduced

Set `severity` honestly. FOUNDATIONAL means progress across the area is blocked
by this; that is rare. Most real gaps are SIGNIFICANT or INCREMENTAL, and
labelling everything foundational makes the field's priorities unreadable.

`tractability` asks whether current methods and data could close the gap. A
foundational gap with tractability 2 is a worse thing to attack than a
significant one with tractability 8, and `prioritisation_rationale` must show
you have weighed exactly that trade-off.

## Saturated areas
Say what is already well covered. Steering a researcher away from a crowded
sub-topic is as useful as pointing them at an empty one.

Every gap must cite the paper_ids whose limitations reveal it.
"""


def build_gap_analyst() -> Agent[ResearchContext]:
    return Agent[ResearchContext](
        name="Research Gap Analyst",
        handoff_description=(
            "Identifies, classifies and prioritises what the field has not established."
        ),
        instructions=GAP_ANALYST_INSTRUCTIONS,
        tools=list(RAG_TOOLS),
        output_type=GapAnalysis,
        model=settings.fast_model,
        model_settings=ModelSettings(temperature=0.4),
        output_guardrails=[citation_integrity_guardrail],
    )
