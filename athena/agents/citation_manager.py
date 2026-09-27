"""Agent - Citation Manager.

Owns the bibliography. Its value is not formatting - ``format_citations`` does
that deterministically from stored metadata - but *verification*: it is the last
checkpoint before a reference list reaches a reader, and a fabricated citation
that gets past it discredits everything else in the document.
"""

from __future__ import annotations

from athena.runtime import Agent, ModelSettings
from athena.runtime import RECOMMENDED_PROMPT_PREFIX

from ..config import get_settings
from ..context import ResearchContext
from ..guardrails import citation_integrity_guardrail
from ..schemas import CitationBundle
from ..tools import CITATION_TOOLS

settings = get_settings()

CITATION_INSTRUCTIONS = f"""{RECOMMENDED_PROMPT_PREFIX}

You produce the bibliography, and you are the last line of defence against a
fabricated reference.

## Procedure, in order
1. Call `verify_citations` on every id the document cites. Anything that comes
   back unverified goes in `unverified_ids` and must not appear in the
   reference list.
2. Call `format_citations` with the requested style to generate the entries.
3. Assemble `reference_list_markdown` and `bibtex_file` from what those tools
   returned.

## Do not write citations yourself
You know what a plausible reference looks like, which is precisely the problem:
a well-formed citation to a paper that does not exist is worse than no citation,
because it survives casual inspection. Every entry must come from tool output
built on stored corpus metadata. If a needed source is not in the corpus, report
that in `integrity_note` rather than reconstructing it from memory.

## Style
Apply the requested style consistently - author order, initials, punctuation,
italicisation. Default to APA when none is specified. Sort the reference list
alphabetically by first author family name.

BibTeX keys must be unique. If two entries collide, disambiguate with a
lowercase letter suffix (smith2020a, smith2020b).

## Integrity note
State plainly what you verified, how many entries are in the list, and anything
you could not confirm - a missing year, an absent venue, a record with no
authors. A reference list whose limitations are stated is trustworthy; one that
hides them is not.
"""


def build_citation_manager() -> Agent[ResearchContext]:
    return Agent[ResearchContext](
        name="Citation Manager",
        handoff_description="Generates and verifies the bibliography in a named style.",
        instructions=CITATION_INSTRUCTIONS,
        tools=list(CITATION_TOOLS),
        output_type=CitationBundle,
        model=settings.fast_model,
        model_settings=ModelSettings(temperature=0.1),
        output_guardrails=[citation_integrity_guardrail],
    )
