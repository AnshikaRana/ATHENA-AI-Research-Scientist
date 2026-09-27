"""Citation generation and verification.

Citations are generated **deterministically from stored metadata**, never by the
model. This is the single most effective defence against the failure mode that
discredits AI research assistants: a fluent, correctly-formatted reference to a
paper that does not exist.

The Citation Manager agent decides *what* to cite and in which style. This
module decides what the citation *says*, and it can only say what the corpus
already knows.
"""

from __future__ import annotations

import re
from typing import Literal

from athena.runtime import RunContextWrapper, function_tool

from ..context import ResearchContext
from ..logging_setup import get_logger, log_event
from ..schemas import Paper

logger = get_logger("tools.citations")

Style = Literal["APA", "IEEE", "MLA", "Nature"]


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------


def _split_name(name: str) -> tuple[str, str]:
    """Return (family, given). Sources are inconsistent, so be forgiving."""
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return ("Unknown", "")
    if "," in name:  # "Family, Given"
        family, _, given = name.partition(",")
        return (family.strip(), given.strip())
    parts = name.split(" ")
    return (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (parts[0], "")


def _initials(given: str) -> str:
    return " ".join(f"{part[0]}." for part in given.split() if part)


def _apa_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    formatted = []
    for author in authors[:20]:
        family, given = _split_name(author)
        formatted.append(f"{family}, {_initials(given)}".strip().rstrip(","))
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def _ieee_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    formatted = []
    for author in authors[:6]:
        family, given = _split_name(author)
        formatted.append(f"{_initials(given)} {family}".strip())
    if len(authors) > 6:
        formatted.append("et al.")
    return ", ".join(formatted)


def _mla_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    family, given = _split_name(authors[0])
    first = f"{family}, {given}".strip().rstrip(",")
    return first if len(authors) == 1 else f"{first}, et al."


def _in_text(authors: list[str], year: int) -> str:
    year_text = str(year) if year else "n.d."
    if not authors:
        return f"(Unknown, {year_text})"
    family, _ = _split_name(authors[0])
    if len(authors) == 1:
        return f"({family}, {year_text})"
    if len(authors) == 2:
        second, _ = _split_name(authors[1])
        return f"({family} & {second}, {year_text})"
    return f"({family} et al., {year_text})"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _bibtex_key(paper: Paper) -> str:
    family, _ = _split_name(paper.authors[0]) if paper.authors else ("unknown", "")
    first_word = next(
        (w.lower() for w in re.findall(r"[A-Za-z]+", paper.title) if len(w) > 3), "untitled"
    )
    slug = re.sub(r"[^a-z0-9]", "", family.lower()) or "unknown"
    return f"{slug}{paper.year or 'nd'}{first_word}"


def _bibtex_entry(paper: Paper, key: str) -> str:
    entry_type = "article" if paper.venue and "arxiv" not in paper.venue.lower() else "misc"
    fields = [
        f"  title        = {{{paper.title}}}",
        f"  author       = {{{' and '.join(paper.authors) or 'Unknown'}}}",
    ]
    if paper.year:
        fields.append(f"  year         = {{{paper.year}}}")
    if paper.venue:
        fields.append(f"  journal      = {{{paper.venue}}}")
    if paper.paper_id.startswith("doi:"):
        fields.append(f"  doi          = {{{paper.paper_id[4:]}}}")
    if paper.paper_id.startswith("arxiv:"):
        fields.append(f"  eprint       = {{{paper.paper_id[6:]}}}")
        fields.append("  archivePrefix = {arXiv}")
    if paper.url:
        fields.append(f"  url          = {{{paper.url}}}")
    return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"


def render_reference(paper: Paper, style: Style) -> str:
    """Render one reference-list entry. Missing fields degrade, never fabricate."""
    year = str(paper.year) if paper.year else "n.d."
    venue = paper.venue or "Preprint"
    title = paper.title.rstrip(".")

    if style == "APA":
        base = f"{_apa_authors(paper.authors)} ({year}). {title}. *{venue}*."
    elif style == "IEEE":
        base = f'{_ieee_authors(paper.authors)}, "{title}," {venue}, {year}.'
    elif style == "MLA":
        # rstrip the period: "et al." already ends with one.
        base = f'{_mla_authors(paper.authors).rstrip(".")}. "{title}." {venue}, {year}.'
    else:  # Nature
        base = f"{_ieee_authors(paper.authors)}. {title}. *{venue}* ({year})."

    if paper.paper_id.startswith("doi:"):
        base += f" https://doi.org/{paper.paper_id[4:]}"
    elif paper.url:
        base += f" {paper.url}"
    return base


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@function_tool
async def format_citations(
    wrapper: RunContextWrapper[ResearchContext], paper_ids: list[str], style: str
) -> str:
    """Generate verified citations for papers, in a named style.

    Citations are built from stored corpus metadata, so any id not in the corpus
    is reported as unverified rather than invented.

    Args:
        paper_ids: Corpus ids to cite. Pass an empty list to cite the whole corpus.
        style: One of "APA", "IEEE", "MLA", "Nature".
    """
    ctx = wrapper.context
    chosen_style: Style = style if style in {"APA", "IEEE", "MLA", "Nature"} else "APA"  # type: ignore[assignment]

    ids = paper_ids or list(ctx.corpus.keys())
    papers = ctx.resolve_citations(ids)
    missing = ctx.unknown_citations(ids)

    if not papers:
        return (
            f"None of the requested ids are in the corpus. Unverified: {missing}. "
            "Do not cite them."
        )

    lines = [f"{len(papers)} verified citations in {chosen_style} style:\n"]
    bibtex_entries = []
    for paper in sorted(papers, key=lambda p: (p.authors[0] if p.authors else "", -p.year)):
        key = _bibtex_key(paper)
        lines.append(
            f"[{paper.paper_id}]\n"
            f"  in-text:   {_in_text(paper.authors, paper.year)}\n"
            f"  reference: {render_reference(paper, chosen_style)}\n"
            f"  bibtex key: {key}"
        )
        bibtex_entries.append(_bibtex_entry(paper, key))

    if missing:
        lines.append(f"\nUNVERIFIED - not in corpus, do not cite: {missing}")

    log_event(
        logger,
        "citations formatted",
        style=chosen_style,
        verified=len(papers),
        unverified=len(missing),
    )
    lines.append("\nBibTeX:\n" + "\n\n".join(bibtex_entries))
    return "\n".join(lines)


@function_tool
async def verify_citations(
    wrapper: RunContextWrapper[ResearchContext], paper_ids: list[str]
) -> str:
    """Check whether cited ids exist in the retrieved corpus.

    Args:
        paper_ids: The ids appearing in a draft, to be checked before publication.
    """
    ctx = wrapper.context
    known = ctx.resolve_citations(paper_ids)
    missing = ctx.unknown_citations(paper_ids)
    lines = [f"Checked {len(paper_ids)} ids: {len(known)} verified, {len(missing)} unverified."]
    for paper in known:
        lines.append(f"  OK   [{paper.paper_id}] {paper.title[:70]} ({paper.year})")
    for pid in missing:
        lines.append(f"  FAIL [{pid}] not in corpus - remove it or retrieve the source")
    return "\n".join(lines)
