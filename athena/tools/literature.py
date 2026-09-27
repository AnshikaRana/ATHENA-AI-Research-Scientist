"""Literature discovery tools: arXiv, OpenAlex and Crossref.

Three independent public APIs, deliberately. Each has different coverage
(arXiv = preprints, OpenAlex = broad indexed metadata + citation counts,
Crossref = publisher-of-record DOIs), so querying all three and merging gives a
corpus no single source would produce.

Context-management note: these tools do **not** return full paper records to the
model. They write the records into ``ResearchContext.corpus`` and return a
compact digest. A 40-paper corpus therefore costs a few hundred tokens in the
transcript instead of tens of thousands.
"""

from __future__ import annotations

import asyncio
import html
import re

import feedparser
from athena.runtime import RunContextWrapper, function_tool

from ..config import get_settings
from ..context import ResearchContext
from ..errors import ExternalAPIError
from ..logging_setup import get_logger, log_event
from ..schemas import Paper
from ._http import fetch_json, fetch_text

logger = get_logger("tools.literature")

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Collapse whitespace and unescape entities from API payloads."""
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild the linear text."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indices in inverted_index.items():
        for index in indices:
            positions.append((index, word))
    positions.sort(key=lambda pair: pair[0])
    return _clean(" ".join(word for _, word in positions))


def _parse_arxiv(feed_text: str, limit: int) -> list[Paper]:
    feed = feedparser.parse(feed_text)
    papers: list[Paper] = []
    for entry in feed.entries[:limit]:
        raw_id = getattr(entry, "id", "")
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        if not arxiv_id:
            continue
        year = 0
        published = getattr(entry, "published", "")
        if len(published) >= 4 and published[:4].isdigit():
            year = int(published[:4])
        papers.append(
            Paper(
                paper_id=f"arxiv:{arxiv_id}",
                title=_clean(getattr(entry, "title", "")),
                authors=[a.name for a in getattr(entry, "authors", [])][:10],
                year=year,
                venue="arXiv preprint",
                abstract=_clean(getattr(entry, "summary", "")),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source="arxiv",
                citation_count=0,  # arXiv's API does not expose citations
            )
        )
    return papers


def _parse_openalex(payload: dict, limit: int) -> list[Paper]:
    papers: list[Paper] = []
    for work in payload.get("results", [])[:limit]:
        work_id = (work.get("id") or "").rsplit("/", 1)[-1]
        if not work_id:
            continue
        venue = ""
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        if source:
            venue = _clean(source.get("display_name") or "")
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        papers.append(
            Paper(
                paper_id=f"openalex:{work_id}" if not doi else f"doi:{doi}",
                title=_clean(work.get("title") or work.get("display_name") or ""),
                authors=[
                    _clean((a.get("author") or {}).get("display_name") or "")
                    for a in (work.get("authorships") or [])[:10]
                ],
                year=int(work.get("publication_year") or 0),
                venue=venue,
                abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
                url=work.get("doi") or work.get("id") or "",
                source="openalex",
                citation_count=int(work.get("cited_by_count") or 0),
            )
        )
    return papers


def _safe_year(date_parts: object) -> int:
    """Crossref emits ``date-parts`` as [[2024, 3]], [[None]] or even [[]].

    Every one of those shapes shows up in real responses, so parse defensively
    and fall back to 0 (our "unknown year" sentinel) rather than raising.
    """
    try:
        first = date_parts[0]  # type: ignore[index]
        value = first[0]
        return int(value) if value is not None else 0
    except (TypeError, IndexError, ValueError):
        return 0


def _parse_crossref(payload: dict, limit: int) -> list[Paper]:
    papers: list[Paper] = []
    for item in payload.get("message", {}).get("items", [])[:limit]:
        doi = item.get("DOI") or ""
        if not doi:
            continue
        titles = item.get("title") or []
        containers = item.get("container-title") or []
        year = _safe_year((item.get("issued") or {}).get("date-parts"))
        authors = [
            _clean(f"{a.get('given', '')} {a.get('family', '')}")
            for a in (item.get("author") or [])[:10]
        ]
        papers.append(
            Paper(
                paper_id=f"doi:{doi}",
                title=_clean(titles[0] if titles else ""),
                authors=[a for a in authors if a],
                year=year,
                venue=_clean(containers[0] if containers else ""),
                abstract=_clean(item.get("abstract") or ""),
                url=item.get("URL") or f"https://doi.org/{doi}",
                source="crossref",
                citation_count=int(item.get("is-referenced-by-count") or 0),
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Raw async search functions (also used directly for parallel execution)
# ---------------------------------------------------------------------------


async def search_arxiv_raw(query: str, limit: int) -> list[Paper]:
    text = await fetch_text(
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        },
    )
    return _parse_arxiv(text, limit)


async def search_openalex_raw(query: str, limit: int) -> list[Paper]:
    payload = await fetch_json(
        OPENALEX_API,
        params={"search": query, "per_page": limit, "sort": "relevance_score:desc"},
    )
    return _parse_openalex(payload, limit)


async def search_crossref_raw(query: str, limit: int) -> list[Paper]:
    payload = await fetch_json(
        CROSSREF_API, params={"query": query, "rows": limit, "sort": "relevance"}
    )
    return _parse_crossref(payload, limit)


SOURCES = {
    "arxiv": search_arxiv_raw,
    "openalex": search_openalex_raw,
    "crossref": search_crossref_raw,
}


def _digest(ctx: ResearchContext, papers: list[Paper], label: str, added: int) -> str:
    """A compact, model-facing summary. Full records stay in the context."""
    if not papers:
        return f"{label}: no results."
    lines = [f"{label}: {len(papers)} results, {added} new (corpus now {len(ctx.corpus)})."]
    for paper in papers[:12]:
        authors = paper.authors[0] + " et al." if paper.authors else "unknown"
        lines.append(
            f"- [{paper.paper_id}] {paper.title} ({authors}, {paper.year}"
            f"{f', {paper.citation_count} cites' if paper.citation_count else ''})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent-facing tools
# ---------------------------------------------------------------------------


@function_tool(failure_error_function=lambda ctx, err: f"Literature search failed: {err}")
async def search_literature(
    wrapper: RunContextWrapper[ResearchContext],
    query: str,
    sources: list[str],
    max_results_per_source: int,
) -> str:
    """Search academic databases for papers and add them to the shared corpus.

    Queries the requested sources concurrently and merges the results, removing
    duplicates by identifier and by normalised title.

    Args:
        query: A focused search phrase, e.g. "graph neural network expressivity".
        sources: Any of "arxiv", "openalex", "crossref". Use all three for breadth.
        max_results_per_source: How many records to pull from each source (1-15).
    """
    ctx = wrapper.context
    # The ceiling is configuration, not the model's choice: corpus size drives
    # the prompt size of every downstream stage, so it is the main cost lever.
    ceiling = get_settings().max_papers_per_source
    limit = max(1, min(int(max_results_per_source or ceiling), ceiling))
    chosen = [s for s in (sources or []) if s in SOURCES] or list(SOURCES)

    log_event(logger, "literature search", query=query, sources=chosen, limit=limit)

    # --- Parallel fan-out across the three databases ----------------------
    results = await asyncio.gather(
        *(SOURCES[name](query, limit) for name in chosen), return_exceptions=True
    )

    all_papers: list[Paper] = []
    report_lines: list[str] = []
    for name, result in zip(chosen, results):
        if isinstance(result, Exception):
            # One dead source must not kill the search; report and continue.
            message = f"{name} unavailable ({type(result).__name__})"
            ctx.record_error(message)
            log_event(logger, message, source=name, error=str(result))
            report_lines.append(f"{name}: FAILED - {message}")
            continue
        all_papers.extend(result)
        report_lines.append(f"{name}: {len(result)} records")

    if not all_papers:
        raise ExternalAPIError("search_literature", "all sources returned no results")

    added = ctx.add_papers(all_papers)
    ctx.costs.api_calls += len(chosen)
    header = "; ".join(report_lines)
    return _digest(ctx, all_papers, f"Query '{query}' [{header}]", added)


@function_tool
async def list_corpus(wrapper: RunContextWrapper[ResearchContext]) -> str:
    """List every paper currently in the shared corpus with its citation id.

    Use this before writing any claim, so that citations refer to real papers.
    """
    ctx = wrapper.context
    if not ctx.corpus:
        return "Corpus is empty. Run search_literature first."
    lines = [f"Corpus contains {len(ctx.corpus)} papers:"]
    ranked = sorted(
        ctx.corpus.values(), key=lambda p: (-p.citation_count, -p.year)
    )
    for paper in ranked:
        lines.append(
            f"- [{paper.paper_id}] {paper.title} ({paper.year}, {paper.venue or paper.source}, "
            f"{paper.citation_count} cites)"
        )
    return "\n".join(lines)


@function_tool
async def get_paper_details(
    wrapper: RunContextWrapper[ResearchContext], paper_ids: list[str]
) -> str:
    """Fetch full abstracts for specific papers by their corpus ids.

    Args:
        paper_ids: Citation ids exactly as shown by list_corpus, e.g. "arxiv:2301.00001".
    """
    ctx = wrapper.context
    papers = ctx.resolve_citations(paper_ids)
    missing = ctx.unknown_citations(paper_ids)
    if not papers:
        return f"None of {paper_ids} are in the corpus. Unknown ids: {missing}"

    blocks: list[str] = []
    for paper in papers:
        blocks.append(
            f"### [{paper.paper_id}] {paper.title}\n"
            f"Authors: {', '.join(paper.authors) or 'unknown'}\n"
            f"Year: {paper.year} | Venue: {paper.venue or 'n/a'} | "
            f"Citations: {paper.citation_count}\n"
            f"URL: {paper.url}\n"
            f"Abstract: {paper.abstract or '(not available from source)'}"
        )
    if missing:
        blocks.append(f"WARNING - not in corpus, do not cite: {missing}")
    return "\n\n".join(blocks)
