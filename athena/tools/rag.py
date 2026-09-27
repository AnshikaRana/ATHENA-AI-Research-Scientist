"""Retrieval-augmented generation over the run's paper corpus.

The Synthesist agent is forbidden from writing claims out of its own parametric
knowledge. It must call ``rag_search`` and cite the chunks that come back. These
two tools are what make that possible.
"""

from __future__ import annotations

from athena.runtime import RunContextWrapper, function_tool

from ..context import ResearchContext
from ..logging_setup import get_logger, log_event
from ..memory.vector_store import VectorStore

logger = get_logger("tools.rag")


def _ensure_store(ctx: ResearchContext) -> VectorStore:
    if ctx.vector_store is None:
        ctx.vector_store = VectorStore(name=ctx.run_id)
    return ctx.vector_store


@function_tool(failure_error_function=lambda ctx, err: f"Indexing failed: {err}")
async def index_corpus(wrapper: RunContextWrapper[ResearchContext]) -> str:
    """Embed every paper in the corpus into the vector index for retrieval.

    Call this once after literature search and before asking any grounded
    question. Papers already indexed are skipped, so calling it again is cheap.
    """
    ctx = wrapper.context
    store = _ensure_store(ctx)

    if not ctx.corpus:
        return "Corpus is empty - nothing to index. Run search_literature first."

    added_chunks = 0
    indexed_docs = 0
    skipped_no_text = []
    for paper in ctx.corpus.values():
        if paper.paper_id in store.indexed_paper_ids:
            continue
        # The abstract is the retrievable body; papers without one (common in
        # Crossref records) contribute title-only text rather than nothing.
        body = paper.abstract or paper.title
        if not body.strip():
            skipped_no_text.append(paper.paper_id)
            continue
        count = await store.add_document(paper.paper_id, paper.title, body)
        added_chunks += count
        indexed_docs += 1 if count else 0

    store.save()
    stats = store.stats()
    log_event(logger, "corpus indexed", **stats, newly_indexed=indexed_docs)
    message = (
        f"Indexed {indexed_docs} new documents ({added_chunks} chunks). "
        f"Index now holds {stats['documents']} documents / {stats['chunks']} chunks."
    )
    if skipped_no_text:
        message += f" Skipped {len(skipped_no_text)} records with no text."
    return message


@function_tool(failure_error_function=lambda ctx, err: f"Retrieval failed: {err}")
async def rag_search(
    wrapper: RunContextWrapper[ResearchContext], question: str, top_k: int
) -> str:
    """Retrieve the passages most relevant to a question, with citation ids.

    Returns verbatim excerpts. Ground every claim you write in these excerpts
    and cite the paper_id shown in brackets.

    Args:
        question: A specific question, e.g. "what limits message passing expressivity?".
        top_k: How many passages to retrieve (3-10 is usually right).
    """
    ctx = wrapper.context
    store = _ensure_store(ctx)
    if not store.chunks:
        return "The index is empty. Call index_corpus first."

    k = max(1, min(int(top_k or 6), 12))
    hits = await store.search(question, k=k)
    if not hits:
        return f"No passages retrieved for: {question}"

    blocks = [f"Retrieved {len(hits)} passages for: {question}\n"]
    for rank, (chunk, score) in enumerate(hits, start=1):
        blocks.append(
            f"[{rank}] paper_id={chunk.paper_id} | relevance={score:.3f}\n"
            f"    Title: {chunk.title}\n"
            f"    Excerpt: {chunk.text}"
        )
    blocks.append(
        "\nCite only the paper_id values listed above. If none of these passages "
        "support the claim you want to make, say the evidence is missing instead."
    )
    return "\n".join(blocks)
