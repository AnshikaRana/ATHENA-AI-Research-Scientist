"""Retrieval and memory: the parts that must rank correctly, not just run."""

from __future__ import annotations

from athena.memory.long_term import LongTermMemory
from athena.memory.vector_store import VectorStore, chunk_text

GNN_TEXT = (
    "Message passing neural networks are bounded by the Weisfeiler-Lehman test. "
    "This limits their ability to count triangles and detect cycles in graphs. "
    "Subgraph aggregation exceeds 1-WL expressive power at higher cost."
)
VIT_TEXT = (
    "Vision transformers split an image into patches and apply self attention. "
    "They require large scale pretraining to outperform convolutional networks."
)


def test_chunking_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} contains filler words." for i in range(120))
    chunks = chunk_text(text, size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)  # size + overlap slack
    assert chunk_text("") == []
    assert chunk_text("Short.") == ["Short."]


async def test_retrieval_ranks_the_right_document_first(tmp_settings):
    store = VectorStore("test_rank")
    await store.add_document("arxiv:1", "GNN Expressivity", GNN_TEXT)
    await store.add_document("arxiv:2", "Vision Transformers", VIT_TEXT)

    hits = await store.search("expressive power of message passing on graphs", k=2)
    assert hits[0][0].paper_id == "arxiv:1"
    assert hits[0][1] > hits[1][1]

    hits = await store.search("image patches and self attention", k=2)
    assert hits[0][0].paper_id == "arxiv:2"


async def test_indexing_is_idempotent(tmp_settings):
    store = VectorStore("test_idem")
    first = await store.add_document("arxiv:1", "GNN", GNN_TEXT)
    second = await store.add_document("arxiv:1", "GNN", GNN_TEXT)
    assert first > 0
    assert second == 0


async def test_vector_store_persists_across_instances(tmp_settings):
    store = VectorStore("test_persist")
    await store.add_document("arxiv:1", "GNN", GNN_TEXT)
    store.save()

    reloaded = VectorStore("test_persist")
    assert reloaded.load() is True
    assert reloaded.stats()["chunks"] == store.stats()["chunks"]
    assert (await reloaded.search("weisfeiler lehman", k=1))[0][0].paper_id == "arxiv:1"


async def test_empty_store_returns_no_hits(tmp_settings):
    assert await VectorStore("empty").search("anything", k=3) == []


async def test_long_term_memory_recalls_across_sessions(tmp_settings):
    memory = LongTermMemory(tmp_settings.long_term_memory_db)
    await memory.remember(
        session_id="s1", run_id="r1", kind="finding", topic="GNN expressivity",
        content="Message passing GNNs are bounded by the 1-WL test on expressive power.",
    )
    await memory.remember(
        session_id="s2", run_id="r2", kind="preference", topic="method style",
        content="User prefers experiment designs with explicit statistical power analysis.",
    )

    hits = await memory.recall("expressive power of message passing graphs", k=2)
    assert hits[0].topic == "GNN expressivity"

    hits = await memory.recall("statistical power analysis in experiments", k=2)
    assert hits[0].topic == "method style"

    stats = memory.stats()
    assert stats["total"] == 2
    assert stats["sessions"] == 2


async def test_recall_on_empty_memory_is_safe(tmp_settings):
    memory = LongTermMemory(tmp_settings.long_term_memory_db)
    assert await memory.recall("anything") == []
