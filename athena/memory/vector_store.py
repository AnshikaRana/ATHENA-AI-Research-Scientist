"""A small, dependency-light vector store backing Athena's RAG.

Design choices worth defending in a review:

* **Embeddings are lexical by choice.** Every run - live or offline - uses a
  deterministic hashed bag-of-words embedding rather than a network embedding
  model. The whole RAG path is exercisable with no API key and no network, at
  the cost of semantic matching: retrieval is lexical overlap over unigrams and
  bigrams.
* **Storage is numpy + JSON on disk.** A FAISS/Chroma dependency buys nothing at
  the corpus sizes this system produces (tens to low hundreds of chunks) and
  costs portability.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from ..logging_setup import get_logger, log_event

logger = get_logger("rag")

MOCK_DIM = 512


@dataclass
class Chunk:
    """One retrievable unit of text plus the citation it came from."""

    chunk_id: str
    paper_id: str
    title: str
    text: str
    position: int


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# Very common English words carry no retrieval signal and, worse, create
# spurious matches in a hashed space. Dropping them is the cheapest possible
# approximation of IDF weighting.
_STOPWORDS = frozenset(
    """a an the of and or to in on for with by is are was were be been being
    this that these those it its as at from we our they their he she you i
    do does did how what which who whom when where why can could should would
    will shall may might must have has had not no but if then than so such
    about into over under more most other some any each""".split()
)


def _features(text: str) -> list[str]:
    """Unigrams plus bigrams.

    Bigrams matter more than they look: without them "expressive power" and
    "power analysis" share the token "power" and collide in the hashed space,
    which is exactly the kind of false match that makes offline retrieval
    embarrassing. The bigram features separate them.
    """
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _hashed_embedding(text: str, dim: int = MOCK_DIM) -> np.ndarray:
    """Deterministic bag-of-features embedding via signed feature hashing.

    Used when no API key is available. Cosine similarity over these vectors
    behaves like sublinear-tf-weighted lexical overlap over unigrams and
    bigrams. It is genuinely lexical, not semantic - paraphrases with no shared
    vocabulary will not match. That limitation is inherent to offline mode.
    """
    counts: dict[str, int] = {}
    for feature in _features(text):
        counts[feature] = counts.get(feature, 0) + 1
    if not counts:
        return np.zeros(dim, dtype=np.float32)

    vector = np.zeros(dim, dtype=np.float32)
    for feature, count in counts.items():
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        # Sublinear tf: a word appearing 10x is not 10x as important.
        vector[index] += sign * (1.0 + np.log(count))

    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


async def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts.

    Retrieval uses the deterministic hashed bag-of-features embedding below in
    every mode - live and offline alike. It behaves like sublinear-tf-weighted
    lexical overlap over unigrams and bigrams: genuinely lexical, not semantic,
    so paraphrases with no shared vocabulary will not match. That is the
    accepted tradeoff for keeping RAG key-free and network-free.
    """
    if not texts:
        return np.zeros((0, MOCK_DIM), dtype=np.float32)
    return np.vstack([_hashed_embedding(t) for t in texts])


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, *, size: int = 900, overlap: int = 150) -> list[str]:
    """Split on sentence boundaries into overlapping windows.

    Overlap matters: a claim that straddles a chunk boundary would otherwise be
    retrievable from neither chunk.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class VectorStore:
    """An in-memory index with optional persistence to ``data/vectors``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None
        self._indexed_papers: set[str] = set()

    # -- persistence ------------------------------------------------------

    @property
    def _dir(self) -> Path:
        return get_settings().vector_store_dir / self.name

    def save(self) -> None:
        if self.matrix is None or not self.chunks:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._dir / "vectors.npy", self.matrix)
        (self._dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> bool:
        vectors_path = self._dir / "vectors.npy"
        chunks_path = self._dir / "chunks.json"
        if not (vectors_path.exists() and chunks_path.exists()):
            return False
        self.matrix = np.load(vectors_path)
        self.chunks = [
            Chunk(**c) for c in json.loads(chunks_path.read_text(encoding="utf-8"))
        ]
        self._indexed_papers = {c.paper_id for c in self.chunks}
        return True

    # -- indexing ---------------------------------------------------------

    @property
    def indexed_paper_ids(self) -> set[str]:
        return set(self._indexed_papers)

    async def add_document(self, paper_id: str, title: str, text: str) -> int:
        """Chunk, embed and index one document. Returns chunks added."""
        if paper_id in self._indexed_papers:
            return 0
        pieces = chunk_text(text)
        if not pieces:
            return 0

        new_chunks = [
            Chunk(
                chunk_id=f"{paper_id}#{i}",
                paper_id=paper_id,
                title=title,
                text=piece,
                position=i,
            )
            for i, piece in enumerate(pieces)
        ]
        # Prefix the title so a chunk carries its own provenance into the vector.
        vectors = await embed_texts([f"{title}. {c.text}" for c in new_chunks])

        self.chunks.extend(new_chunks)
        self.matrix = (
            vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        )
        self._indexed_papers.add(paper_id)
        return len(new_chunks)

    # -- retrieval --------------------------------------------------------

    async def search(self, query: str, k: int = 6) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks by cosine similarity, best first."""
        if self.matrix is None or not self.chunks:
            return []
        query_vector = (await embed_texts([query]))[0]
        scores = self.matrix @ query_vector  # both sides are L2-normalised
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(top, key=lambda i: -scores[i])
        results = [(self.chunks[i], float(scores[i])) for i in ranked]
        log_event(
            logger,
            "rag search",
            query=query[:120],
            returned=len(results),
            top_score=round(results[0][1], 4) if results else None,
        )
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "documents": len(self._indexed_papers),
            "chunks": len(self.chunks),
            "dimensions": int(self.matrix.shape[1]) if self.matrix is not None else 0,
        }
