"""Shared fixtures. Every test runs offline: no API key, no spend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force offline mode before athena.config is imported anywhere.
os.environ["ATHENA_MOCK_MODE"] = "true"
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_API_KEY_2", None)


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Redirect all persistence into a temp directory for isolation."""
    from athena.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("ATHENA_DATA_DIR", str(tmp_path))
    settings = get_settings()
    settings.data_dir = tmp_path
    settings.session_db = tmp_path / "sessions.db"
    settings.long_term_memory_db = tmp_path / "ltm.db"
    settings.vector_store_dir = tmp_path / "vectors"
    settings.artifacts_dir = tmp_path / "artifacts"
    settings.uploads_dir = tmp_path / "uploads"
    settings.log_dir = tmp_path / "logs"
    settings.ensure_dirs()
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def sample_papers():
    from athena.schemas import Paper

    return [
        Paper(
            paper_id="arxiv:2205.13328",
            title="How Powerful are K-hop Message Passing Graph Neural Networks",
            authors=["Jiarui Feng", "Yixin Chen"],
            year=2022,
            venue="NeurIPS",
            abstract=(
                "We analyse the expressive power of K-hop message passing and show "
                "it is strictly more powerful than 1-hop message passing but still "
                "bounded above by the 3-WL test on regular graphs."
            ),
            url="https://arxiv.org/abs/2205.13328",
            source="arxiv",
            citation_count=210,
        ),
        Paper(
            paper_id="doi:10.1109/tnn.2008.2005605",
            title="The Graph Neural Network Model",
            authors=["Franco Scarselli", "Marco Gori"],
            year=2008,
            venue="IEEE Transactions on Neural Networks",
            abstract="We introduce a neural network model for processing graph structured data.",
            url="https://doi.org/10.1109/tnn.2008.2005605",
            source="openalex",
            citation_count=9686,
        ),
    ]
