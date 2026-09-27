"""Central configuration for the Athena AI Research Scientist system.

All runtime knobs live here so that the rest of the codebase never reads
``os.environ`` directly. Settings are loaded from environment variables and an
optional ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Runtime settings for the whole application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="ATHENA_",
        extra="ignore",
    )

    # --- Model access -----------------------------------------------------
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY_2")
    mock_mode: bool = Field(
        default=False,
        description="Run entirely offline against recorded fixtures. No API spend.",
    )

    # Two tiers keep cost down: a cheap model for high-volume mechanical work
    # (routing, extraction) and a stronger model for the reasoning-heavy agents.
    fast_model: str = "gpt-4.1-mini"
    reasoning_model: str = "gpt-4.1-mini"
    vision_model: str = "gpt-4.1-mini"
    transcription_model: str = "whisper-1"

    # --- Budget guardrails ------------------------------------------------
    max_turns: int = 12
    max_papers_per_source: int = 5
    # Total corpus ceiling. Agents run many searches, so a per-search cap
    # alone does not bound the corpus - and corpus size drives the prompt
    # size of every downstream stage.
    max_corpus_papers: int = 12
    # Hard spending ceiling for one run, in USD. No new stage starts once the
    # run's metered spend reaches it.
    max_run_usd: float = 1.00
    request_timeout_seconds: float = 30.0
    max_tool_retries: int = 3

    # --- Storage ----------------------------------------------------------
    data_dir: Path = DATA_DIR
    session_db: Path = DATA_DIR / "sessions" / "sessions.db"
    long_term_memory_db: Path = DATA_DIR / "sessions" / "long_term_memory.db"
    vector_store_dir: Path = DATA_DIR / "vectors"
    artifacts_dir: Path = DATA_DIR / "artifacts"
    uploads_dir: Path = DATA_DIR / "uploads"
    log_dir: Path = DATA_DIR / "logs"

    # --- Behaviour --------------------------------------------------------
    require_human_approval: bool = True
    contact_email: str = "research@athena.local"  # polite UA for public APIs

    @property
    def live(self) -> bool:
        """True when we should actually hit the OpenAI API."""
        return not self.mock_mode and bool(self.openai_api_key)

    def ensure_dirs(self) -> None:
        for path in (
            self.session_db.parent,
            self.vector_store_dir,
            self.artifacts_dir,
            self.uploads_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (used by the Streamlit UI)."""
    get_settings.cache_clear()
    return get_settings()
