"""Structured logging.

Two sinks, deliberately:

* a human-readable Rich console stream for live runs, and
* a machine-readable JSONL file per run, which the Streamlit UI tails to show
  the agent trace and which doubles as the audit log for the report.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

_console = Console(stderr=True)
_RUN_ID: str | None = None
_JSONL_PATH: Path | None = None


class JsonlHandler(logging.Handler):
    """Append every record to a newline-delimited JSON file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Anything passed via `extra={"event": ...}` is merged in verbatim.
            for key, value in getattr(record, "athena_extra", {}).items():
                payload[key] = value
            if record.exc_info:
                payload["exception"] = self.format(record)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")
        except Exception:  # logging must never take down the run
            self.handleError(record)


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def configure_logging(run_id: str | None = None, *, verbose: bool = False) -> str:
    """Install handlers for a run and return the run id."""
    global _RUN_ID, _JSONL_PATH

    settings = get_settings()
    _RUN_ID = run_id or new_run_id()
    _JSONL_PATH = settings.log_dir / f"{_RUN_ID}.jsonl"

    root = logging.getLogger("athena")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    root.propagate = False

    console_handler = RichHandler(
        console=_console, show_path=False, rich_tracebacks=True, markup=False
    )
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(console_handler)

    jsonl_handler = JsonlHandler(_JSONL_PATH)
    jsonl_handler.setLevel(logging.DEBUG)
    root.addHandler(jsonl_handler)

    return _RUN_ID


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"athena.{name}")


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Log a message plus structured fields that land in the JSONL sink."""
    logger.log(level, message, extra={"athena_extra": {"run_id": _RUN_ID, **fields}})


def current_log_path() -> Path | None:
    return _JSONL_PATH


def read_run_events(run_id: str) -> list[dict[str, Any]]:
    """Read back a run's JSONL trace (used by the UI and by tests)."""
    path = get_settings().log_dir / f"{run_id}.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
