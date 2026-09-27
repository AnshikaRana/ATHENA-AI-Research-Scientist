"""Multi-modal ingestion: PDFs, figures and audio.

A research assistant that can only read a search API is not much of a research
assistant. These tools let a user drop in the actual material they are working
from - a paper PDF, a figure from a poster, a recorded seminar - and have it
join the same corpus and RAG index as the retrieved literature.

PDF parsing is fully local and always works. Figure analysis (OpenAI vision) and
audio transcription (OpenAI Whisper) need an API key, so offline they return an
explicit, clearly labelled placeholder rather than pretending to have read the
file.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

from athena.runtime import RunContextWrapper, function_tool
from pypdf import PdfReader

from ..config import get_settings
from ..context import ResearchContext
from ..errors import DocumentParseError
from ..logging_setup import get_logger, log_event
from ..schemas import Paper

logger = get_logger("tools.documents")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm"}
MAX_PDF_PAGES = 40


def _slug(text: str) -> str:
    """A citable id fragment: lowercase, no spaces, bounded length.

    The id goes into the corpus and agents cite it verbatim. A raw filename
    stem contains spaces, and the citation pattern stops at whitespace - so
    "upload:LitSearch A Retrieval Benchmark" was only ever extracted as
    "upload:LitSearch", which matched nothing and was reported as fabricated.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return (cleaned[:48].rstrip("_")) or "document"


def _resolve(path_str: str) -> Path:
    """Resolve a user-supplied path, allowing bare filenames in the uploads dir."""
    settings = get_settings()
    path = Path(path_str).expanduser()
    if not path.is_absolute() and not path.exists():
        candidate = settings.uploads_dir / path.name
        if candidate.exists():
            return candidate
    return path


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def extract_pdf_text(path: Path, max_pages: int = MAX_PDF_PAGES) -> tuple[str, str, int]:
    """Return (title, body_text, pages_read). Pure local parsing, no API."""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises a wide variety of errors
        raise DocumentParseError("ingest_pdf", f"cannot open {path.name}: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many PDFs are "encrypted" with an empty password
        except Exception as exc:
            raise DocumentParseError(
                "ingest_pdf", f"{path.name} is password protected"
            ) from exc

    pages = reader.pages[:max_pages]
    parts: list[str] = []
    for page in pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")  # one bad page should not lose the document

    body = _dehyphenate("\n".join(parts).strip())
    if not body:
        raise DocumentParseError(
            "ingest_pdf",
            f"{path.name} yielded no extractable text - it is probably a scan. "
            "Run OCR on it first.",
        )

    title = _derive_title(reader, body, path)
    return title, body, len(pages)


def _derive_title(reader: PdfReader, body: str, path: Path) -> str:
    """Best-effort title, in decreasing order of reliability.

    PDF ``/Title`` metadata is authoritative but frequently blank or set to a
    build-tool artefact ("Microsoft Word - paper.doc"). When it is unusable we
    fall back to the first substantial line of extracted text, which for a
    paper is the title line, and only then to the filename.
    """
    try:
        metadata_title = str((reader.metadata or {}).get("/Title", "") or "").strip()
    except Exception:
        metadata_title = ""

    junk_markers = ("microsoft word", "untitled", ".doc", ".tex", ".dvi")
    if metadata_title and not any(m in metadata_title.lower() for m in junk_markers):
        return metadata_title

    for line in body.splitlines():
        candidate = line.strip()
        # A title line: long enough to be meaningful, short enough not to be a
        # paragraph, and not an arXiv stamp or page furniture.
        if 15 <= len(candidate) <= 200 and not candidate.lower().startswith("arxiv:"):
            return candidate

    return path.stem.replace("_", " ")


def _dehyphenate(text: str) -> str:
    """Rejoin words that the PDF's line layout split ("re-\\nquiring").

    Only a hyphen followed by a newline and a lowercase letter is joined, so
    genuine hyphenated tokens that happen to wrap ("ISO/IEC-\\n19794") survive.
    """
    return re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)


_ARXIV_STAMP = re.compile(
    r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?\s*\[[^\]]+\]\s*(\d{1,2}\s+\w+\s+(\d{4}))?"
)
_NAME = re.compile(r"^[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){1,3}$")


def _derive_metadata(body: str, title: str) -> tuple[list[str], int, str, str]:
    """Best-effort (authors, year, venue, url) from the text of a paper.

    A citation built from an uploaded paper used to read "Unknown (n.d.)", which
    undermined the one reference a document-grounded report depends on. The
    byline and the arXiv margin stamp are reliable enough in typeset papers to
    do better; anything not found stays empty rather than guessed.
    """
    authors: list[str] = []
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    try:
        start = lines.index(title) + 1
    except ValueError:
        start = 1
    for line in lines[start : start + 3]:
        names = [n.strip() for n in re.split(r",|\band\b", line) if n.strip()]
        if len(names) >= 2 and all(_NAME.match(n) for n in names):
            authors = names
            break

    year, venue, url = 0, "", ""
    stamp = _ARXIV_STAMP.search(body)
    if stamp:
        arxiv_id = stamp.group(1)
        venue = f"arXiv preprint arXiv:{arxiv_id}"
        url = f"https://arxiv.org/abs/{arxiv_id}"
        if stamp.group(3):
            year = int(stamp.group(3))
        else:
            year = 2000 + int(arxiv_id[:2])
    return authors, year, venue, url


async def ingest_pdf_file(ctx: ResearchContext, file_path: str) -> tuple[Paper, int, str]:
    """Parse a PDF into the corpus and the RAG index. Returns (paper, chunks, body).

    Shared by the ``ingest_pdf`` tool and the pipeline's document-grounded mode,
    which ingests deterministically instead of asking a model to do it.
    """
    path = _resolve(file_path)
    if not path.exists():
        raise DocumentParseError("ingest_pdf", f"no such file: {path}")

    title, body, pages_read = extract_pdf_text(path)
    paper_id = f"upload:{_slug(path.stem)}"
    authors, year, venue, url = _derive_metadata(body, title)

    paper = Paper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue or "user upload",
        # The abstract slot carries the head of the document, which is where
        # title/abstract/intro live; the full body goes to RAG.
        abstract=body[:4000],
        url=url or str(path),
        source="user_upload",
        citation_count=0,
    )
    ctx.add_papers([paper], force=True)
    ctx.attachments.append(
        {"kind": "pdf", "path": str(path), "title": title, "pages": pages_read}
    )
    ctx.source_texts[paper_id] = body

    # Index the *whole* body, not just the truncated abstract field.
    from ..memory.vector_store import VectorStore

    if ctx.vector_store is None:
        ctx.vector_store = VectorStore(name=ctx.run_id)
    chunks = await ctx.vector_store.add_document(paper_id, title, body)
    ctx.vector_store.save()

    log_event(
        logger, "pdf ingested", path=str(path), pages=pages_read, chunks=chunks
    )
    return paper, chunks, body


@function_tool(failure_error_function=lambda ctx, err: f"PDF ingestion failed: {err}")
async def ingest_pdf(wrapper: RunContextWrapper[ResearchContext], file_path: str) -> str:
    """Read a local PDF, add it to the corpus and index it for retrieval.

    Args:
        file_path: Path to the PDF. A bare filename is looked up in data/uploads.
    """
    paper, chunks, body = await ingest_pdf_file(wrapper.context, file_path)
    pages = wrapper.context.attachments[-1]["pages"]
    return (
        f"Ingested '{paper.title}' ({pages} pages, {len(body):,} chars) as "
        f"[{paper.paper_id}] and indexed {chunks} chunks. It is now searchable "
        f"via rag_search."
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@function_tool(failure_error_function=lambda ctx, err: f"Figure analysis failed: {err}")
async def describe_figure(
    wrapper: RunContextWrapper[ResearchContext], file_path: str, question: str
) -> str:
    """Interpret a scientific figure, chart or diagram image.

    Args:
        file_path: Path to a .png/.jpg/.webp image.
        question: What to determine, e.g. "what trend does this plot show?".
    """
    ctx = wrapper.context
    settings = get_settings()
    path = _resolve(file_path)
    if not path.exists():
        raise DocumentParseError("describe_figure", f"no such file: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise DocumentParseError(
            "describe_figure", f"{path.suffix} is not a supported image type"
        )

    ctx.attachments.append({"kind": "image", "path": str(path), "question": question})

    if not settings.live:
        return (
            f"[OFFLINE MODE] Figure '{path.name}' was registered as an attachment but "
            f"not visually analysed, because no OPENAI_API_KEY_2 is configured. "
            f"Question was: {question}. Do not invent findings from this figure."
        )

    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.vision_model,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific figure analyst. Describe only what is "
                    "visibly present: axes, units, series, trends and magnitudes. "
                    "If something is illegible, say so rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            },
        ],
    )
    ctx.costs.api_calls += 1
    description = response.choices[0].message.content or ""
    log_event(logger, "figure analysed", path=str(path), chars=len(description))
    return f"Figure analysis of {path.name}:\n{description}"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


@function_tool(failure_error_function=lambda ctx, err: f"Transcription failed: {err}")
async def transcribe_audio(
    wrapper: RunContextWrapper[ResearchContext], file_path: str
) -> str:
    """Transcribe a recorded talk, interview or voice note into text.

    Args:
        file_path: Path to an audio file (.mp3, .wav, .m4a, .webm).
    """
    ctx = wrapper.context
    settings = get_settings()
    path = _resolve(file_path)
    if not path.exists():
        raise DocumentParseError("transcribe_audio", f"no such file: {path}")
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise DocumentParseError(
            "transcribe_audio", f"{path.suffix} is not a supported audio type"
        )

    ctx.attachments.append({"kind": "audio", "path": str(path)})

    if not settings.live:
        return (
            f"[OFFLINE MODE] Audio '{path.name}' registered but not transcribed - "
            "no OPENAI_API_KEY_2 configured. Do not invent its contents."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    with path.open("rb") as handle:
        result = await client.audio.transcriptions.create(
            model=settings.transcription_model, file=handle
        )
    ctx.costs.api_calls += 1
    text = getattr(result, "text", "") or ""
    log_event(logger, "audio transcribed", path=str(path), chars=len(text))
    return f"Transcript of {path.name} ({len(text):,} chars):\n\n{text}"
