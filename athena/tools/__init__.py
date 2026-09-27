"""The tool layer.

Eighteen tools across eight capability groups. Agents receive only the subset
their role needs - a Scientific Writer with a literature search tool would
wander off and search instead of writing.
"""

from .analysis import run_python_analysis, statistical_power_analysis
from .artifacts import commission_deep_sweep, save_research_artifact
from .citations import format_citations, verify_citations
from .datasets import analyze_dataset, compare_groups
from .documents import describe_figure, ingest_pdf, transcribe_audio
from .literature import get_paper_details, list_corpus, search_literature
from .memory_tools import recall_memory, remember_finding
from .rag import index_corpus, rag_search

# Grouped for readability at the agent definitions.
LITERATURE_TOOLS = [search_literature, list_corpus, get_paper_details, commission_deep_sweep]
RAG_TOOLS = [index_corpus, rag_search, list_corpus]
MEMORY_TOOLS = [remember_finding, recall_memory]
DOCUMENT_TOOLS = [ingest_pdf, describe_figure, transcribe_audio]
ANALYSIS_TOOLS = [statistical_power_analysis, run_python_analysis]
DATASET_TOOLS = [analyze_dataset, compare_groups]
CITATION_TOOLS = [format_citations, verify_citations, list_corpus]
ARTIFACT_TOOLS = [save_research_artifact]

ALL_TOOLS = [
    search_literature, list_corpus, get_paper_details, commission_deep_sweep,
    index_corpus, rag_search,
    remember_finding, recall_memory,
    ingest_pdf, describe_figure, transcribe_audio,
    statistical_power_analysis, run_python_analysis,
    analyze_dataset, compare_groups,
    format_citations, verify_citations,
    save_research_artifact,
]

__all__ = [
    "search_literature", "list_corpus", "get_paper_details", "commission_deep_sweep",
    "index_corpus", "rag_search", "remember_finding", "recall_memory",
    "ingest_pdf", "describe_figure", "transcribe_audio",
    "statistical_power_analysis", "run_python_analysis",
    "analyze_dataset", "compare_groups",
    "format_citations", "verify_citations",
    "save_research_artifact",
    "LITERATURE_TOOLS", "RAG_TOOLS", "MEMORY_TOOLS", "DOCUMENT_TOOLS",
    "ANALYSIS_TOOLS", "DATASET_TOOLS", "CITATION_TOOLS", "ARTIFACT_TOOLS",
    "ALL_TOOLS",
]
