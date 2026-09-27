"""Generate the architecture diagram from the live agent team.

Written as a generator rather than a hand-drawn picture so the diagram cannot
drift from the code: it is built by introspecting ``build_team()``, so an agent
or tool added tomorrow appears in the diagram on the next run.

    python scripts/generate_diagram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from athena.agents import build_team  # noqa: E402
from athena.tools import ALL_TOOLS  # noqa: E402

DOCS = ROOT / "docs"

# Which tools belong to which capability group, for the diagram's tool layer.
TOOL_GROUPS = {
    "Literature APIs": ["search_literature", "commission_deep_sweep", "get_paper_details", "list_corpus"],
    "RAG": ["index_corpus", "rag_search"],
    "Documents (multi-modal)": ["ingest_pdf", "describe_figure", "transcribe_audio"],
    "Analysis": ["statistical_power_analysis", "run_python_analysis", "analyze_dataset", "compare_groups"],
    "Citations": ["format_citations", "verify_citations"],
    "Memory": ["remember_finding", "recall_memory"],
    "Artefacts (approval-gated)": ["save_research_artifact"],
}

STAGE_SEQUENCE = [
    ("Principal Investigator", "ResearchPlan"),
    ("Literature Review Agent", "LiteratureReview"),
    ("Knowledge Synthesist", "KnowledgeSynthesis"),
    ("Research Gap Analyst", "GapAnalysis"),
    ("Hypothesis Generator", "HypothesisSet"),
    ("Experiment Planner", "ExperimentDesign"),
    ("Citation Manager", "CitationBundle"),
    ("Scientific Writer", "ResearchReport"),
]


def _node_id(name: str) -> str:
    """Stable, deterministic node id.

    Deliberately not derived from ``hash()``: Python randomises string hashing
    per process, which would make this generator emit a different diagram on
    every run and produce noisy diffs.
    """
    initials = "".join(part[0] for part in name.split() if part)
    return f"{initials}{len(name)}".upper()


def build_mermaid() -> str:
    lines = [
        "```mermaid",
        "flowchart TB",
        "    %% ---------- Entry ----------",
        f'    USER(["Researcher"]) --> {_node_id("Principal Investigator")}',
        "",
        "    %% ---------- Deterministic study pipeline ----------",
        "    subgraph PIPELINE[\"Study pipeline - deterministic sequence\"]",
        "        direction TB",
    ]
    for name, output in STAGE_SEQUENCE:
        lines.append(f'        {_node_id(name)}["{name}<br/><i>{output}</i>"]')
    lines.append("")
    ordered = [_node_id(name) for name, _ in STAGE_SEQUENCE]
    for a, b in zip(ordered, ordered[1:]):
        lines.append(f"        {a} --> {b}")
    lines += [
        "    end",
        "",
        "    %% ---------- Shared state ----------",
        "    subgraph STATE[\"Memory\"]",
        "        direction LR",
        "        CTX[(\"ResearchContext<br/>working memory\")]",
        "        SESS[(\"SQLiteSession<br/>conversation\")]",
        "        LTM[(\"Long-term memory<br/>cross-session\")]",
        "        VEC[(\"Vector store<br/>RAG index\")]",
        "    end",
        "",
        "    %% ---------- Tools ----------",
        "    subgraph TOOLS[\"Tool layer\"]",
        "        direction LR",
    ]
    for group, tools in TOOL_GROUPS.items():
        group_id = "TG" + "".join(ch for ch in group if ch.isalnum()).upper()[:14]
        label = f"{len(tools)} tool" + ("s" if len(tools) != 1 else "")
        lines.append(f'        {group_id}["{group}<br/><i>{label}</i>"]')
    lines += [
        "    end",
        "",
        "    PIPELINE --> TOOLS",
        "    PIPELINE <--> STATE",
        "    TOOLS <--> STATE",
        "",
        "    %% ---------- External systems ----------",
        "    TOOLS --> EXT[\"arXiv · OpenAlex · Crossref · OpenAI\"]",
        "",
        "    %% ---------- Human in the loop ----------",
        f'    {_node_id("Scientific Writer")} ==>|"gated tool call"| APPROVAL{{"Human approval"}}',
        "    APPROVAL -->|approve| ARTIFACT[(\"Research dossier\")]",
        "    APPROVAL -->|defer| CKPT[(\"Checkpoint<br/>resume later\")]",
        "",
        "    classDef agent fill:#e8f0fe,stroke:#3367d6,stroke-width:1px;",
        "    classDef store fill:#fef7e0,stroke:#f9ab00,stroke-width:1px;",
        "    classDef human fill:#fce8e6,stroke:#d93025,stroke-width:2px;",
        f"    class {','.join(ordered)} agent;",
        "    class CTX,SESS,LTM,VEC,ARTIFACT,CKPT store;",
        "    class APPROVAL,USER human;",
        "```",
    ]
    return "\n".join(lines)


def build_handoff_table() -> str:
    team = build_team()
    rows = ["| Agent | Structured output | Tools | Guardrails |", "|---|---|---|---|"]
    for spec in team.describe():
        guardrails = spec["input_guardrails"] + spec["output_guardrails"]
        rows.append(
            f"| {spec['name']} | `{spec['output_type']}` | {len(spec['tools'])} | "
            f"{', '.join(guardrails) or '—'} |"
        )
    return "\n".join(rows)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    team = build_team()

    content = f"""# Architecture Diagram

*Generated by `scripts/generate_diagram.py` from the live agent definitions, so
it cannot drift from the code.*

{len(team.all_agents)} agents · {len(ALL_TOOLS)} tools · {len(STAGE_SEQUENCE)} pipeline stages

{build_mermaid()}

## Reading the diagram

- **Solid arrows** are the deterministic pipeline: stage order is fixed in code
  because research stages have hard dependencies.
- **The `search_literature` tool** fans out across three databases concurrently -
  the one place with genuine parallelism.
- **The citation-integrity output guardrail** rejects any `paper_id` an agent
  cites that is not in the retrieved corpus - fabrication is blocked
  deterministically, not by a reviewer.
- **The red path** is the human-in-the-loop gate. Deferring checkpoints the run
  so the decision can be made in a different process, or a week later.

## Agent roster

{build_handoff_table()}
"""
    output = DOCS / "03_architecture_diagram.md"
    output.write_text(content, encoding="utf-8")
    print(f"Wrote {output} ({len(content):,} chars)")


if __name__ == "__main__":
    main()
