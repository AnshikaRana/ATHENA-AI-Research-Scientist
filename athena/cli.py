"""Command-line interface for Athena.

    python -m athena.cli study "your research question"
    python -m athena.cli resume <run_id>
    python -m athena.cli agents
    python -m athena.cli memory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from athena.runtime import ToolApprovalItem
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agents import build_team
from .config import get_settings
from .context import ResearchContext
from .errors import ApprovalDeferred
from .memory.long_term import LongTermMemory
from .pipeline import ResearchPipeline
from .schemas import ResearchReport

console = Console()


# ---------------------------------------------------------------------------
# Human approval, interactively
# ---------------------------------------------------------------------------


async def interactive_approval(item: ToolApprovalItem) -> tuple[str, str]:
    """Ask the operator to approve, reject or defer a gated tool call."""
    try:
        arguments = json.loads(item.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        arguments = {"raw": str(item.arguments)[:400]}

    preview = {
        key: (value[:300] + "..." if isinstance(value, str) and len(value) > 300 else value)
        for key, value in arguments.items()
    }

    console.print(
        Panel(
            f"[bold]Tool:[/bold] {item.name}\n\n"
            f"[bold]Arguments:[/bold]\n{json.dumps(preview, indent=2)}",
            title="[yellow]Human approval required[/yellow]",
            border_style="yellow",
        )
    )
    answer = console.input(
        "\n[bold]Approve?[/bold] [green]y[/green]es / [red]n[/red]o / "
        "[cyan]d[/cyan]efer (decide later): "
    ).strip().lower()

    if answer.startswith("d"):
        raise ApprovalDeferred(item.name, "run checkpointed")
    if answer.startswith("y"):
        return True, "approved interactively"
    note = console.input("[dim]Reason for rejection (optional): [/dim]").strip()
    return False, note or "rejected interactively"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_summary(pipeline: ResearchPipeline, ctx: ResearchContext) -> None:
    table = Table(title="Pipeline stages", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Status")
    table.add_column("Time", justify="right")
    for stage in pipeline.stages:
        table.add_row(
            stage.stage.value,
            stage.agent,
            "[green]ok[/green]" if stage.ok else "[red]failed[/red]",
            f"{stage.duration_s:.1f}s",
        )
    console.print(table)

    summary = ctx.summary()
    console.print(
        Panel(
            f"Corpus: [bold]{summary['corpus_size']}[/bold] papers | "
            f"Indexed chunks: [bold]{summary['indexed_chunks']}[/bold]\n"
            f"Approvals: [bold]{summary['approvals']}[/bold] | "
            f"Errors: [bold]{summary['errors']}[/bold] | "
            f"Tokens: [bold]{summary['tokens']:,}[/bold] | "
            f"Est. cost: [bold]${summary['estimated_usd']}[/bold]",
            title="Run summary",
            border_style="blue",
        )
    )

    if ctx.errors:
        console.print("[yellow]Recorded issues:[/yellow]")
        for error in ctx.errors[:8]:
            console.print(f"  - {error[:160]}")


def render_report(report: ResearchReport) -> None:
    console.print(Panel(report.title, style="bold green"))
    console.print(Markdown(f"**Executive summary**\n\n{report.executive_summary}"))
    for section in report.sections:
        console.print(Markdown(f"## {section.heading}\n\n{section.body_markdown}"))
    if report.references:
        console.print(Markdown("## References\n" + "\n".join(f"- {r}" for r in report.references)))
    console.print(f"\n[dim]Overall confidence: {report.confidence.value}[/dim]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_study(args: argparse.Namespace) -> int:
    pipeline = ResearchPipeline(
        session_id=args.session,
        approval_handler=None if args.yes else interactive_approval,
        verbose=args.verbose,
    )
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]{args.question}[/bold]\n\n"
            f"Mode: {'[green]live[/green]' if settings.live else '[yellow]offline (fixtures)[/yellow]'} | "
            f"Session: {args.session} | Run: {pipeline.run_id}",
            title="Athena - AI Research Scientist",
            border_style="cyan",
        )
    )

    try:
        ctx = await pipeline.run_study(args.question, attachments=args.attach or None)
    except ApprovalDeferred as exc:
        console.print(
            f"\n[yellow]Approval deferred.[/yellow] Resume later with:\n"
            f"  python -m athena.cli resume {pipeline.run_id}"
        )
        console.print(f"[dim]{exc}[/dim]")
        return 2

    render_summary(pipeline, ctx)
    if ctx.report:
        render_report(ctx.report)
    manifest = pipeline.save_manifest(ctx)
    console.print(f"\n[dim]Manifest: {manifest}[/dim]")
    console.print(f"[dim]Trace log: data/logs/{pipeline.run_id}.jsonl[/dim]")
    return 0 if ctx.report else 1


async def cmd_resume(args: argparse.Namespace) -> int:
    pipeline = ResearchPipeline(session_id=args.session, run_id=args.run_id)
    ctx = pipeline.load_checkpointed_context()
    if ctx is None:
        console.print(f"[red]No checkpoint found for run {args.run_id}[/red]")
        return 1

    console.print(
        Panel(
            f"Question: {ctx.research_question}\n"
            f"Corpus: {len(ctx.corpus)} papers | Stage reached: {ctx.stage.value}",
            title=f"Resuming {args.run_id}",
            border_style="cyan",
        )
    )
    result = await pipeline.resume_from_checkpoint(
        pipeline.team.scientific_writer, ctx, approve=not args.reject, note=args.note
    )
    if result is None:
        console.print("[red]Nothing to resume.[/red]")
        return 1

    output = result.final_output
    if isinstance(output, ResearchReport):
        render_report(output)
    else:
        console.print(str(output)[:4000])
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    team = build_team()
    table = Table(title=f"Athena research team ({len(team.all_agents)} agents)")
    table.add_column("Agent", style="cyan")
    table.add_column("Output", style="magenta")
    table.add_column("Tools", justify="right")
    table.add_column("Handoffs", justify="right")
    table.add_column("Guardrails", justify="right")
    for spec in team.describe():
        table.add_row(
            str(spec["name"]),
            str(spec["output_type"]),
            str(len(spec["tools"])),
            str(len(spec["handoffs"])),
            str(len(spec["input_guardrails"]) + len(spec["output_guardrails"])),
        )
    console.print(table)

    if args.json:
        console.print_json(json.dumps(team.describe(), indent=2))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    memory = LongTermMemory()
    stats = memory.stats()
    console.print(Panel(json.dumps(stats, indent=2), title="Long-term memory", border_style="blue"))

    if args.query:
        records = asyncio.run(memory.recall(args.query, k=args.limit))
        table = Table(title=f"Recall: {args.query}")
        table.add_column("Score", justify="right")
        table.add_column("Kind", style="cyan")
        table.add_column("Topic", style="magenta")
        table.add_column("Content")
        for record in records:
            table.add_row(f"{record.score:.3f}", record.kind, record.topic, record.content[:90])
        console.print(table)
    else:
        for record in memory.recent(args.limit):
            console.print(f"  [{record.kind}] {record.topic}: {record.content[:110]}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="athena", description="Athena - a multi-agent AI Research Scientist"
    )
    parser.add_argument("--session", default="default", help="Session id for conversation memory")
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    study = sub.add_parser("study", help="Run the full eight-stage research pipeline")
    study.add_argument("question")
    study.add_argument("--attach", action="append", help="PDF/image/audio to ingest (repeatable)")
    study.add_argument("--yes", action="store_true", help="Auto-approve gated tools")

    resume = sub.add_parser("resume", help="Resume a run awaiting human approval")
    resume.add_argument("run_id")
    resume.add_argument("--reject", action="store_true", help="Reject instead of approve")
    resume.add_argument("--note", default="", help="Reason recorded in the audit trail")

    agents_cmd = sub.add_parser("agents", help="Show the agent roster")
    agents_cmd.add_argument("--json", action="store_true")

    memory_cmd = sub.add_parser("memory", help="Inspect long-term memory")
    memory_cmd.add_argument("--query", help="Semantic recall query")
    memory_cmd.add_argument("--limit", type=int, default=10)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if not settings.live and args.command == "study":
        console.print(
            "[yellow]No OPENAI_API_KEY_2 set - running offline. Model outputs are "
            "fixtures; literature, RAG and analysis tools are real.[/yellow]\n"
        )

    try:
        if args.command == "study":
            return asyncio.run(cmd_study(args))
        if args.command == "resume":
            return asyncio.run(cmd_resume(args))
        if args.command == "agents":
            return cmd_agents(args)
        if args.command == "memory":
            return cmd_memory(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
