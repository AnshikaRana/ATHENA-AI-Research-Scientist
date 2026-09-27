"""Agent team assembly.

Eight specialised agents. Four map directly onto roles named in the project
brief; the other four exist because the pipeline needs planning, grounded
synthesis and divergent idea generation to work at all.

    Brief's role            Agent in this system
    ---------------------   --------------------------------
    Literature Review       Literature Review Agent
    Research Gap Analysis    Research Gap Analyst
    Experiment Planning     Experiment Planner
    Citation Manager        Citation Manager
    Scientific Writer       Scientific Writer

    (additional)            Principal Investigator - planning and scoping
                            Knowledge Synthesist   - RAG-grounded synthesis
                            Hypothesis Generator   - divergent proposals

``build_team()`` constructs them once.
"""

from __future__ import annotations

from dataclasses import dataclass

from athena.runtime import Agent

from ..context import ResearchContext
from .citation_manager import build_citation_manager
from .experiment_planner import build_experiment_planner
from .gap_analyst import build_gap_analyst
from .hypothesis_generator import build_hypothesis_generator
from .literature_review import build_literature_review_agent
from .principal_investigator import build_principal_investigator
from .scientific_writer import build_scientific_writer
from .synthesist import build_synthesist


@dataclass(frozen=True)
class ResearchTeam:
    """Every agent in the system, constructed and wired."""

    principal_investigator: Agent[ResearchContext]
    literature_review: Agent[ResearchContext]
    synthesist: Agent[ResearchContext]
    gap_analyst: Agent[ResearchContext]
    hypothesis_generator: Agent[ResearchContext]
    experiment_planner: Agent[ResearchContext]
    citation_manager: Agent[ResearchContext]
    scientific_writer: Agent[ResearchContext]

    @property
    def specialists(self) -> list[Agent[ResearchContext]]:
        """The stage agents between planning and writing."""
        return [
            self.literature_review,
            self.synthesist,
            self.gap_analyst,
            self.hypothesis_generator,
            self.experiment_planner,
            self.citation_manager,
        ]

    @property
    def all_agents(self) -> list[Agent[ResearchContext]]:
        return [
            self.principal_investigator,
            *self.specialists,
            self.scientific_writer,
        ]

    def describe(self) -> list[dict[str, object]]:
        """Machine-readable roster, used by the docs generator and the UI."""
        return [
            {
                "name": agent.name,
                "role": agent.handoff_description or "",
                "tools": [tool.name for tool in agent.tools],
                "output_type": getattr(agent.output_type, "__name__", "text"),
                "handoffs": [
                    getattr(h, "agent_name", getattr(h, "name", "?"))
                    for h in agent.handoffs
                ],
                "input_guardrails": [g.get_name() for g in agent.input_guardrails],
                "output_guardrails": [g.get_name() for g in agent.output_guardrails],
                "model": str(agent.model),
                "temperature": getattr(agent.model_settings, "temperature", None),
            }
            for agent in self.all_agents
        ]


def build_team() -> ResearchTeam:
    """Construct the full team. Call once per process."""
    return ResearchTeam(
        principal_investigator=build_principal_investigator(),
        literature_review=build_literature_review_agent(),
        synthesist=build_synthesist(),
        gap_analyst=build_gap_analyst(),
        hypothesis_generator=build_hypothesis_generator(),
        experiment_planner=build_experiment_planner(),
        citation_manager=build_citation_manager(),
        scientific_writer=build_scientific_writer(),
    )


__all__ = ["ResearchTeam", "build_team"]
