"""Every structured output must compile under strict JSON-schema mode.

Strict mode forbids optional fields, defaults and free-form dicts. A schema that
violates it fails at *runtime on the first API call*, which is an expensive way
to find out, so it is checked here instead.
"""

from __future__ import annotations

import pytest
from athena.runtime import AgentOutputSchema

from athena import schemas

STRUCTURED_OUTPUTS = [
    schemas.ResearchPlan,
    schemas.LiteratureReview,
    schemas.KnowledgeSynthesis,
    schemas.GapAnalysis,
    schemas.HypothesisSet,
    schemas.ExperimentDesign,
    schemas.DataInterpretation,
    schemas.CritiqueReport,
    schemas.CitationBundle,
    schemas.ResearchReport,
    schemas.RelevanceCheck,
    schemas.IntegrityCheck,
]


@pytest.mark.parametrize("model", STRUCTURED_OUTPUTS, ids=lambda m: m.__name__)
def test_schema_is_strict_mode_compatible(model):
    schema = AgentOutputSchema(model)
    assert schema.is_strict_json_schema()
    generated = schema.json_schema()
    assert generated["additionalProperties"] is False


def test_every_agent_output_is_covered():
    """Guard against adding an agent output type and forgetting to test it."""
    from athena.agents import build_team

    declared = {
        agent.output_type
        for agent in build_team().all_agents
        if agent.output_type is not None
    }
    assert declared.issubset(set(STRUCTURED_OUTPUTS))


def test_pipeline_stages_are_unique():
    values = [stage.value for stage in schemas.Stage]
    assert len(values) == len(set(values))
