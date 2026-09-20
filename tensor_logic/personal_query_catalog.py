"""Canonical query catalog for high-value Personal Physics questions.

The catalog is intentionally smaller than the full tensor schema. It defines
trusted dependency/source contracts for queries we want to execute repeatedly.
"""

from __future__ import annotations

from .query_coverage import CoverageRequirement
from .query_runtime import (
    DerivedViewSpec,
    PrimitiveTensorSpec,
    QueryRegistry,
    SourceRequirement,
)


TRAVEL_TARGET = "travel_feasibility_candidate"
RELATIONSHIP_TARGET = "relationship_attention_candidate"


QUERY_COVERAGE: dict[str, tuple[CoverageRequirement, ...]] = {
    RELATIONSHIP_TARGET: (
        CoverageRequirement(
            name="relationship_messages",
            sources=(
                "capture-source:gmail",
                "capture-source:imessage",
                "capture-source:whatsapp",
                "capture-source:linkedin",
            ),
            minimum_readable=1,
            max_age_seconds=900,
        ),
        CoverageRequirement(
            name="commitment_sources",
            sources=(
                "capture-source:google_tasks",
                "capture-source:apple_reminders",
            ),
            minimum_readable=1,
            max_age_seconds=1800,
        ),
    ),
}


def build_personal_query_registry() -> QueryRegistry:
    registry = QueryRegistry()

    # Calendar / travel.
    calendar = (
        SourceRequirement(
            "capture-source:google_calendar",
            max_age_seconds=900,
        ),
    )
    for name in ("attends", "located_at", "starts_at", "ends_at"):
        registry.add_primitive(
            PrimitiveTensorSpec(name, calendar)
        )

    # Route times are not currently represented by LifeOps source_health, so
    # the runtime that invokes the route adapter must inject this SourceState.
    registry.add_primitive(
        PrimitiveTensorSpec(
            "travel_minutes",
            (
                SourceRequirement(
                    "route_engine",
                    max_age_seconds=300,
                ),
            ),
        )
    )

    # Relationship / commitments.
    registry.add_primitive(
        PrimitiveTensorSpec(
            "relationship_importance",
            (SourceRequirement("human_policy"),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec("last_interaction_age_days")
    )
    registry.add_primitive(
        PrimitiveTensorSpec("commitment_target_person")
    )
    registry.add_primitive(
        PrimitiveTensorSpec("commitment_completed")
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "policy_weight",
            (SourceRequirement("human_policy"),),
        )
    )

    # Tool/evidence path useful for consequential claims.
    registry.add_primitive(
        PrimitiveTensorSpec("produced_evidence")
    )
    registry.add_primitive(
        PrimitiveTensorSpec("supports_claim")
    )

    registry.add_view(
        DerivedViewSpec(
            output="calendar_transition",
            inputs=(
                "attends",
                "located_at",
                "starts_at",
                "ends_at",
            ),
            operator_id="calendar_transition_adapter",
            operator_version="1",
        )
    )
    registry.add_view(
        DerivedViewSpec(
            output=TRAVEL_TARGET,
            inputs=("calendar_transition", "travel_minutes"),
            operator_id="travel_feasibility",
            operator_version="1",
        )
    )

    registry.add_view(
        DerivedViewSpec(
            output="open_commitment_person",
            inputs=(
                "commitment_target_person",
                "commitment_completed",
            ),
            operator_id="open_commitment_projection",
            operator_version="1",
        )
    )
    registry.add_view(
        DerivedViewSpec(
            output=RELATIONSHIP_TARGET,
            inputs=(
                "relationship_importance",
                "last_interaction_age_days",
                "open_commitment_person",
                "policy_weight",
            ),
            operator_id="relationship_attention_score",
            operator_version="1",
        )
    )

    registry.add_view(
        DerivedViewSpec(
            output="tool_backed_claim",
            inputs=("produced_evidence", "supports_claim"),
            operator_id="binary_compose",
            operator_version="1",
        )
    )

    return registry
