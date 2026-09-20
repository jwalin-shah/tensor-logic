from tensor_logic.personal_query_catalog import (
    QUERY_COVERAGE,
    RELATIONSHIP_TARGET,
    TRAVEL_TARGET,
    build_personal_query_registry,
)
from tensor_logic.query_coverage import CoverageStatus, assess_coverage
from tensor_logic.query_runtime import SourceState


def test_travel_query_has_only_calendar_and_route_hard_sources():
    registry = build_personal_query_registry()
    query = registry.compile(TRAVEL_TARGET)

    assert query.primitive_tensors == (
        "attends",
        "ends_at",
        "located_at",
        "starts_at",
        "travel_minutes",
    )
    assert query.view_order == (
        "calendar_transition",
        TRAVEL_TARGET,
    )
    assert tuple(req.source for req in query.required_sources) == (
        "capture-source:google_calendar",
        "route_engine",
    )


def test_relationship_query_uses_human_policy_as_hard_gate_and_coverage_for_sources():
    registry = build_personal_query_registry()
    query = registry.compile(RELATIONSHIP_TARGET)

    assert query.primitive_tensors == (
        "commitment_completed",
        "commitment_target_person",
        "last_interaction_age_days",
        "policy_weight",
        "relationship_importance",
    )
    assert tuple(req.source for req in query.required_sources) == (
        "human_policy",
    )

    coverage_names = tuple(
        requirement.name
        for requirement in QUERY_COVERAGE[RELATIONSHIP_TARGET]
    )
    assert coverage_names == (
        "relationship_messages",
        "commitment_sources",
    )


def test_relationship_query_can_be_usable_with_partial_message_coverage():
    requirements = QUERY_COVERAGE[RELATIONSHIP_TARGET]
    states = {
        "capture-source:gmail": SourceState(
            "capture-source:gmail",
            readable=True,
            age_seconds=30,
        ),
        "capture-source:imessage": SourceState(
            "capture-source:imessage",
            readable=True,
            age_seconds=20,
        ),
        "capture-source:whatsapp": SourceState(
            "capture-source:whatsapp",
            readable=False,
        ),
        "capture-source:linkedin": SourceState(
            "capture-source:linkedin",
            readable=False,
        ),
        "capture-source:google_tasks": SourceState(
            "capture-source:google_tasks",
            readable=True,
            age_seconds=40,
        ),
        "capture-source:apple_reminders": SourceState(
            "capture-source:apple_reminders",
            readable=True,
            age_seconds=50,
        ),
    }

    message_coverage = assess_coverage(requirements[0], states)
    commitments = assess_coverage(requirements[1], states)

    assert message_coverage.status == CoverageStatus.PARTIAL
    assert message_coverage.usable is True
    assert commitments.status == CoverageStatus.COMPLETE


def test_calendar_delta_does_not_dirty_relationship_attention():
    registry = build_personal_query_registry()
    delta = registry.invalidate({"starts_at"})

    assert delta.dirty_views == (
        "calendar_transition",
        TRAVEL_TARGET,
    )
    assert RELATIONSHIP_TARGET not in delta.dirty_views


def test_commitment_delta_propagates_to_relationship_attention_only():
    registry = build_personal_query_registry()
    delta = registry.invalidate({"commitment_completed"})

    assert delta.recompute_order == (
        "open_commitment_person",
        RELATIONSHIP_TARGET,
    )
    assert TRAVEL_TARGET not in delta.dirty_views
