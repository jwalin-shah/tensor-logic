from tensor_logic.personal_query_operators import (
    CalendarTransition,
    TravelFeasibility,
    RelationshipAttention,
    binary_compose_relation,
    build_personal_operator_registry,
    calendar_transition_adapter,
    open_commitment_projection,
    relationship_attention_score,
    travel_feasibility,
)
from tensor_logic.personal_query_catalog import (
    RELATIONSHIP_TARGET,
    TRAVEL_TARGET,
    build_personal_query_registry,
)
from tensor_logic.query_execution import (
    QueryExecutionAssessment,
    QueryExecutionStatus,
)
from tensor_logic.query_runtime import QueryReadiness
from tensor_logic.selective_executor import execute_compiled_query


def _ready(partial: bool = False):
    return QueryExecutionAssessment(
        status=(
            QueryExecutionStatus.READY_PARTIAL
            if partial
            else QueryExecutionStatus.READY_COMPLETE
        ),
        runnable=True,
        complete_coverage=not partial,
        readiness=QueryReadiness(
            ready=True,
            blocked_sources=(),
            stale_sources=(),
            missing_sources=(),
        ),
        coverage=(),
        reasons=("coverage_partial:relationship_messages",) if partial else (),
    )


def test_calendar_transition_adapter_orders_events_and_computes_gap():
    rows = calendar_transition_adapter(
        (
            {"me": ("e2", "e1")},
            {"e1": "home", "e2": "office"},
            {"e1": 100.0, "e2": 170.0},
            {"e1": 130.0, "e2": 220.0},
        ),
        {},
    )

    assert rows == (
        CalendarTransition(
            person="me",
            from_event="e1",
            to_event="e2",
            origin="home",
            destination="office",
            from_end=130.0,
            to_start=170.0,
            gap_minutes=40.0,
        ),
    )


def test_travel_feasibility_exposes_slack_and_feasible_boolean():
    transitions = (
        CalendarTransition(
            "me",
            "e1",
            "e2",
            "home",
            "office",
            130.0,
            170.0,
            40.0,
        ),
    )
    rows = travel_feasibility(
        (transitions, {("home", "office"): 25.0}),
        {"buffer_minutes": 5.0},
    )

    assert rows == (
        TravelFeasibility(
            person="me",
            from_event="e1",
            to_event="e2",
            origin="home",
            destination="office",
            gap_minutes=40.0,
            route_minutes=25.0,
            slack_minutes=10.0,
            feasible=True,
        ),
    )


def test_missing_route_fails_closed():
    transitions = (
        CalendarTransition(
            "me",
            "e1",
            "e2",
            "home",
            "office",
            0.0,
            30.0,
            30.0,
        ),
    )
    try:
        travel_feasibility((transitions, {}), {})
    except ValueError as exc:
        assert "missing route minutes" in str(exc)
    else:
        raise AssertionError("missing route evidence should fail closed")


def test_open_commitments_project_to_people():
    out = open_commitment_projection(
        (
            {"c1": "alice", "c2": "alice", "c3": "bob"},
            {"c1": False, "c2": True, "c3": False},
        ),
        {},
    )

    assert out == {"alice": 1.0, "bob": 1.0}


def test_relationship_attention_is_componentized_and_sorted():
    rows = relationship_attention_score(
        (
            {"alice": 0.9, "bob": 0.4},
            {"alice": 40.0, "bob": 2.0},
            {"alice": 1.0, "bob": 0.0},
            {
                "importance": 0.45,
                "staleness": 0.25,
                "open_commitment": 0.30,
                "staleness_scale_days": 30.0,
                "commitment_scale": 2.0,
            },
        ),
        {},
    )

    assert rows[0].person == "alice"
    assert rows[0].score > rows[1].score
    assert rows[0].staleness_component == 1.0
    assert rows[0].open_commitment_component == 0.5
    assert rows[0].weights == (
        ("importance", 0.45),
        ("staleness", 0.25),
        ("open_commitment", 0.30),
    )


def test_binary_compose_relation_is_exact():
    out = binary_compose_relation(
        (
            {("tool1", "evidence1"), ("tool2", "evidence2")},
            {("evidence1", "claim1"), ("evidence2", "claim2")},
        ),
        {},
    )
    assert out == frozenset(
        {("tool1", "claim1"), ("tool2", "claim2")}
    )


def test_end_to_end_travel_query_executes_only_needed_primitives():
    registry = build_personal_query_registry()
    query = registry.compile(TRAVEL_TARGET)
    reads = []
    values = {
        "attends": {"me": ("e1", "e2")},
        "located_at": {"e1": "home", "e2": "office"},
        "starts_at": {"e1": 100.0, "e2": 170.0},
        "ends_at": {"e1": 130.0, "e2": 220.0},
        "travel_minutes": {("home", "office"): 25.0},
        "relationship_importance": {"alice": 1.0},
    }

    result = execute_compiled_query(
        registry,
        query,
        lambda name: reads.append(name) or values[name],
        build_personal_operator_registry(),
        assessment=_ready(),
    )

    assert reads == [
        "attends",
        "ends_at",
        "located_at",
        "starts_at",
        "travel_minutes",
    ]
    assert len(result.output) == 1
    assert result.output[0].feasible is True
    assert result.output[0].slack_minutes == 15.0
    assert result.trace.executed_views == (
        "calendar_transition",
        TRAVEL_TARGET,
    )


def test_end_to_end_relationship_query_runs_under_partial_coverage():
    registry = build_personal_query_registry()
    query = registry.compile(RELATIONSHIP_TARGET)
    reads = []
    values = {
        "relationship_importance": {"alice": 0.8, "bob": 0.6},
        "last_interaction_age_days": {"alice": 35.0, "bob": 5.0},
        "commitment_target_person": {
            "c1": "alice",
            "c2": "bob",
        },
        "commitment_completed": {
            "c1": False,
            "c2": True,
        },
        "policy_weight": {
            "importance": 0.5,
            "staleness": 0.2,
            "open_commitment": 0.3,
        },
    }

    result = execute_compiled_query(
        registry,
        query,
        lambda name: reads.append(name) or values[name],
        build_personal_operator_registry(),
        assessment=_ready(partial=True),
    )

    assert result.output[0].person == "alice"
    assert isinstance(result.output[0], RelationshipAttention)
    assert result.output[0].open_commitments == 1.0
    assert reads == [
        "commitment_completed",
        "commitment_target_person",
        "last_interaction_age_days",
        "policy_weight",
        "relationship_importance",
    ]
