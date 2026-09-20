from tensor_logic.materialized_dag import (
    MaterializedDagCache,
    execute_materialized_dag,
    primitive_artifact,
    value_digest,
)
from tensor_logic.personal_query_catalog import (
    RELATIONSHIP_TARGET,
    TRAVEL_TARGET,
    build_personal_query_registry,
)
from tensor_logic.personal_query_operators import (
    TravelFeasibility,
    build_personal_operator_registry,
)


def _artifact_loader(values, revisions, reads):
    def load(name):
        reads.append(name)
        return primitive_artifact(
            name,
            values[name],
            revision=revisions[name],
        )
    return load


def test_dataclass_operator_output_has_deterministic_digest():
    row = TravelFeasibility(
        person="me",
        from_event="e1",
        to_event="e2",
        origin="home",
        destination="office",
        gap_minutes=40.0,
        route_minutes=25.0,
        slack_minutes=15.0,
        feasible=True,
    )

    assert value_digest((row,)) == value_digest((row,))
    changed = TravelFeasibility(
        **{
            **row.__dict__,
            "slack_minutes": 14.0,
        }
    )
    assert value_digest((row,)) != value_digest((changed,))


def test_repeated_travel_query_reuses_both_derived_views():
    registry = build_personal_query_registry()
    query = registry.compile(TRAVEL_TARGET)
    cache = MaterializedDagCache()
    reads = []

    values = {
        "attends": {"me": ("e1", "e2")},
        "located_at": {"e1": "home", "e2": "office"},
        "starts_at": {"e1": 100.0, "e2": 170.0},
        "ends_at": {"e1": 130.0, "e2": 220.0},
        "travel_minutes": {("home", "office"): 25.0},
    }
    revisions = {name: "1" for name in values}

    first = execute_materialized_dag(
        registry,
        query,
        _artifact_loader(values, revisions, reads),
        build_personal_operator_registry(),
        cache,
    )
    second = execute_materialized_dag(
        registry,
        query,
        _artifact_loader(values, revisions, reads),
        build_personal_operator_registry(),
        cache,
    )

    assert first.output.value[0].slack_minutes == 15.0
    assert first.trace.cache_misses == (
        "calendar_transition",
        TRAVEL_TARGET,
    )
    assert second.trace.cache_hits == (
        "calendar_transition",
        TRAVEL_TARGET,
    )
    assert second.trace.executed_views == ()


def test_route_revision_change_reuses_calendar_transition_only():
    registry = build_personal_query_registry()
    query = registry.compile(TRAVEL_TARGET)
    cache = MaterializedDagCache()

    values = {
        "attends": {"me": ("e1", "e2")},
        "located_at": {"e1": "home", "e2": "office"},
        "starts_at": {"e1": 100.0, "e2": 170.0},
        "ends_at": {"e1": 130.0, "e2": 220.0},
        "travel_minutes": {("home", "office"): 25.0},
    }
    revisions = {name: "1" for name in values}

    execute_materialized_dag(
        registry,
        query,
        _artifact_loader(values, revisions, []),
        build_personal_operator_registry(),
        cache,
    )

    values["travel_minutes"] = {("home", "office"): 35.0}
    revisions["travel_minutes"] = "2"

    second = execute_materialized_dag(
        registry,
        query,
        _artifact_loader(values, revisions, []),
        build_personal_operator_registry(),
        cache,
    )

    assert second.output.value[0].slack_minutes == 5.0
    assert second.trace.cache_hits == ("calendar_transition",)
    assert second.trace.cache_misses == (TRAVEL_TARGET,)
    assert second.trace.executed_views == (TRAVEL_TARGET,)


def test_relationship_policy_change_reuses_open_commitment_projection():
    registry = build_personal_query_registry()
    query = registry.compile(RELATIONSHIP_TARGET)
    cache = MaterializedDagCache()

    values = {
        "relationship_importance": {"alice": 0.8},
        "last_interaction_age_days": {"alice": 30.0},
        "commitment_target_person": {"c1": "alice"},
        "commitment_completed": {"c1": False},
        "policy_weight": {
            "importance": 0.5,
            "staleness": 0.2,
            "open_commitment": 0.3,
        },
    }
    revisions = {name: "1" for name in values}

    first = execute_materialized_dag(
        registry,
        query,
        _artifact_loader(values, revisions, []),
        build_personal_operator_registry(),
        cache,
    )

    values["policy_weight"] = {
        "importance": 0.2,
        "staleness": 0.2,
        "open_commitment": 0.6,
    }
    revisions["policy_weight"] = "2"

    second = execute_materialized_dag(
        registry,
        query,
        _artifact_loader(values, revisions, []),
        build_personal_operator_registry(),
        cache,
    )

    assert first.output.value[0].score != second.output.value[0].score
    assert second.trace.cache_hits == ("open_commitment_person",)
    assert second.trace.cache_misses == (RELATIONSHIP_TARGET,)
