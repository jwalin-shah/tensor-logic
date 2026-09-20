from tensor_logic.query_runtime import (
    DerivedViewSpec,
    PrimitiveTensorSpec,
    QueryRegistry,
    SourceRequirement,
    SourceState,
)


def _registry():
    registry = QueryRegistry()

    registry.add_primitive(
        PrimitiveTensorSpec(
            "attends",
            (SourceRequirement("calendar", max_age_seconds=300),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "located_at",
            (SourceRequirement("calendar", max_age_seconds=300),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "travel_minutes",
            (SourceRequirement("routes", max_age_seconds=120),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "requested_followup",
            (SourceRequirement("messages", max_age_seconds=600),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "commitment_completed",
            (SourceRequirement("tasks", max_age_seconds=600),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "goal_priority",
            (SourceRequirement("human_policy"),),
        )
    )

    registry.add_view(
        DerivedViewSpec(
            output="event_place",
            inputs=("attends", "located_at"),
            operator_id="binary_compose",
            operator_version="1",
        )
    )
    registry.add_view(
        DerivedViewSpec(
            output="travel_pressure",
            inputs=("event_place", "travel_minutes"),
            operator_id="travel_join",
            operator_version="1",
        )
    )
    registry.add_view(
        DerivedViewSpec(
            output="unresolved_followup",
            inputs=("requested_followup", "commitment_completed"),
            operator_id="followup_rule",
            operator_version="1",
        )
    )
    registry.add_view(
        DerivedViewSpec(
            output="attention_candidate",
            inputs=("unresolved_followup", "goal_priority"),
            operator_id="weighted_score",
            operator_version="2",
        )
    )
    return registry


def test_compile_selects_only_transitive_dependencies_for_target():
    registry = _registry()
    query = registry.compile("travel_pressure")

    assert query.primitive_tensors == (
        "attends",
        "located_at",
        "travel_minutes",
    )
    assert query.view_order == (
        "event_place",
        "travel_pressure",
    )
    assert tuple(req.source for req in query.required_sources) == (
        "calendar",
        "routes",
    )

    # Unrelated relationship state is not loaded for a travel query.
    assert "requested_followup" not in query.primitive_tensors
    assert "messages" not in tuple(
        req.source for req in query.required_sources
    )


def test_relationship_query_has_different_minimal_plan():
    registry = _registry()
    query = registry.compile("attention_candidate")

    assert query.primitive_tensors == (
        "commitment_completed",
        "goal_priority",
        "requested_followup",
    )
    assert query.view_order == (
        "unresolved_followup",
        "attention_candidate",
    )
    assert tuple(req.source for req in query.required_sources) == (
        "human_policy",
        "messages",
        "tasks",
    )


def test_source_gate_distinguishes_missing_unreadable_and_stale():
    registry = _registry()
    query = registry.compile("travel_pressure")

    readiness = registry.readiness(
        query,
        {
            "calendar": SourceState(
                "calendar",
                readable=True,
                age_seconds=500,
            ),
            "routes": SourceState(
                "routes",
                readable=False,
                age_seconds=10,
            ),
        },
    )

    assert readiness.ready is False
    assert readiness.stale_sources == ("calendar",)
    assert readiness.blocked_sources == ("routes",)
    assert readiness.missing_sources == ()


def test_missing_source_blocks_query_instead_of_implying_false():
    registry = _registry()
    query = registry.compile("attention_candidate")

    readiness = registry.readiness(
        query,
        {
            "messages": SourceState(
                "messages",
                readable=True,
                age_seconds=10,
            ),
            "tasks": SourceState(
                "tasks",
                readable=True,
                age_seconds=10,
            ),
        },
    )

    assert readiness.ready is False
    assert readiness.missing_sources == ("human_policy",)


def test_delta_propagates_only_to_transitive_dependents():
    registry = _registry()

    calendar_delta = registry.invalidate({"located_at"})

    assert calendar_delta.dirty_views == (
        "event_place",
        "travel_pressure",
    )
    assert calendar_delta.recompute_order == (
        "event_place",
        "travel_pressure",
    )

    relationship_delta = registry.invalidate({"requested_followup"})

    assert relationship_delta.dirty_views == (
        "attention_candidate",
        "unresolved_followup",
    )
    assert relationship_delta.recompute_order == (
        "unresolved_followup",
        "attention_candidate",
    )


def test_unrelated_delta_leaves_other_query_views_clean():
    registry = _registry()
    delta = registry.invalidate({"goal_priority"})

    assert delta.dirty_views == ("attention_candidate",)
    assert "travel_pressure" not in delta.dirty_views
    assert "event_place" not in delta.dirty_views


def test_query_digest_is_deterministic():
    left = _registry().compile("travel_pressure")
    right = _registry().compile("travel_pressure")

    assert left.digest == right.digest
    assert left.definition_digests == right.definition_digests


def test_stricter_duplicate_source_requirement_wins():
    registry = QueryRegistry()
    registry.add_primitive(
        PrimitiveTensorSpec(
            "a",
            (SourceRequirement("s", max_age_seconds=100),),
        )
    )
    registry.add_primitive(
        PrimitiveTensorSpec(
            "b",
            (SourceRequirement("s", max_age_seconds=20),),
        )
    )
    registry.add_view(
        DerivedViewSpec(
            output="c",
            inputs=("a", "b"),
            operator_id="join",
            operator_version="1",
        )
    )

    query = registry.compile("c")

    assert query.required_sources == (
        SourceRequirement("s", max_age_seconds=20),
    )


def test_cycle_is_rejected():
    registry = QueryRegistry()
    registry.add_view(
        DerivedViewSpec(
            output="a",
            inputs=("b",),
            operator_id="x",
            operator_version="1",
        )
    )

    try:
        registry.add_view(
            DerivedViewSpec(
                output="b",
                inputs=("a",),
                operator_id="x",
                operator_version="1",
            )
        )
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cyclic derived views should fail closed")
