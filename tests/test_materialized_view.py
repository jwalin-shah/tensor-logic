from tensor_logic.materialized_view import (
    TensorQueryPlan,
    ViewCache,
    binary_composition_witnesses,
)
from tensor_logic.world_tensor import (
    CoordinateProvenance,
    TensorWorld,
)


def _fixture():
    world = TensorWorld()
    world.add_axis("Person", "Person", ("p0", "p1"))
    world.add_axis("Event", "Event", ("e0", "e1", "e2"))
    world.add_axis("Topic", "Topic", ("t0", "t1"))

    attends = world.add_tensor(
        "attends",
        ("Person", "Event"),
    )
    event_topic = world.add_tensor(
        "event_topic",
        ("Event", "Topic"),
    )

    attends.set(
        ("p0", "e0"),
        1.0,
        provenance=CoordinateProvenance(
            evidence_refs=("calendar:e0",),
            source_refs=("calendar",),
            admission_ref="admit:e0",
        ),
    )
    attends.set(
        ("p0", "e1"),
        1.0,
        provenance=CoordinateProvenance(
            evidence_refs=("calendar:e1",),
            source_refs=("calendar",),
            admission_ref="admit:e1",
        ),
    )
    event_topic.set(
        ("e0", "t0"),
        1.0,
        provenance=CoordinateProvenance(
            evidence_refs=("event:e0:topic",),
            source_refs=("event_source",),
            admission_ref="admit:topic0",
        ),
    )
    event_topic.set(
        ("e1", "t0"),
        1.0,
        provenance=CoordinateProvenance(
            evidence_refs=("event:e1:topic",),
            source_refs=("event_source",),
            admission_ref="admit:topic1",
        ),
    )

    plan = TensorQueryPlan(
        plan_id="person-topic-v1",
        operator_id="sparse_binary_compose",
        operator_version="1",
        input_tensors=("attends", "event_topic"),
        output_tensor="person_topic",
        freshness_requirements=("calendar:fresh",),
    )
    return world, attends, event_topic, plan


def test_view_cache_hits_on_identical_revisions():
    _, attends, event_topic, plan = _fixture()
    cache = ViewCache()

    first, first_hit = cache.materialize_binary(
        plan,
        attends,
        event_topic,
        snapshot_digest="snapshot:1",
    )
    second, second_hit = cache.materialize_binary(
        plan,
        attends,
        event_topic,
        snapshot_digest="snapshot:1",
    )

    assert first_hit is False
    assert second_hit is True
    assert first is second
    assert first.metadata.output_nnz == 1
    assert first.metadata.digest == second.metadata.digest


def test_binary_witnesses_return_exact_input_provenance():
    _, attends, event_topic, plan = _fixture()
    cache = ViewCache()
    view, _ = cache.materialize_binary(
        plan,
        attends,
        event_topic,
        snapshot_digest="snapshot:1",
    )

    witnesses = binary_composition_witnesses(
        view,
        attends,
        event_topic,
        ("p0", "t0"),
    )

    assert [w.intermediate_symbol for w in witnesses] == [
        "e0",
        "e1",
    ]
    assert witnesses[0].left_provenance["evidence_refs"] == [
        "calendar:e0"
    ]
    assert witnesses[0].right_provenance["evidence_refs"] == [
        "event:e0:topic"
    ]


def test_tensor_delta_forces_cache_miss_and_old_view_becomes_stale():
    _, attends, event_topic, plan = _fixture()
    cache = ViewCache()
    old_view, _ = cache.materialize_binary(
        plan,
        attends,
        event_topic,
        snapshot_digest="snapshot:1",
    )
    old_revision = attends.revision_token

    attends.set(("p1", "e1"), 1.0)

    assert attends.revision_token != old_revision

    try:
        binary_composition_witnesses(
            old_view,
            attends,
            event_topic,
            ("p0", "t0"),
        )
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("expected stale-view rejection")

    new_view, hit = cache.materialize_binary(
        plan,
        attends,
        event_topic,
        snapshot_digest="snapshot:2",
    )
    assert hit is False
    assert new_view.metadata.input_revision_tokens != (
        old_view.metadata.input_revision_tokens
    )


def test_dependency_invalidation_removes_cached_views():
    _, attends, event_topic, plan = _fixture()
    cache = ViewCache()
    cache.materialize_binary(
        plan,
        attends,
        event_topic,
        snapshot_digest="snapshot:1",
    )

    assert cache.size == 1
    assert cache.invalidate_input("attends") == 1
    assert cache.size == 0
