from tensor_logic.reasoning_persistence import (
    MaterializedDerivedView,
    StatelessDerivedView,
    VersionedSource,
)


def test_materialized_view_avoids_repeated_source_reads_and_derivations():
    source = VersionedSource({"a": 3})
    stateless = StatelessDerivedView(source, lambda key, value: value * 2)
    materialized = MaterializedDerivedView(
        source,
        lambda key, value: value * 2,
    )

    for _ in range(5):
        assert stateless.get("a") == 6
        assert materialized.get("a") == 6

    assert stateless.metrics.source_reads == 5
    assert stateless.metrics.derivations == 5
    assert materialized.metrics.source_reads == 1
    assert materialized.metrics.derivations == 1
    assert materialized.metrics.cache_hits == 4


def test_source_revision_invalidates_cached_result():
    source = VersionedSource({"a": 3})
    materialized = MaterializedDerivedView(
        source,
        lambda key, value: value * 2,
    )

    assert materialized.get("a") == 6
    source.set("a", 5)
    assert materialized.get("a") == 10

    assert materialized.metrics.source_reads == 2
    assert materialized.metrics.derivations == 2
    assert materialized.metrics.invalidations == 1


def test_explicit_invalidation_can_target_one_key():
    source = VersionedSource({"a": 1, "b": 2})
    materialized = MaterializedDerivedView(
        source,
        lambda key, value: value + 1,
    )

    materialized.get("a")
    materialized.get("b")
    assert materialized.invalidate("a") == 1

    materialized.get("b")
    materialized.get("a")

    assert materialized.metrics.cache_hits == 1
    assert materialized.metrics.derivations == 3
