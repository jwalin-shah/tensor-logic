from tensor_logic.materialized_dag import (
    MaterializedDagCache,
    execute_materialized_dag,
    primitive_artifact,
)
from tensor_logic.query_runtime import (
    DerivedViewSpec,
    PrimitiveTensorSpec,
    QueryRegistry,
)
from tensor_logic.selective_executor import QueryOperatorRegistry


def _registry():
    r = QueryRegistry()
    for name in ("a", "b", "c", "unrelated"):
        r.add_primitive(PrimitiveTensorSpec(name))
    r.add_view(
        DerivedViewSpec(
            "ab",
            ("a", "b"),
            "sum",
            "1",
        )
    )
    r.add_view(
        DerivedViewSpec(
            "abc",
            ("ab", "c"),
            "mul",
            "1",
        )
    )
    return r


def _operators(calls):
    ops = QueryOperatorRegistry()

    def add(inputs, params):
        calls.append("sum")
        return inputs[0] + inputs[1]

    def mul(inputs, params):
        calls.append("mul")
        return inputs[0] * inputs[1]

    ops.register("sum", "1", add)
    ops.register("mul", "1", mul)
    return ops


def _loader(values, revisions, reads):
    def load(name):
        reads.append(name)
        return primitive_artifact(
            name,
            values[name],
            revision=revisions[name],
        )
    return load


def test_second_run_reuses_all_derived_views():
    registry = _registry()
    query = registry.compile("abc")
    cache = MaterializedDagCache()
    calls = []
    reads = []
    values = {"a": 2, "b": 3, "c": 4}
    revisions = {"a": "1", "b": "1", "c": "1"}

    first = execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, reads),
        _operators(calls),
        cache,
    )
    second = execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, reads),
        _operators(calls),
        cache,
    )

    assert first.output.value == 20
    assert second.output.value == 20
    assert first.trace.cache_misses == ("ab", "abc")
    assert first.trace.cache_hits == ()
    assert second.trace.cache_hits == ("ab", "abc")
    assert second.trace.executed_views == ()
    assert calls == ["sum", "mul"]


def test_leaf_change_reuses_clean_intermediate_view():
    registry = _registry()
    query = registry.compile("abc")
    cache = MaterializedDagCache()
    calls = []
    values = {"a": 2, "b": 3, "c": 4}
    revisions = {"a": "1", "b": "1", "c": "1"}

    execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, []),
        _operators(calls),
        cache,
    )

    values["c"] = 10
    revisions["c"] = "2"
    second = execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, []),
        _operators(calls),
        cache,
    )

    assert second.output.value == 50
    assert second.trace.cache_hits == ("ab",)
    assert second.trace.cache_misses == ("abc",)
    assert second.trace.executed_views == ("abc",)
    assert calls == ["sum", "mul", "mul"]


def test_upstream_change_recomputes_downstream_chain():
    registry = _registry()
    query = registry.compile("abc")
    cache = MaterializedDagCache()
    calls = []
    values = {"a": 2, "b": 3, "c": 4}
    revisions = {"a": "1", "b": "1", "c": "1"}

    execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, []),
        _operators(calls),
        cache,
    )

    values["a"] = 5
    revisions["a"] = "2"
    second = execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, []),
        _operators(calls),
        cache,
    )

    assert second.output.value == 32
    assert second.trace.cache_hits == ()
    assert second.trace.cache_misses == ("ab", "abc")
    assert second.trace.executed_views == ("ab", "abc")


def test_unrelated_state_is_never_loaded():
    registry = _registry()
    query = registry.compile("abc")
    cache = MaterializedDagCache()
    reads = []
    values = {"a": 2, "b": 3, "c": 4, "unrelated": 999}
    revisions = {
        "a": "1",
        "b": "1",
        "c": "1",
        "unrelated": "100",
    }

    execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, reads),
        _operators([]),
        cache,
    )

    assert reads == ["a", "b", "c"]
    assert "unrelated" not in reads


def test_same_value_new_revision_invalidates_primitive_dependency():
    registry = _registry()
    query = registry.compile("abc")
    cache = MaterializedDagCache()
    calls = []
    values = {"a": 2, "b": 3, "c": 4}
    revisions = {"a": "1", "b": "1", "c": "1"}

    execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, []),
        _operators(calls),
        cache,
    )

    revisions["a"] = "2"
    second = execute_materialized_dag(
        registry,
        query,
        _loader(values, revisions, []),
        _operators(calls),
        cache,
    )

    # Provenance/source revision changed even though the numeric value did not.
    assert second.trace.cache_misses == ("ab", "abc")


def test_primitive_artifact_requires_revision():
    try:
        primitive_artifact("x", 1, revision="")
    except ValueError as exc:
        assert "revision" in str(exc)
    else:
        raise AssertionError("revisionless primitive should fail closed")
