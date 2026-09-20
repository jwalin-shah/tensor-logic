from tensor_logic.query_execution import (
    QueryExecutionAssessment,
    QueryExecutionStatus,
)
from tensor_logic.query_runtime import (
    DerivedViewSpec,
    PrimitiveTensorSpec,
    QueryReadiness,
    QueryRegistry,
)
from tensor_logic.selective_executor import (
    QueryOperatorRegistry,
    execute_compiled_query,
)


def _registry():
    r = QueryRegistry()
    for name in ("a", "b", "c", "unrelated"):
        r.add_primitive(PrimitiveTensorSpec(name))
    r.add_view(
        DerivedViewSpec(
            output="ab",
            inputs=("a", "b"),
            operator_id="sum",
            operator_version="1",
        )
    )
    r.add_view(
        DerivedViewSpec(
            output="abc",
            inputs=("ab", "c"),
            operator_id="mul",
            operator_version="1",
        )
    )
    r.add_view(
        DerivedViewSpec(
            output="other",
            inputs=("unrelated",),
            operator_id="identity",
            operator_version="1",
        )
    )
    return r


def _operators():
    ops = QueryOperatorRegistry()
    ops.register("sum", "1", lambda inputs, params: inputs[0] + inputs[1])
    ops.register("mul", "1", lambda inputs, params: inputs[0] * inputs[1])
    ops.register("identity", "1", lambda inputs, params: inputs[0])
    return ops


def test_executor_reads_only_compiled_primitives_and_views():
    registry = _registry()
    query = registry.compile("abc")
    reads = []

    values = {
        "a": 2,
        "b": 3,
        "c": 4,
        "unrelated": 999,
    }

    result = execute_compiled_query(
        registry,
        query,
        lambda name: reads.append(name) or values[name],
        _operators(),
    )

    assert result.output == 20
    assert reads == ["a", "b", "c"]
    assert "unrelated" not in reads
    assert result.trace.executed_views == ("ab", "abc")
    assert result.trace.operator_keys == ("sum@1", "mul@1")
    assert result.trace.digest


def test_primitive_only_query_executes_no_view():
    registry = _registry()
    query = registry.compile("a")
    reads = []

    result = execute_compiled_query(
        registry,
        query,
        lambda name: reads.append(name) or 7,
        _operators(),
    )

    assert result.output == 7
    assert reads == ["a"]
    assert result.trace.executed_views == ()
    assert result.trace.operator_keys == ()


def test_blocked_preflight_prevents_any_primitive_read():
    registry = _registry()
    query = registry.compile("abc")
    reads = []
    assessment = QueryExecutionAssessment(
        status=QueryExecutionStatus.BLOCKED,
        runnable=False,
        complete_coverage=False,
        readiness=QueryReadiness(
            ready=False,
            blocked_sources=("calendar",),
            stale_sources=(),
            missing_sources=(),
        ),
        coverage=(),
        reasons=("unreadable:calendar",),
    )

    try:
        execute_compiled_query(
            registry,
            query,
            lambda name: reads.append(name) or 1,
            _operators(),
            assessment=assessment,
        )
    except ValueError as exc:
        assert "blocked by preflight" in str(exc)
    else:
        raise AssertionError("blocked query should not execute")

    assert reads == []


def test_missing_operator_fails_before_producing_derived_output():
    registry = _registry()
    query = registry.compile("abc")
    ops = QueryOperatorRegistry()
    ops.register("sum", "1", lambda inputs, params: sum(inputs))

    try:
        execute_compiled_query(
            registry,
            query,
            lambda name: {"a": 1, "b": 2, "c": 3}[name],
            ops,
        )
    except ValueError as exc:
        assert "mul@1" in str(exc)
    else:
        raise AssertionError("unknown operator should fail closed")


def test_trace_digest_is_deterministic():
    registry = _registry()
    query = registry.compile("abc")

    left = execute_compiled_query(
        registry,
        query,
        lambda name: {"a": 2, "b": 3, "c": 4}[name],
        _operators(),
    )
    right = execute_compiled_query(
        registry,
        query,
        lambda name: {"a": 2, "b": 3, "c": 4}[name],
        _operators(),
    )

    assert left.trace.digest == right.trace.digest
