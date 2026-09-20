from tensor_logic.query_coverage import CoverageRequirement
from tensor_logic.query_execution import (
    QueryExecutionStatus,
    assess_query_execution,
)
from tensor_logic.query_runtime import (
    PrimitiveTensorSpec,
    QueryRegistry,
    SourceRequirement,
    SourceState,
)


def _query():
    registry = QueryRegistry()
    registry.add_primitive(
        PrimitiveTensorSpec(
            "x",
            (SourceRequirement("core", max_age_seconds=30),),
        )
    )
    return registry, registry.compile("x")


def test_ready_complete_when_hard_gate_and_all_coverage_are_fresh():
    registry, query = _query()
    states = {
        "core": SourceState("core", True, age_seconds=1),
        "a": SourceState("a", True, age_seconds=1),
        "b": SourceState("b", True, age_seconds=1),
    }

    result = assess_query_execution(
        registry,
        query,
        states,
        (
            CoverageRequirement(
                "optional_sources",
                ("a", "b"),
                minimum_readable=1,
                max_age_seconds=30,
            ),
        ),
    )

    assert result.status == QueryExecutionStatus.READY_COMPLETE
    assert result.runnable is True
    assert result.complete_coverage is True
    assert result.reasons == ()


def test_ready_partial_when_optional_coverage_is_incomplete():
    registry, query = _query()
    states = {
        "core": SourceState("core", True, age_seconds=1),
        "a": SourceState("a", True, age_seconds=1),
        "b": SourceState("b", False),
    }

    result = assess_query_execution(
        registry,
        query,
        states,
        (
            CoverageRequirement(
                "optional_sources",
                ("a", "b"),
                minimum_readable=1,
                max_age_seconds=30,
            ),
        ),
    )

    assert result.status == QueryExecutionStatus.READY_PARTIAL
    assert result.runnable is True
    assert result.complete_coverage is False
    assert result.reasons == ("coverage_partial:optional_sources",)


def test_hard_staleness_blocks_even_if_optional_coverage_is_complete():
    registry, query = _query()
    states = {
        "core": SourceState("core", True, age_seconds=100),
        "a": SourceState("a", True, age_seconds=1),
        "b": SourceState("b", True, age_seconds=1),
    }

    result = assess_query_execution(
        registry,
        query,
        states,
        (
            CoverageRequirement(
                "optional_sources",
                ("a", "b"),
                minimum_readable=1,
                max_age_seconds=30,
            ),
        ),
    )

    assert result.status == QueryExecutionStatus.BLOCKED
    assert result.runnable is False
    assert "stale:core" in result.reasons


def test_coverage_minimum_can_block_execution():
    registry, query = _query()
    states = {
        "core": SourceState("core", True, age_seconds=1),
        "a": SourceState("a", False),
        "b": SourceState("b", False),
    }

    result = assess_query_execution(
        registry,
        query,
        states,
        (
            CoverageRequirement(
                "minimum_context",
                ("a", "b"),
                minimum_readable=1,
            ),
        ),
    )

    assert result.status == QueryExecutionStatus.BLOCKED
    assert result.runnable is False
    assert result.reasons == ("coverage_blocked:minimum_context",)
