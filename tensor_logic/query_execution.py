"""One execution envelope combining hard source gates and evidence coverage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .query_coverage import (
    CoverageAssessment,
    CoverageRequirement,
    CoverageStatus,
    assess_coverage,
)
from .query_runtime import (
    CompiledQuery,
    QueryReadiness,
    QueryRegistry,
    SourceState,
)


class QueryExecutionStatus(str, Enum):
    READY_COMPLETE = "ready_complete"
    READY_PARTIAL = "ready_partial"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class QueryExecutionAssessment:
    status: QueryExecutionStatus
    runnable: bool
    complete_coverage: bool
    readiness: QueryReadiness
    coverage: tuple[CoverageAssessment, ...]
    reasons: tuple[str, ...]


def assess_query_execution(
    registry: QueryRegistry,
    query: CompiledQuery,
    states: Mapping[str, SourceState],
    coverage_requirements: Sequence[CoverageRequirement] = (),
) -> QueryExecutionAssessment:
    readiness = registry.readiness(query, states)
    coverage = tuple(
        assess_coverage(requirement, states)
        for requirement in coverage_requirements
    )

    reasons: list[str] = []
    if readiness.blocked_sources:
        reasons.append(
            "unreadable:" + ",".join(readiness.blocked_sources)
        )
    if readiness.stale_sources:
        reasons.append(
            "stale:" + ",".join(readiness.stale_sources)
        )
    if readiness.missing_sources:
        reasons.append(
            "missing:" + ",".join(readiness.missing_sources)
        )

    blocked_coverage = tuple(
        item.name
        for item in coverage
        if item.status == CoverageStatus.BLOCKED
    )
    partial_coverage = tuple(
        item.name
        for item in coverage
        if item.status == CoverageStatus.PARTIAL
    )
    if blocked_coverage:
        reasons.append(
            "coverage_blocked:" + ",".join(blocked_coverage)
        )
    if partial_coverage:
        reasons.append(
            "coverage_partial:" + ",".join(partial_coverage)
        )

    runnable = readiness.ready and not blocked_coverage
    complete = runnable and all(
        item.status == CoverageStatus.COMPLETE
        for item in coverage
    )

    if not runnable:
        status = QueryExecutionStatus.BLOCKED
    elif complete:
        status = QueryExecutionStatus.READY_COMPLETE
    else:
        status = QueryExecutionStatus.READY_PARTIAL

    return QueryExecutionAssessment(
        status=status,
        runnable=runnable,
        complete_coverage=complete,
        readiness=readiness,
        coverage=coverage,
        reasons=tuple(reasons),
    )
