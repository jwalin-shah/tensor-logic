"""LifeOps-aware preflight for canonical Personal Physics queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .lifeops_query_health import lifeops_source_states
from .personal_query_catalog import (
    QUERY_COVERAGE,
    build_personal_query_registry,
)
from .query_execution import (
    QueryExecutionAssessment,
    assess_query_execution,
)
from .query_runtime import CompiledQuery, SourceState


@dataclass(frozen=True)
class PersonalQueryPreflight:
    query: CompiledQuery
    assessment: QueryExecutionAssessment
    state_revisions: tuple[tuple[str, str | None], ...]


def preflight_personal_query(
    target: str,
    source_health: Mapping[str, Any],
    *,
    extra_states: Mapping[str, SourceState] | None = None,
) -> PersonalQueryPreflight:
    registry = build_personal_query_registry()
    query = registry.compile(target)

    states = lifeops_source_states(source_health)
    if extra_states:
        states.update(extra_states)

    assessment = assess_query_execution(
        registry,
        query,
        states,
        QUERY_COVERAGE.get(target, ()),
    )

    relevant_sources = {
        req.source
        for req in query.required_sources
    }
    for coverage in QUERY_COVERAGE.get(target, ()):
        relevant_sources.update(coverage.sources)

    revisions = tuple(
        sorted(
            (
                source,
                states[source].revision
                if source in states
                else None,
            )
            for source in relevant_sources
        )
    )

    return PersonalQueryPreflight(
        query=query,
        assessment=assessment,
        state_revisions=revisions,
    )
