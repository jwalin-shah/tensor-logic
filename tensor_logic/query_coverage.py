"""Coverage semantics for queries over partially available LifeOps sources.

Hard readiness answers "can this computation run safely?"
Coverage answers "how complete is the evidence universe we intended to inspect?"

This prevents two bad behaviors:
- blocking useful computation because one optional channel is unavailable;
- silently treating unavailable channels as evidence that nothing exists there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .query_runtime import SourceState


class CoverageStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class CoverageRequirement:
    name: str
    sources: tuple[str, ...]
    minimum_readable: int = 1
    max_age_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coverage requirement name is required")
        if not self.sources:
            raise ValueError("coverage requirement needs at least one source")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("coverage sources must be unique")
        if not 1 <= self.minimum_readable <= len(self.sources):
            raise ValueError("minimum_readable outside source count")
        if self.max_age_seconds is not None and self.max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative")


@dataclass(frozen=True)
class CoverageAssessment:
    name: str
    status: CoverageStatus
    usable: bool
    complete: bool
    readable_sources: tuple[str, ...]
    stale_sources: tuple[str, ...]
    unavailable_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]
    coverage_fraction: float


def assess_coverage(
    requirement: CoverageRequirement,
    states: Mapping[str, SourceState],
) -> CoverageAssessment:
    readable: list[str] = []
    stale: list[str] = []
    unavailable: list[str] = []
    missing: list[str] = []

    for source in requirement.sources:
        state = states.get(source)
        if state is None:
            missing.append(source)
            continue
        if not state.readable:
            unavailable.append(source)
            continue
        if (
            requirement.max_age_seconds is not None
            and (
                state.age_seconds is None
                or state.age_seconds > requirement.max_age_seconds
            )
        ):
            stale.append(source)
            continue
        readable.append(source)

    usable = len(readable) >= requirement.minimum_readable
    complete = len(readable) == len(requirement.sources)

    if complete:
        status = CoverageStatus.COMPLETE
    elif usable:
        status = CoverageStatus.PARTIAL
    else:
        status = CoverageStatus.BLOCKED

    return CoverageAssessment(
        name=requirement.name,
        status=status,
        usable=usable,
        complete=complete,
        readable_sources=tuple(sorted(readable)),
        stale_sources=tuple(sorted(stale)),
        unavailable_sources=tuple(sorted(unavailable)),
        missing_sources=tuple(sorted(missing)),
        coverage_fraction=len(readable) / len(requirement.sources),
    )
