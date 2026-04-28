from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class EvalResult:
    artifact: str
    score: float
    secondary_score: float = 0.0
    asi: str = ""
    asi_kind: Literal["proof", "why_not", "engine_error"] = "proof"


def _dominates(a: EvalResult, b: EvalResult) -> bool:
    """True if a Pareto-dominates b (higher is better on both axes)."""
    return (
        a.score >= b.score
        and a.secondary_score >= b.secondary_score
        and (a.score > b.score or a.secondary_score > b.secondary_score)
    )


def _update_frontier(
    frontier: list[EvalResult], candidate: EvalResult, frontier_size: int
) -> list[EvalResult]:
    """Add candidate to Pareto frontier if non-dominated; prune dominated entries."""
    if any(_dominates(existing, candidate) for existing in frontier):
        return frontier
    pruned = [e for e in frontier if not _dominates(candidate, e)]
    pruned.append(candidate)
    # Sort: best primary first; tiebreak secondary desc, then shorter artifact (Occam)
    pruned.sort(key=lambda e: (-e.score, -e.secondary_score, len(e.artifact)))
    return pruned[:frontier_size]
