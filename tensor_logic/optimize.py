from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class EvalResult:
    artifact: str
    score: float
    secondary_score: float = 0.0
    asi: str = ""
    asi_kind: Literal["proof", "why_not", "wrong_rule", "engine_error"] = "proof"


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


def optimize(
    propose: Callable[[str], str],
    evaluate: Callable[[str], EvalResult],
    accept: Callable[[EvalResult], bool],
    pareto_axes: tuple[str, str],
    max_steps: int = 50,
    frontier_size: int = 10,
    stagnation_k: int = 5,
) -> list[EvalResult]:
    """
    Run the propose-evaluate-accept loop with Pareto frontier tracking.

    Args:
        propose: feedback_str -> artifact_str. Receives "" on first call.
        evaluate: artifact_str -> EvalResult.
        accept: stopping criterion; return True to halt early.
        pareto_axes: names of (primary, secondary) score dimensions (for logging).
        max_steps: hard iteration cap.
        frontier_size: max entries in Pareto frontier.
        stagnation_k: halt if frontier artifact set unchanged for this many consecutive steps.

    Returns:
        Current Pareto frontier at termination.
    """
    frontier: list[EvalResult] = []
    feedback = ""
    stagnation_count = 0
    prev_artifacts: set[str] = set()
    prev_best_score: float = -1.0

    for _step in range(max_steps):
        artifact = propose(feedback)
        result = evaluate(artifact)
        frontier = _update_frontier(frontier, result, frontier_size)
        feedback = result.asi

        if accept(result):
            return frontier

        current_artifacts = {e.artifact for e in frontier}
        current_best = frontier[0].score if frontier else 0.0
        if current_artifacts == prev_artifacts and current_best <= prev_best_score:
            stagnation_count += 1
            if stagnation_count >= stagnation_k:
                return frontier
        else:
            stagnation_count = 0
        prev_artifacts = current_artifacts
        prev_best_score = current_best

    return frontier
