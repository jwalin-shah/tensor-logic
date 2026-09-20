"""Evaluation contract for learned symbolic rule hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class RuleHypothesis:
    rule_id: str
    expression: str
    producer: str
    model_version: str
    source_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleSplitResult:
    split_id: str
    accuracy: float
    precision: float
    recall: float
    counterexamples: int
    examples: int

    def __post_init__(self) -> None:
        for name in ("accuracy", "precision", "recall"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.counterexamples < 0 or self.examples <= 0:
            raise ValueError("invalid split counts")
        if self.counterexamples > self.examples:
            raise ValueError("counterexamples exceed examples")


@dataclass(frozen=True)
class RuleEvaluation:
    mean_accuracy: float
    min_accuracy: float
    accuracy_std: float
    total_counterexamples: int
    stable: bool
    admission_candidate: bool
    reasons: tuple[str, ...]


def evaluate_rule_hypothesis(
    results: Sequence[RuleSplitResult],
    *,
    min_split_accuracy: float = 0.90,
    max_accuracy_std: float = 0.05,
    max_counterexamples: int = 0,
) -> RuleEvaluation:
    if len(results) < 2:
        raise ValueError("need at least two independent splits")
    accuracies = [row.accuracy for row in results]
    total_counterexamples = sum(row.counterexamples for row in results)
    std = pstdev(accuracies)
    minimum = min(accuracies)

    reasons: list[str] = []
    if minimum < min_split_accuracy:
        reasons.append("heldout_accuracy_below_threshold")
    if std > max_accuracy_std:
        reasons.append("split_instability")
    if total_counterexamples > max_counterexamples:
        reasons.append("counterexamples_exceed_threshold")

    stable = std <= max_accuracy_std
    return RuleEvaluation(
        mean_accuracy=mean(accuracies),
        min_accuracy=minimum,
        accuracy_std=std,
        total_counterexamples=total_counterexamples,
        stable=stable,
        admission_candidate=not reasons,
        reasons=tuple(reasons),
    )
