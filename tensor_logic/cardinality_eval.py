"""Grouped evaluation helpers for System-1 cardinality experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .decision_benchmark import (
    BenchmarkMetrics,
    DecisionCase,
    DecisionResult,
    evaluate_results,
)


@dataclass(frozen=True)
class CardinalityMetrics:
    option_count: int
    cases: int
    decisions: int
    metrics: BenchmarkMetrics


def evaluate_by_option_count(
    cases: Sequence[DecisionCase],
    results: Sequence[DecisionResult],
) -> tuple[CardinalityMetrics, ...]:
    case_by_id = {case.case_id: case for case in cases}
    grouped_cases: dict[int, list[DecisionCase]] = {}
    grouped_results: dict[int, list[DecisionResult]] = {}

    for case in cases:
        counts = {
            len(question.labels)
            for question in case.questions
            if question.primitive == "choice"
        }
        if len(counts) != 1:
            raise ValueError(
                "cardinality evaluator requires exactly one choice "
                "option count per case"
            )
        option_count = next(iter(counts))
        grouped_cases.setdefault(option_count, []).append(case)

    for result in results:
        case = case_by_id.get(result.case_id)
        if case is None:
            raise ValueError(f"unknown result case: {result.case_id}")
        if result.primitive != "choice":
            raise ValueError(
                "cardinality evaluator only accepts choice results"
            )
        option_count = len(result.labels)
        grouped_results.setdefault(option_count, []).append(result)

    rows: list[CardinalityMetrics] = []
    for option_count in sorted(grouped_cases):
        case_group = grouped_cases[option_count]
        result_group = grouped_results.get(option_count, [])
        if not result_group:
            raise ValueError(
                f"no results for option count {option_count}"
            )
        metrics = evaluate_results(case_group, result_group)
        rows.append(
            CardinalityMetrics(
                option_count=option_count,
                cases=len(case_group),
                decisions=len(result_group),
                metrics=metrics,
            )
        )
    return tuple(rows)


def cardinality_degradation(
    grouped: Sequence[CardinalityMetrics],
    *,
    metric: str = "accuracy",
) -> Mapping[int, float]:
    if not grouped:
        return {}
    values: dict[int, float] = {}
    for row in grouped:
        value = getattr(row.metrics, metric, None)
        if value is None or not isinstance(value, (int, float)):
            raise ValueError(
                f"metric {metric!r} is not a scalar BenchmarkMetrics field"
            )
        values[row.option_count] = float(value)

    smallest = min(values)
    baseline = values[smallest]
    return {
        option_count: value - baseline
        for option_count, value in sorted(values.items())
    }
