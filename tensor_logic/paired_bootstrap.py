"""Paired bootstrap uncertainty for raw-vs-structured decision experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

from .decision_benchmark import DecisionResult
from .system1_tasks import PairedDecisionCases


@dataclass(frozen=True)
class BootstrapInterval:
    metric: str
    mean_delta: float
    lower: float
    upper: float
    confidence: float
    samples: int


@dataclass(frozen=True)
class PairedBootstrapReport:
    right_minus_left: tuple[BootstrapInterval, ...]
    scenario_count: int
    decision_count_per_side: int
    seed: int


def paired_bootstrap_report(
    pairs: Sequence[PairedDecisionCases],
    left_results: Sequence[DecisionResult],
    right_results: Sequence[DecisionResult],
    *,
    split: str,
    left_representation: str = "raw",
    right_representation: str = "structured",
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapReport:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if left_representation not in {"raw", "structured"}:
        raise ValueError("invalid left representation")
    if right_representation not in {"raw", "structured"}:
        raise ValueError("invalid right representation")

    selected = [
        pair for pair in pairs
        if pair.scenario.split == split
    ]
    if not selected:
        raise ValueError(f"no pairs for split {split!r}")

    left_by_key = _result_map(left_results)
    right_by_key = _result_map(right_results)

    per_scenario: dict[str, dict[str, float]] = {}
    left_decisions = 0
    right_decisions = 0

    for pair in selected:
        scenario_id = pair.scenario.scenario_id
        left_case = getattr(pair, left_representation)
        right_case = getattr(pair, right_representation)

        if tuple(q.question_id for q in left_case.questions) != tuple(
            q.question_id for q in right_case.questions
        ):
            raise ValueError("paired cases have different question sets")

        left_metrics = []
        right_metrics = []
        for question in left_case.questions:
            question_id = question.question_id
            left_row = left_by_key.get(
                (left_case.case_id, question_id)
            )
            right_row = right_by_key.get(
                (right_case.case_id, question_id)
            )
            if left_row is None or right_row is None:
                raise ValueError(
                    f"missing paired result for {scenario_id}:{question_id}"
                )

            left_target = left_case.targets[question_id]
            right_target = right_case.targets[question_id]
            if left_target != right_target:
                raise ValueError("paired cases have different targets")

            left_metrics.append(_decision_metrics(left_row, left_target))
            right_metrics.append(_decision_metrics(right_row, right_target))
            left_decisions += 1
            right_decisions += 1

        per_scenario[scenario_id] = {
            metric: (
                sum(row[metric] for row in right_metrics)
                - sum(row[metric] for row in left_metrics)
            ) / len(left_metrics)
            for metric in ("accuracy", "nll", "brier")
        }

    if left_decisions != right_decisions:
        raise AssertionError("paired sides produced different decision counts")

    scenario_ids = tuple(sorted(per_scenario))
    rng = random.Random(seed)
    bootstrap: dict[str, list[float]] = {
        "accuracy": [],
        "nll": [],
        "brier": [],
    }

    for _ in range(iterations):
        sample = [
            rng.choice(scenario_ids)
            for _ in range(len(scenario_ids))
        ]
        for metric in bootstrap:
            bootstrap[metric].append(
                sum(per_scenario[item][metric] for item in sample)
                / len(sample)
            )

    alpha = (1.0 - confidence) / 2.0
    intervals = []
    for metric in ("accuracy", "nll", "brier"):
        observed = (
            sum(per_scenario[item][metric] for item in scenario_ids)
            / len(scenario_ids)
        )
        values = sorted(bootstrap[metric])
        intervals.append(
            BootstrapInterval(
                metric=metric,
                mean_delta=observed,
                lower=_quantile(values, alpha),
                upper=_quantile(values, 1.0 - alpha),
                confidence=confidence,
                samples=iterations,
            )
        )

    return PairedBootstrapReport(
        right_minus_left=tuple(intervals),
        scenario_count=len(scenario_ids),
        decision_count_per_side=left_decisions,
        seed=seed,
    )


def _result_map(
    rows: Sequence[DecisionResult],
) -> dict[tuple[str, str], DecisionResult]:
    out: dict[tuple[str, str], DecisionResult] = {}
    for row in rows:
        key = (row.case_id, row.question_id)
        if key in out:
            raise ValueError(f"duplicate result: {key}")
        out[key] = row
    return out


def _decision_metrics(
    row: DecisionResult,
    target: str,
) -> dict[str, float]:
    target_probability = max(
        float(row.probabilities[target]),
        1e-12,
    )
    brier = sum(
        (
            float(row.probabilities[label])
            - (1.0 if label == target else 0.0)
        )
        ** 2
        for label in row.labels
    )
    return {
        "accuracy": float(row.prediction == target),
        "nll": -math.log(target_probability),
        "brier": brier,
    }


def _quantile(
    sorted_values: Sequence[float],
    q: float,
) -> float:
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )
