"""Backend-neutral typed decision benchmark primitives.

This module is deliberately agnostic to Laya, Jev, classifiers, and LLMs.
Every backend normalizes its output into the same probability-bearing schema,
so calibration, selective automation, latency, and OOD behavior can be compared
without conflating provider-specific APIs with evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable, Literal, Mapping, Sequence


Primitive = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class DecisionQuestion:
    question_id: str
    primitive: Primitive
    labels: tuple[str, ...]
    instructions: str = ""

    def __post_init__(self) -> None:
        if self.primitive in {"choice", "score"} and len(self.labels) < 2:
            raise ValueError("choice/score questions require >=2 labels")
        if self.primitive == "noul" and self.labels not in {
            ("false", "true"),
            ("no", "yes"),
        }:
            raise ValueError(
                "noul labels must be ('false','true') or ('no','yes')"
            )


@dataclass(frozen=True)
class DecisionCase:
    case_id: str
    state: object
    questions: tuple[DecisionQuestion, ...]
    targets: Mapping[str, str]
    target_distributions: Mapping[str, Mapping[str, float]] | None = None
    split: str = "test"
    domain: str = "synthetic"


@dataclass(frozen=True)
class DecisionResult:
    case_id: str
    question_id: str
    primitive: Primitive
    labels: tuple[str, ...]
    prediction: str
    probabilities: Mapping[str, float]
    confidence: float
    latency_ms: float
    backend: str
    model: str
    model_version: str
    calibration_version: str = "none"

    def __post_init__(self) -> None:
        if self.prediction not in self.labels:
            raise ValueError("prediction is not a legal label")
        if set(self.probabilities) != set(self.labels):
            raise ValueError("probabilities must cover exactly the legal labels")
        values = tuple(float(self.probabilities[label]) for label in self.labels)
        if any((value < 0.0 or value > 1.0) for value in values):
            raise ValueError("probabilities must be between 0 and 1")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("probabilities must sum to 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.latency_ms < 0.0:
            raise ValueError("latency cannot be negative")


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    accuracy: float | None
    risk: float | None


@dataclass(frozen=True)
class BenchmarkMetrics:
    count: int
    accuracy: float
    brier: float
    nll: float
    ece: float
    p50_latency_ms: float
    p95_latency_ms: float
    selective_curve: tuple[SelectivePoint, ...]


def evaluate_results(
    cases: Iterable[DecisionCase],
    results: Iterable[DecisionResult],
    *,
    ece_bins: int = 10,
    thresholds: Sequence[float] = (
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
    ),
) -> BenchmarkMetrics:
    case_by_id = {case.case_id: case for case in cases}
    rows = list(results)
    if not rows:
        raise ValueError("no benchmark results")
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")

    correct: list[float] = []
    confidences: list[float] = []
    brier_terms: list[float] = []
    nll_terms: list[float] = []
    latencies: list[float] = []

    for result in rows:
        case = case_by_id.get(result.case_id)
        if case is None:
            raise ValueError(f"unknown case: {result.case_id}")
        target = case.targets.get(result.question_id)
        if target is None:
            raise ValueError(
                f"case {result.case_id} has no target for "
                f"{result.question_id}"
            )
        if target not in result.labels:
            raise ValueError("target is not a legal label")

        is_correct = float(result.prediction == target)
        correct.append(is_correct)
        confidences.append(result.confidence)
        latencies.append(result.latency_ms)

        brier = 0.0
        for label in result.labels:
            expected = 1.0 if label == target else 0.0
            probability = float(result.probabilities[label])
            brier += (probability - expected) ** 2
        brier_terms.append(brier)

        target_probability = max(
            float(result.probabilities[target]),
            1e-12,
        )
        nll_terms.append(-math.log(target_probability))

    accuracy = sum(correct) / len(correct)
    brier = sum(brier_terms) / len(brier_terms)
    nll = sum(nll_terms) / len(nll_terms)
    ece = _ece(confidences, correct, bins=ece_bins)

    sorted_latency = sorted(latencies)
    p50 = float(median(sorted_latency))
    p95 = _percentile(sorted_latency, 0.95)

    selective = tuple(
        _selective_point(
            threshold,
            confidences,
            correct,
        )
        for threshold in thresholds
    )

    return BenchmarkMetrics(
        count=len(rows),
        accuracy=accuracy,
        brier=brier,
        nll=nll,
        ece=ece,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        selective_curve=selective,
    )


def distribution_soft_accuracy(
    case: DecisionCase,
    result: DecisionResult,
) -> float | None:
    """Overlap between predicted and teacher distributions, if available."""
    if case.target_distributions is None:
        return None
    teacher = case.target_distributions.get(result.question_id)
    if teacher is None:
        return None
    if set(teacher) != set(result.labels):
        raise ValueError("teacher distribution labels do not match result")
    return sum(
        min(float(teacher[label]), float(result.probabilities[label]))
        for label in result.labels
    )


def option_order_changed(
    original: DecisionResult,
    permuted: DecisionResult,
) -> bool:
    if original.case_id != permuted.case_id:
        raise ValueError("case IDs differ")
    if original.question_id != permuted.question_id:
        raise ValueError("question IDs differ")
    if set(original.labels) != set(permuted.labels):
        raise ValueError("legal label sets differ")
    return original.prediction != permuted.prediction


def _ece(
    confidences: Sequence[float],
    correct: Sequence[float],
    *,
    bins: int,
) -> float:
    total = len(confidences)
    value = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        if index == bins - 1:
            members = [
                i for i, confidence in enumerate(confidences)
                if lo <= confidence <= hi
            ]
        else:
            members = [
                i for i, confidence in enumerate(confidences)
                if lo <= confidence < hi
            ]
        if not members:
            continue
        bucket_confidence = sum(confidences[i] for i in members) / len(members)
        bucket_accuracy = sum(correct[i] for i in members) / len(members)
        value += (len(members) / total) * abs(
            bucket_confidence - bucket_accuracy
        )
    return value


def _selective_point(
    threshold: float,
    confidences: Sequence[float],
    correct: Sequence[float],
) -> SelectivePoint:
    indices = [
        index
        for index, confidence in enumerate(confidences)
        if confidence >= threshold
    ]
    coverage = len(indices) / len(confidences)
    if not indices:
        return SelectivePoint(
            threshold=threshold,
            coverage=0.0,
            accuracy=None,
            risk=None,
        )
    accuracy = sum(correct[index] for index in indices) / len(indices)
    return SelectivePoint(
        threshold=threshold,
        coverage=coverage,
        accuracy=accuracy,
        risk=1.0 - accuracy,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("no values")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )
