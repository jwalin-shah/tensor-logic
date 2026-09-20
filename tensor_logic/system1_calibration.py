"""Backend-neutral post-hoc calibration for typed decision results.

Temperature scaling is applied directly to probability distributions, so it can
calibrate Laya, MiniJev, classifiers, or constrained LLM outputs without access
to their internal logits. For a distribution p:

    p_T(i) = p(i)^(1/T) / sum_j p(j)^(1/T)

This is equivalent to ordinary logit temperature scaling up to an additive
normalization constant.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

from .decision_benchmark import (
    DecisionCase,
    DecisionResult,
    evaluate_results,
)


@dataclass(frozen=True)
class QuestionTemperature:
    question_id: str
    temperature: float
    dev_nll_before: float
    dev_nll_after: float

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class TemperatureCalibration:
    by_question: tuple[QuestionTemperature, ...]

    @property
    def version(self) -> str:
        payload = [
            {
                "question_id": row.question_id,
                "temperature": row.temperature,
            }
            for row in self.by_question
        ]
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "temperature:" + hashlib.sha256(raw).hexdigest()[:16]

    def temperature_for(self, question_id: str) -> float:
        for row in self.by_question:
            if row.question_id == question_id:
                return row.temperature
        raise KeyError(question_id)


def fit_temperature_calibration(
    cases: Iterable[DecisionCase],
    results: Iterable[DecisionResult],
    *,
    temperatures: Sequence[float] | None = None,
) -> TemperatureCalibration:
    cases_by_id = {case.case_id: case for case in cases}
    rows = list(results)
    if not rows:
        raise ValueError("no results to calibrate")

    candidates = tuple(
        temperatures
        if temperatures is not None
        else _default_temperature_grid()
    )
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("temperature candidates must be positive")

    grouped: dict[str, list[DecisionResult]] = {}
    for row in rows:
        grouped.setdefault(row.question_id, []).append(row)

    fitted: list[QuestionTemperature] = []
    for question_id in sorted(grouped):
        question_rows = grouped[question_id]
        before = _mean_nll(
            question_rows,
            cases_by_id,
            temperature=1.0,
        )

        scored = [
            (
                _mean_nll(
                    question_rows,
                    cases_by_id,
                    temperature=temperature,
                ),
                float(temperature),
            )
            for temperature in candidates
        ]
        best_nll, best_temperature = min(
            scored,
            key=lambda item: (item[0], abs(math.log(item[1]))),
        )
        fitted.append(
            QuestionTemperature(
                question_id=question_id,
                temperature=best_temperature,
                dev_nll_before=before,
                dev_nll_after=best_nll,
            )
        )

    return TemperatureCalibration(by_question=tuple(fitted))


def apply_temperature_calibration(
    calibration: TemperatureCalibration,
    results: Iterable[DecisionResult],
) -> tuple[DecisionResult, ...]:
    calibrated: list[DecisionResult] = []

    for row in results:
        temperature = calibration.temperature_for(row.question_id)
        probabilities = temperature_scale_distribution(
            row.probabilities,
            temperature=temperature,
        )
        prediction = max(
            row.labels,
            key=lambda label: probabilities[label],
        )
        confidence = probabilities[prediction]

        calibrated.append(
            DecisionResult(
                case_id=row.case_id,
                question_id=row.question_id,
                primitive=row.primitive,
                labels=row.labels,
                prediction=prediction,
                probabilities=probabilities,
                confidence=confidence,
                latency_ms=row.latency_ms,
                backend=row.backend,
                model=row.model,
                model_version=row.model_version,
                calibration_version=calibration.version,
            )
        )

    return tuple(calibrated)


def temperature_scale_distribution(
    probabilities: Mapping[str, float],
    *,
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not probabilities:
        raise ValueError("probabilities cannot be empty")

    labels = tuple(probabilities)
    values = [float(probabilities[label]) for label in labels]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("probabilities must be in [0,1]")
    if not math.isclose(
        sum(values),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError("probabilities must sum to 1")

    floor = 1e-12
    log_scaled = [
        math.log(max(value, floor)) / temperature
        for value in values
    ]
    maximum = max(log_scaled)
    weights = [math.exp(value - maximum) for value in log_scaled]
    total = sum(weights)

    return {
        label: weight / total
        for label, weight in zip(labels, weights)
    }


def calibration_report(
    cases: Iterable[DecisionCase],
    raw_results: Iterable[DecisionResult],
    calibration: TemperatureCalibration,
) -> dict[str, object]:
    cases_tuple = tuple(cases)
    raw_tuple = tuple(raw_results)
    calibrated = apply_temperature_calibration(
        calibration,
        raw_tuple,
    )
    before = evaluate_results(cases_tuple, raw_tuple)
    after = evaluate_results(cases_tuple, calibrated)

    return {
        "calibration_version": calibration.version,
        "questions": [
            {
                "question_id": row.question_id,
                "temperature": row.temperature,
                "dev_nll_before": row.dev_nll_before,
                "dev_nll_after": row.dev_nll_after,
            }
            for row in calibration.by_question
        ],
        "before": {
            "accuracy": before.accuracy,
            "brier": before.brier,
            "nll": before.nll,
            "ece": before.ece,
        },
        "after": {
            "accuracy": after.accuracy,
            "brier": after.brier,
            "nll": after.nll,
            "ece": after.ece,
        },
    }


def _mean_nll(
    results: Sequence[DecisionResult],
    cases_by_id: Mapping[str, DecisionCase],
    *,
    temperature: float,
) -> float:
    values: list[float] = []
    for row in results:
        case = cases_by_id.get(row.case_id)
        if case is None:
            raise ValueError(f"unknown case: {row.case_id}")
        target = case.targets.get(row.question_id)
        if target is None:
            raise ValueError(
                f"case {row.case_id} missing target for {row.question_id}"
            )
        scaled = temperature_scale_distribution(
            row.probabilities,
            temperature=temperature,
        )
        values.append(-math.log(max(scaled[target], 1e-12)))
    return sum(values) / len(values)


def _default_temperature_grid() -> tuple[float, ...]:
    # Deterministic log-spaced search from 0.25 to 4.0 inclusive.
    lower = math.log(0.25)
    upper = math.log(4.0)
    count = 161
    return tuple(
        math.exp(lower + (upper - lower) * index / (count - 1))
        for index in range(count)
    )
