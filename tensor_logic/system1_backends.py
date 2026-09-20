"""Adapters for typed System-1 decision backends.

The module keeps provider-specific API details outside the common evaluator.
Actual model packages are imported lazily so the benchmark core remains light.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping

import torch

from .decision_benchmark import (
    DecisionCase,
    DecisionQuestion,
    DecisionResult,
)
from .typed_decision import TensorDecisionModel


def laya_questions(
    questions: tuple[DecisionQuestion, ...],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for question in questions:
        if question.primitive == "choice":
            out[question.question_id] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": {label: label for label in question.labels},
            }
        elif question.primitive == "score":
            out[question.question_id] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.labels),
            }
        elif question.primitive == "noul":
            out[question.question_id] = {
                "type": "noul",
                "instructions": question.instructions,
            }
        else:
            raise ValueError(f"unsupported primitive: {question.primitive}")
    return out


def normalize_laya_response(
    case: DecisionCase,
    response: Mapping[str, Any],
    *,
    latency_ms: float,
    model: str,
    model_version: str = "unknown",
    calibration_version: str = "model",
) -> tuple[DecisionResult, ...]:
    answers = response.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("Laya response missing answers mapping")

    results: list[DecisionResult] = []
    for question in case.questions:
        answer = answers.get(question.question_id)
        if not isinstance(answer, Mapping):
            raise ValueError(
                f"Laya response missing answer for {question.question_id}"
            )

        if question.primitive == "choice":
            probabilities = {
                label: float(answer["probabilities"][label])
                for label in question.labels
            }
            prediction = str(answer["choice"])
            confidence = float(answer["confidence"])

        elif question.primitive == "score":
            raw = answer.get("probabilities", {})
            probabilities = {
                label: float(raw[str(index)])
                for index, label in enumerate(question.labels)
            }
            best_index = max(
                range(len(question.labels)),
                key=lambda index: probabilities[question.labels[index]],
            )
            prediction = question.labels[best_index]
            confidence = float(answer["confidence"])

        else:
            p_true = float(answer["noul"])
            probabilities = {
                question.labels[0]: 1.0 - p_true,
                question.labels[1]: p_true,
            }
            prediction = (
                question.labels[1]
                if p_true >= 0.5
                else question.labels[0]
            )
            confidence = float(answer["confidence"])

        results.append(
            DecisionResult(
                case_id=case.case_id,
                question_id=question.question_id,
                primitive=question.primitive,
                labels=question.labels,
                prediction=prediction,
                probabilities=probabilities,
                confidence=confidence,
                latency_ms=latency_ms,
                backend="laya",
                model=model,
                model_version=model_version,
                calibration_version=calibration_version,
            )
        )
    return tuple(results)


class LayaBackend:
    """Thin lazy wrapper around the public laya package."""

    def __init__(
        self,
        *,
        model_id: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        device: str | None = None,
    ) -> None:
        import laya

        self.model_id = model_id
        self.subfolder = subfolder
        self.agent = laya.load(
            model_id,
            subfolder=subfolder,
            device=device,
        )

    def run(self, case: DecisionCase) -> tuple[DecisionResult, ...]:
        start = time.perf_counter()
        response = self.agent.predict(
            case.state,
            laya_questions(case.questions),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        model_name = (
            self.model_id
            if self.subfolder is None
            else f"{self.model_id}:{self.subfolder}"
        )
        return normalize_laya_response(
            case,
            response,
            latency_ms=latency_ms,
            model=model_name,
        )


class MiniJevBackend:
    """Adapter for the local TensorDecisionModel."""

    def __init__(
        self,
        model: TensorDecisionModel,
        feature_fn: Callable[[object], torch.Tensor],
        *,
        model_version: str = "local",
    ) -> None:
        self.model = model
        self.feature_fn = feature_fn
        self.model_version = model_version

    def run(self, case: DecisionCase) -> tuple[DecisionResult, ...]:
        state = self.feature_fn(case.state)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        start = time.perf_counter()
        batch = self.model.decide(state)
        latency_ms = (time.perf_counter() - start) * 1000.0

        results: list[DecisionResult] = []
        choice_by_name = {
            row.question: row
            for row in batch.choices[0]
        }
        score_by_name = {
            row.question: row
            for row in batch.scores[0]
        }
        noul_by_name = {
            row.question: row
            for row in batch.nouls[0]
        }

        for question in case.questions:
            if question.primitive == "choice":
                row = choice_by_name[question.question_id]
                probabilities = dict(row.probabilities)
                prediction = row.option
                confidence = row.confidence
            elif question.primitive == "score":
                row = score_by_name[question.question_id]
                probabilities = dict(row.probabilities)
                prediction = max(
                    probabilities,
                    key=probabilities.get,
                )
                confidence = row.confidence
            else:
                row = noul_by_name[question.question_id]
                p_true = row.probability_yes
                probabilities = {
                    question.labels[0]: 1.0 - p_true,
                    question.labels[1]: p_true,
                }
                prediction = (
                    question.labels[1]
                    if p_true >= 0.5
                    else question.labels[0]
                )
                confidence = max(p_true, 1.0 - p_true)

            results.append(
                DecisionResult(
                    case_id=case.case_id,
                    question_id=question.question_id,
                    primitive=question.primitive,
                    labels=question.labels,
                    prediction=prediction,
                    probabilities=probabilities,
                    confidence=float(confidence),
                    latency_ms=latency_ms,
                    backend="minijev",
                    model="TensorDecisionModel",
                    model_version=self.model_version,
                )
            )

        return tuple(results)


class ProbabilityCallableBackend:
    """Adapter for classifiers or constrained LLMs that return distributions.

    callable_fn(state, questions) must return:
      {
        question_id: {
          "probabilities": {label: probability, ...},
          "prediction": optional legal label,
          "confidence": optional scalar,
        }
      }

    This intentionally refuses label-only outputs because calibration metrics
    require a full distribution.
    """

    def __init__(
        self,
        callable_fn: Callable[
            [object, tuple[DecisionQuestion, ...]],
            Mapping[str, Mapping[str, Any]],
        ],
        *,
        backend: str,
        model: str,
        model_version: str,
        calibration_version: str = "none",
    ) -> None:
        self.callable_fn = callable_fn
        self.backend = backend
        self.model = model
        self.model_version = model_version
        self.calibration_version = calibration_version

    def run(self, case: DecisionCase) -> tuple[DecisionResult, ...]:
        start = time.perf_counter()
        raw = self.callable_fn(case.state, case.questions)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rows: list[DecisionResult] = []

        for question in case.questions:
            answer = raw.get(question.question_id)
            if not isinstance(answer, Mapping):
                raise ValueError(
                    f"missing distribution for {question.question_id}"
                )
            probs_raw = answer.get("probabilities")
            if not isinstance(probs_raw, Mapping):
                raise ValueError(
                    f"{question.question_id} missing probabilities"
                )
            probabilities = {
                label: float(probs_raw[label])
                for label in question.labels
            }
            prediction = str(
                answer.get(
                    "prediction",
                    max(probabilities, key=probabilities.get),
                )
            )
            confidence = float(
                answer.get(
                    "confidence",
                    probabilities[prediction],
                )
            )
            rows.append(
                DecisionResult(
                    case_id=case.case_id,
                    question_id=question.question_id,
                    primitive=question.primitive,
                    labels=question.labels,
                    prediction=prediction,
                    probabilities=probabilities,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    backend=self.backend,
                    model=self.model,
                    model_version=self.model_version,
                    calibration_version=self.calibration_version,
                )
            )
        return tuple(rows)
