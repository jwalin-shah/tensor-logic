"""Training/evaluation utilities for the structured System-1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

import torch

from .decision_benchmark import (
    BenchmarkMetrics,
    DecisionCase,
    DecisionResult,
    distribution_soft_accuracy,
    evaluate_results,
)
from .system1_backends import MiniJevBackend
from .system1_tasks import (
    FEATURE_KEYS,
    PairedDecisionCases,
    SYSTEM1_QUESTIONS,
    structured_feature_vector,
)
from .typed_decision import TensorDecisionModel


CHOICE_QUESTIONS = {
    "route_model": ("fast", "deep"),
    "source_choice": (
        "local_cache",
        "direct_authority",
        "web_search",
        "human",
    ),
}
SCORE_QUESTIONS = {
    "priority": ("low", "medium", "high"),
}
NOUL_QUESTIONS = ("escalate", "auto_authorize")


@dataclass(frozen=True)
class TrainingHistory:
    losses: tuple[float, ...]


@dataclass(frozen=True)
class SystemOneEvaluation:
    split: str
    cases: int
    decisions: int
    metrics: BenchmarkMetrics
    mean_soft_accuracy: float | None


def build_structured_baseline() -> TensorDecisionModel:
    return TensorDecisionModel(
        len(FEATURE_KEYS),
        choice_questions=CHOICE_QUESTIONS,
        score_questions=SCORE_QUESTIONS,
        noul_questions=NOUL_QUESTIONS,
    )


def train_structured_baseline(
    pairs: Sequence[PairedDecisionCases],
    *,
    split: str = "train",
    epochs: int = 250,
    learning_rate: float = 0.08,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> tuple[TensorDecisionModel, TrainingHistory]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    model = build_structured_baseline()
    features, teachers = _training_tensors(pairs, split=split)
    if features.shape[0] == 0:
        raise ValueError(f"no cases for split {split!r}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    losses: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = _soft_teacher_loss(outputs, teachers)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return model, TrainingHistory(losses=tuple(losses))


def evaluate_structured_baseline(
    model: TensorDecisionModel,
    pairs: Sequence[PairedDecisionCases],
    *,
    split: str,
    model_version: str = "synthetic-v1",
) -> SystemOneEvaluation:
    cases = [
        pair.structured
        for pair in pairs
        if pair.scenario.split == split
    ]
    if not cases:
        raise ValueError(f"no cases for split {split!r}")

    backend = MiniJevBackend(
        model,
        lambda state: torch.tensor(
            structured_feature_vector(state),
            dtype=torch.float32,
        ),
        model_version=model_version,
    )

    results: list[DecisionResult] = []
    soft_scores: list[float] = []
    for case in cases:
        case_results = backend.run(case)
        results.extend(case_results)
        for result in case_results:
            score = distribution_soft_accuracy(case, result)
            if score is not None:
                soft_scores.append(score)

    metrics = evaluate_results(cases, results)
    return SystemOneEvaluation(
        split=split,
        cases=len(cases),
        decisions=len(results),
        metrics=metrics,
        mean_soft_accuracy=(
            mean(soft_scores)
            if soft_scores
            else None
        ),
    )


def _training_tensors(
    pairs: Sequence[PairedDecisionCases],
    *,
    split: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = [
        pair
        for pair in pairs
        if pair.scenario.split == split
    ]
    if not selected:
        return (
            torch.empty((0, len(FEATURE_KEYS)), dtype=torch.float32),
            {},
        )

    features = torch.tensor(
        [
            structured_feature_vector(pair.structured.state)
            for pair in selected
        ],
        dtype=torch.float32,
    )

    route = []
    source = []
    priority = []
    escalate = []
    auto = []

    for pair in selected:
        teacher = pair.structured.target_distributions
        if teacher is None:
            raise ValueError("training case missing teacher distributions")

        route.append(
            [
                float(teacher["route_model"][label])
                for label in CHOICE_QUESTIONS["route_model"]
            ]
        )
        source.append(
            [
                float(teacher["source_choice"][label])
                for label in CHOICE_QUESTIONS["source_choice"]
            ]
        )
        priority.append(
            [
                float(teacher["priority"][label])
                for label in SCORE_QUESTIONS["priority"]
            ]
        )
        escalate.append(float(teacher["escalate"]["true"]))
        auto.append(float(teacher["auto_authorize"]["true"]))

    # TensorDecisionModel pads choice heads to the widest choice question.
    choice_teacher = torch.zeros(
        (
            len(selected),
            len(CHOICE_QUESTIONS),
            max(len(labels) for labels in CHOICE_QUESTIONS.values()),
        ),
        dtype=torch.float32,
    )
    choice_teacher[:, 0, :2] = torch.tensor(route)
    choice_teacher[:, 1, :4] = torch.tensor(source)

    score_teacher = torch.tensor(priority, dtype=torch.float32).unsqueeze(1)
    noul_teacher = torch.tensor(
        list(zip(escalate, auto)),
        dtype=torch.float32,
    )

    return features, {
        "choice": choice_teacher,
        "score": score_teacher,
        "noul": noul_teacher,
    }


def _soft_teacher_loss(
    outputs: dict[str, torch.Tensor],
    teachers: dict[str, torch.Tensor],
) -> torch.Tensor:
    eps = 1e-8

    choice_prob = outputs["choice_probabilities"].clamp_min(eps)
    choice_loss = -(
        teachers["choice"] * torch.log(choice_prob)
    ).sum(dim=-1).mean()

    score_prob = outputs["score_probabilities"].clamp_min(eps)
    score_loss = -(
        teachers["score"] * torch.log(score_prob)
    ).sum(dim=-1).mean()

    noul_prob = outputs["noul_probability_yes"].clamp(
        min=eps,
        max=1.0 - eps,
    )
    target = teachers["noul"]
    noul_loss = -(
        target * torch.log(noul_prob)
        + (1.0 - target) * torch.log(1.0 - noul_prob)
    ).mean()

    return choice_loss + score_loss + noul_loss
