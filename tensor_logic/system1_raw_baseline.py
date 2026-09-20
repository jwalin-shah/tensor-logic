"""Raw-text lower-bound baseline for System-1 representation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from statistics import mean
from typing import Sequence

import torch

from .decision_benchmark import (
    BenchmarkMetrics,
    DecisionResult,
    distribution_soft_accuracy,
    evaluate_results,
)
from .system1_backends import MiniJevBackend
from .system1_tasks import PairedDecisionCases
from .system1_training import (
    CHOICE_QUESTIONS,
    NOUL_QUESTIONS,
    SCORE_QUESTIONS,
    soft_teacher_loss,
    teacher_tensors,
)
from .typed_decision import TensorDecisionModel


@dataclass(frozen=True)
class RawBaselineEvaluation:
    split: str
    cases: int
    decisions: int
    metrics: BenchmarkMetrics
    mean_soft_accuracy: float | None


def hashed_char_ngrams(
    text: str,
    *,
    dim: int = 512,
    min_n: int = 3,
    max_n: int = 5,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    if min_n <= 0 or max_n < min_n:
        raise ValueError("invalid n-gram range")

    normalized = " ".join(text.lower().split())
    vector = torch.zeros(dim, dtype=torch.float32)

    for n in range(min_n, max_n + 1):
        if len(normalized) < n:
            continue
        for start in range(len(normalized) - n + 1):
            gram = normalized[start : start + n].encode("utf-8")
            digest = hashlib.blake2b(gram, digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % dim
            sign = 1.0 if ((raw >> 8) & 1) else -1.0
            vector[index] += sign

    norm = torch.linalg.vector_norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def build_raw_text_baseline(
    *,
    dim: int = 512,
) -> TensorDecisionModel:
    return TensorDecisionModel(
        dim,
        choice_questions=CHOICE_QUESTIONS,
        score_questions=SCORE_QUESTIONS,
        noul_questions=NOUL_QUESTIONS,
    )


def train_raw_text_baseline(
    pairs: Sequence[PairedDecisionCases],
    *,
    split: str = "train",
    dim: int = 512,
    epochs: int = 250,
    learning_rate: float = 0.08,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> tuple[TensorDecisionModel, tuple[float, ...]]:
    selected = [
        pair for pair in pairs
        if pair.scenario.split == split
    ]
    if not selected:
        raise ValueError(f"no cases for split {split!r}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    model = build_raw_text_baseline(dim=dim)
    features = torch.stack(
        [
            hashed_char_ngrams(
                str(pair.raw.state),
                dim=dim,
            )
            for pair in selected
        ]
    )
    teachers = teacher_tensors(pairs, split=split)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    losses: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = soft_teacher_loss(outputs, teachers)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return model, tuple(losses)


def evaluate_raw_text_baseline(
    model: TensorDecisionModel,
    pairs: Sequence[PairedDecisionCases],
    *,
    split: str,
    dim: int = 512,
    model_version: str = "hashed-char-v1",
) -> RawBaselineEvaluation:
    cases = [
        pair.raw
        for pair in pairs
        if pair.scenario.split == split
    ]
    if not cases:
        raise ValueError(f"no cases for split {split!r}")

    backend = MiniJevBackend(
        model,
        lambda state: hashed_char_ngrams(
            str(state),
            dim=dim,
        ),
        model_version=model_version,
    )

    results: list[DecisionResult] = []
    soft_scores: list[float] = []

    for case in cases:
        rows = backend.run(case)
        results.extend(rows)
        for row in rows:
            score = distribution_soft_accuracy(case, row)
            if score is not None:
                soft_scores.append(score)

    metrics = evaluate_results(cases, results)
    return RawBaselineEvaluation(
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
