"""MiniJev v0: typed probabilistic decisions as tensor equations.

This is a Jev-like decision *surface*, not a reproduction of TypeSafe Jev's
undisclosed model architecture or RLCD training system.

Core equations:

Choice:
    logits[b,q,o] = einsum("bd,qod->bqo", H, W_choice) + bias
    p = softmax(masked_logits, dim=-1)

Score:
    logits[b,q,l] = einsum("bd,qld->bql", H, W_score) + bias
    p = softmax(masked_logits, dim=-1)
    score[b,q] = sum_l p[b,q,l] * level_index[l]

Noul:
    logit[b,q] = einsum("bd,qd->bq", H, W_noul) + bias
    p_yes = sigmoid(logit)

All outputs are typed numeric values with explicit probabilities. No text is
generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ChoiceResult:
    question: str
    option: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreResult:
    question: str
    score: float
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulResult:
    question: str
    probability_yes: float
    confidence: float


@dataclass(frozen=True)
class DecisionBatch:
    choices: tuple[tuple[ChoiceResult, ...], ...]
    scores: tuple[tuple[ScoreResult, ...], ...]
    nouls: tuple[tuple[NoulResult, ...], ...]


class TensorDecisionModel(nn.Module):
    """Shared state -> parallel typed decision heads."""

    def __init__(
        self,
        state_dim: int,
        *,
        choice_questions: Mapping[str, Sequence[str]] | None = None,
        score_questions: Mapping[str, Sequence[str]] | None = None,
        noul_questions: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.state_dim = state_dim
        self.choice_names = tuple((choice_questions or {}).keys())
        self.choice_options = tuple(
            tuple(options)
            for options in (choice_questions or {}).values()
        )
        self.score_names = tuple((score_questions or {}).keys())
        self.score_levels = tuple(
            tuple(levels)
            for levels in (score_questions or {}).values()
        )
        self.noul_names = tuple(noul_questions)

        for name, options in zip(self.choice_names, self.choice_options):
            if len(options) < 2:
                raise ValueError(
                    f"choice question {name!r} needs >=2 options"
                )
        for name, levels in zip(self.score_names, self.score_levels):
            if not 2 <= len(levels) <= 10:
                raise ValueError(
                    f"score question {name!r} needs 2-10 ordered levels"
                )

        self.choice_max_options = max(
            (len(options) for options in self.choice_options),
            default=0,
        )
        self.score_max_levels = max(
            (len(levels) for levels in self.score_levels),
            default=0,
        )

        self.choice_weight = nn.Parameter(
            torch.zeros(
                len(self.choice_names),
                self.choice_max_options,
                state_dim,
            )
        )
        self.choice_bias = nn.Parameter(
            torch.zeros(
                len(self.choice_names),
                self.choice_max_options,
            )
        )
        self.score_weight = nn.Parameter(
            torch.zeros(
                len(self.score_names),
                self.score_max_levels,
                state_dim,
            )
        )
        self.score_bias = nn.Parameter(
            torch.zeros(
                len(self.score_names),
                self.score_max_levels,
            )
        )
        self.noul_weight = nn.Parameter(
            torch.zeros(len(self.noul_names), state_dim)
        )
        self.noul_bias = nn.Parameter(
            torch.zeros(len(self.noul_names))
        )

        self.register_buffer(
            "choice_mask",
            _ragged_mask(
                [len(options) for options in self.choice_options],
                self.choice_max_options,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_mask",
            _ragged_mask(
                [len(levels) for levels in self.score_levels],
                self.score_max_levels,
            ),
            persistent=False,
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch,{self.state_dim}]"
            )

        batch = state.shape[0]
        device = state.device
        dtype = state.dtype

        if self.choice_names:
            choice_logits = torch.einsum(
                "bd,qod->bqo",
                state,
                self.choice_weight,
            ) + self.choice_bias.unsqueeze(0)
            choice_logits = choice_logits.masked_fill(
                ~self.choice_mask.unsqueeze(0),
                float("-inf"),
            )
            choice_probabilities = torch.softmax(
                choice_logits,
                dim=-1,
            )
        else:
            choice_logits = torch.empty(
                batch, 0, 0, device=device, dtype=dtype
            )
            choice_probabilities = choice_logits

        if self.score_names:
            score_logits = torch.einsum(
                "bd,qld->bql",
                state,
                self.score_weight,
            ) + self.score_bias.unsqueeze(0)
            score_logits = score_logits.masked_fill(
                ~self.score_mask.unsqueeze(0),
                float("-inf"),
            )
            score_probabilities = torch.softmax(
                score_logits,
                dim=-1,
            )
            level_index = torch.arange(
                self.score_max_levels,
                device=device,
                dtype=dtype,
            )
            score_expected = (
                score_probabilities * level_index
            ).sum(dim=-1)
        else:
            score_logits = torch.empty(
                batch, 0, 0, device=device, dtype=dtype
            )
            score_probabilities = score_logits
            score_expected = torch.empty(
                batch, 0, device=device, dtype=dtype
            )

        if self.noul_names:
            noul_logits = torch.einsum(
                "bd,qd->bq",
                state,
                self.noul_weight,
            ) + self.noul_bias.unsqueeze(0)
            noul_probability_yes = torch.sigmoid(noul_logits)
        else:
            noul_logits = torch.empty(
                batch, 0, device=device, dtype=dtype
            )
            noul_probability_yes = noul_logits

        return {
            "choice_logits": choice_logits,
            "choice_probabilities": choice_probabilities,
            "score_logits": score_logits,
            "score_probabilities": score_probabilities,
            "score_expected": score_expected,
            "noul_logits": noul_logits,
            "noul_probability_yes": noul_probability_yes,
        }

    @torch.no_grad()
    def decide(self, state: torch.Tensor) -> DecisionBatch:
        outputs = self.forward(state)
        choices_by_batch = []
        scores_by_batch = []
        nouls_by_batch = []

        for batch_index in range(state.shape[0]):
            choice_results = []
            for question_index, (
                question,
                options,
            ) in enumerate(
                zip(self.choice_names, self.choice_options)
            ):
                probs = outputs["choice_probabilities"][
                    batch_index, question_index, : len(options)
                ]
                winner_index = int(torch.argmax(probs).item())
                choice_results.append(
                    ChoiceResult(
                        question=question,
                        option=options[winner_index],
                        probabilities={
                            option: float(prob.item())
                            for option, prob in zip(options, probs)
                        },
                        confidence=float(
                            torch.max(probs).item()
                        ),
                    )
                )

            score_results = []
            for question_index, (
                question,
                levels,
            ) in enumerate(
                zip(self.score_names, self.score_levels)
            ):
                probs = outputs["score_probabilities"][
                    batch_index, question_index, : len(levels)
                ]
                score_results.append(
                    ScoreResult(
                        question=question,
                        score=float(
                            outputs["score_expected"][
                                batch_index,
                                question_index,
                            ].item()
                        ),
                        probabilities={
                            level: float(prob.item())
                            for level, prob in zip(levels, probs)
                        },
                        confidence=float(
                            torch.max(probs).item()
                        ),
                    )
                )

            noul_results = []
            for question_index, question in enumerate(
                self.noul_names
            ):
                probability_yes = float(
                    outputs["noul_probability_yes"][
                        batch_index,
                        question_index,
                    ].item()
                )
                noul_results.append(
                    NoulResult(
                        question=question,
                        probability_yes=probability_yes,
                        confidence=abs(
                            probability_yes - 0.5
                        ) * 2.0,
                    )
                )

            choices_by_batch.append(tuple(choice_results))
            scores_by_batch.append(tuple(score_results))
            nouls_by_batch.append(tuple(noul_results))

        return DecisionBatch(
            choices=tuple(choices_by_batch),
            scores=tuple(scores_by_batch),
            nouls=tuple(nouls_by_batch),
        )


def multiclass_brier(
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean multiclass Brier score."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [example,class]")
    one_hot = torch.nn.functional.one_hot(
        target,
        num_classes=probabilities.shape[1],
    ).to(probabilities.dtype)
    return ((probabilities - one_hot) ** 2).sum(dim=1).mean()


def binary_brier(
    probability_yes: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean binary Brier score."""
    return (
        (
            probability_yes.to(torch.float32)
            - target.to(torch.float32)
        )
        ** 2
    ).mean()


def expected_calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    *,
    bins: int = 10,
) -> torch.Tensor:
    """Standard fixed-bin ECE for scalar confidence values."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    confidence = confidence.to(torch.float32).flatten()
    correct = correct.to(torch.float32).flatten()
    if confidence.shape != correct.shape:
        raise ValueError("confidence/correct shapes must match")

    boundaries = torch.linspace(
        0.0,
        1.0,
        bins + 1,
        device=confidence.device,
    )
    total = max(confidence.numel(), 1)
    ece = torch.zeros((), device=confidence.device)
    for i in range(bins):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not torch.any(mask):
            continue
        bucket_confidence = confidence[mask].mean()
        bucket_accuracy = correct[mask].mean()
        ece = ece + (
            mask.sum().to(torch.float32) / total
        ) * torch.abs(bucket_confidence - bucket_accuracy)
    return ece


def _ragged_mask(
    lengths: Sequence[int],
    width: int,
) -> torch.Tensor:
    if not lengths:
        return torch.empty(0, 0, dtype=torch.bool)
    positions = torch.arange(width).unsqueeze(0)
    limits = torch.tensor(lengths).unsqueeze(1)
    return positions < limits
