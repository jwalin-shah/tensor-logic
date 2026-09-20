"""Vectorized cognitive operators over cognitive tensors.

These operators expose bounded, inspectable computations corresponding to
working-memory selection, procedural choice, prediction error, metacognitive
escalation, and plan evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WorkingSetResult:
    indices: torch.Tensor
    scores: torch.Tensor


def select_working_set(
    *,
    relevance: torch.Tensor,
    activation: torch.Tensor,
    goal_alignment: torch.Tensor,
    contradiction_bonus: torch.Tensor | None = None,
    capacity: int = 7,
    relevance_weight: float = 0.45,
    activation_weight: float = 0.30,
    goal_weight: float = 0.25,
    contradiction_weight: float = 0.20,
) -> WorkingSetResult:
    """Select a bounded task working set from candidate memory items."""
    vectors = (relevance, activation, goal_alignment)
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("working-memory inputs must be 1D")
    if len({vector.shape[0] for vector in vectors}) != 1:
        raise ValueError("working-memory inputs must have equal length")
    if not 0 < capacity <= relevance.shape[0]:
        raise ValueError("capacity must be within candidate count")

    score = (
        relevance_weight * relevance
        + activation_weight * activation
        + goal_weight * goal_alignment
    )
    if contradiction_bonus is not None:
        if contradiction_bonus.shape != relevance.shape:
            raise ValueError("contradiction bonus shape mismatch")
        score = score + contradiction_weight * contradiction_bonus

    scores, indices = torch.topk(score, capacity)
    return WorkingSetResult(indices=indices, scores=scores)


def score_operators(
    *,
    goal_value: torch.Tensor,
    success_probability: torch.Tensor,
    expected_information_gain: torch.Tensor,
    resource_cost: torch.Tensor,
    uncertainty: torch.Tensor,
    information_weight: float = 0.20,
    resource_weight: float = 0.15,
    uncertainty_weight: float = 0.25,
) -> torch.Tensor:
    """Expected operator utility for fast procedural selection."""
    tensors = (
        goal_value,
        success_probability,
        expected_information_gain,
        resource_cost,
        uncertainty,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("operator features must be 1D")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("operator feature lengths must match")

    return (
        goal_value * success_probability
        + information_weight * expected_information_gain
        - resource_weight * resource_cost
        - uncertainty_weight * uncertainty
    )


def prediction_error(
    predicted: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Signed prediction error preserved for downstream learning."""
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed shapes must match")
    return observed - predicted


def metacognitive_escalation(
    *,
    confidence: torch.Tensor,
    contradiction: torch.Tensor,
    source_staleness: torch.Tensor,
    expected_regret: torch.Tensor,
    resource_pressure: torch.Tensor,
    threshold: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return escalation score and Boolean escalation decision.

    Higher contradiction, staleness, regret and lower confidence increase the
    score. Resource pressure raises the bar slightly, modeling bounded rationality.
    """
    tensors = (
        confidence,
        contradiction,
        source_staleness,
        expected_regret,
        resource_pressure,
    )
    if any(tensor.shape != confidence.shape for tensor in tensors):
        raise ValueError("metacognitive inputs must share shape")

    score = (
        0.35 * (1.0 - confidence)
        + 0.25 * contradiction
        + 0.15 * source_staleness
        + 0.25 * expected_regret
        - 0.10 * resource_pressure
    )
    return score, score >= threshold


def evaluate_plan_steps(
    *,
    reward: torch.Tensor,
    probability: torch.Tensor,
    time_cost: torch.Tensor,
    resource_cost: torch.Tensor,
    risk: torch.Tensor,
    time_weight: float = 0.10,
    resource_weight: float = 0.10,
    risk_weight: float = 0.25,
) -> torch.Tensor:
    """Bounded model-based value for candidate simulated plan steps."""
    tensors = (reward, probability, time_cost, resource_cost, risk)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("plan features must be 1D")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("plan feature lengths must match")

    return (
        reward * probability
        - time_weight * time_cost
        - resource_weight * resource_cost
        - risk_weight * risk
    )
