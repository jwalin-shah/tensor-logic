"""Resource-rational metacognitive control.

The controller chooses between acting now, thinking longer, and escalating.
It exposes the quantities instead of hiding them inside an agent prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


MetaAction = Literal["ACT", "THINK", "ESCALATE"]


@dataclass(frozen=True)
class MetaState:
    confidence: float
    contradiction: float
    source_staleness: float
    error_cost: float
    compute_budget: float

    def __post_init__(self) -> None:
        for name in (
            "confidence",
            "contradiction",
            "source_staleness",
            "compute_budget",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.error_cost < 0.0:
            raise ValueError("error_cost cannot be negative")


@dataclass(frozen=True)
class ComputationOption:
    action: MetaAction
    expected_accuracy: float
    compute_cost: float
    latency_cost: float = 0.0
    escalation_cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.expected_accuracy <= 1.0:
            raise ValueError("expected_accuracy must be in [0,1]")
        if min(
            self.compute_cost,
            self.latency_cost,
            self.escalation_cost,
        ) < 0.0:
            raise ValueError("costs cannot be negative")


@dataclass(frozen=True)
class MetaDecision:
    action: MetaAction
    utility: float
    utilities: Mapping[MetaAction, float]


def expected_utility(
    state: MetaState,
    option: ComputationOption,
) -> float:
    """Expected downstream utility net of reasoning cost.

    The risk multiplier makes contradictions and stale sources raise the
    expected cost of acting on an error.
    """
    uncertainty_risk = (
        1.0
        + 0.50 * state.contradiction
        + 0.35 * state.source_staleness
    )
    expected_error_loss = (
        (1.0 - option.expected_accuracy)
        * state.error_cost
        * uncertainty_risk
    )

    # Low budget makes additional compute relatively more expensive.
    budget_pressure = 1.0 + (1.0 - state.compute_budget)
    reasoning_cost = (
        option.compute_cost * budget_pressure
        + option.latency_cost
        + option.escalation_cost
    )
    return -expected_error_loss - reasoning_cost


def choose_meta_action(
    state: MetaState,
    options: tuple[ComputationOption, ...],
) -> MetaDecision:
    if not options:
        raise ValueError("at least one computation option is required")
    by_action: dict[MetaAction, float] = {}
    for option in options:
        if option.action in by_action:
            raise ValueError("duplicate action option")
        by_action[option.action] = expected_utility(state, option)

    winner = max(
        by_action,
        key=lambda action: (by_action[action], action),
    )
    return MetaDecision(
        action=winner,
        utility=by_action[winner],
        utilities=dict(by_action),
    )


def value_of_computation(
    *,
    act_accuracy: float,
    improved_accuracy: float,
    error_cost: float,
    computation_cost: float,
) -> float:
    """Expected benefit of additional reasoning minus its cost."""
    if not 0.0 <= act_accuracy <= 1.0:
        raise ValueError("act_accuracy must be in [0,1]")
    if not 0.0 <= improved_accuracy <= 1.0:
        raise ValueError("improved_accuracy must be in [0,1]")
    if error_cost < 0.0 or computation_cost < 0.0:
        raise ValueError("costs cannot be negative")
    avoided_error = (
        max(improved_accuracy - act_accuracy, 0.0)
        * error_cost
    )
    return avoided_error - computation_cost


def fixed_confidence_policy(
    confidence: float,
    *,
    think_threshold: float = 0.80,
    escalate_threshold: float = 0.50,
) -> MetaAction:
    """Simple baseline used by exp96."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    if not 0.0 <= escalate_threshold <= think_threshold <= 1.0:
        raise ValueError("invalid thresholds")
    if confidence >= think_threshold:
        return "ACT"
    if confidence >= escalate_threshold:
        return "THINK"
    return "ESCALATE"
