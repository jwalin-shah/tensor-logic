"""Deterministic paired System-1 benchmark tasks.

Each latent scenario is rendered twice:
- raw prose-like state
- structured feature state

Questions, hard targets, and teacher distributions are identical across the
pair. This isolates representation effects from decision-backend effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

from .decision_benchmark import DecisionCase, DecisionQuestion


FEATURE_KEYS = (
    "complexity",
    "ambiguity",
    "stakes",
    "confidence",
    "source_freshness",
    "contradiction",
    "human_authorized",
    "authority_available",
    "external_search_allowed",
    "commitment_pressure",
    "importance",
)


SYSTEM1_QUESTIONS = (
    DecisionQuestion(
        question_id="route_model",
        primitive="choice",
        labels=("fast", "deep"),
        instructions=(
            "Choose fast for a routine low-risk decision and deep when "
            "additional reasoning is warranted."
        ),
    ),
    DecisionQuestion(
        question_id="escalate",
        primitive="noul",
        labels=("false", "true"),
        instructions=(
            "Should this decision be escalated because confidence, evidence "
            "freshness, or contradictions make immediate action unsafe?"
        ),
    ),
    DecisionQuestion(
        question_id="auto_authorize",
        primitive="noul",
        labels=("false", "true"),
        instructions=(
            "Is this action safe to authorize automatically under the stated "
            "authorization and risk conditions?"
        ),
    ),
    DecisionQuestion(
        question_id="priority",
        primitive="score",
        labels=("low", "medium", "high"),
        instructions="How high should this item rank for attention?",
    ),
    DecisionQuestion(
        question_id="source_choice",
        primitive="choice",
        labels=(
            "local_cache",
            "direct_authority",
            "web_search",
            "human",
        ),
        instructions=(
            "Choose the best next evidence source for this decision."
        ),
    ),
)


@dataclass(frozen=True)
class SystemOneScenario:
    scenario_id: str
    split: str
    complexity: float
    ambiguity: float
    stakes: float
    confidence: float
    source_freshness: float
    contradiction: float
    human_authorized: bool
    authority_available: bool
    external_search_allowed: bool
    commitment_pressure: float
    importance: float

    def __post_init__(self) -> None:
        for name in (
            "complexity",
            "ambiguity",
            "stakes",
            "confidence",
            "source_freshness",
            "contradiction",
            "commitment_pressure",
            "importance",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class PairedDecisionCases:
    scenario: SystemOneScenario
    raw: DecisionCase
    structured: DecisionCase


def scenario_targets(
    scenario: SystemOneScenario,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    route_value = (
        0.45 * scenario.complexity
        + 0.30 * scenario.ambiguity
        + 0.25 * scenario.stakes
    )
    route_dist = _binary_distribution(route_value, threshold=0.55)
    route_target = "deep" if route_dist["true"] >= 0.5 else "fast"

    escalation_value = max(
        1.0 - scenario.confidence,
        1.0 - scenario.source_freshness,
        scenario.contradiction,
    )
    escalate_dist = _binary_distribution(
        escalation_value,
        threshold=0.55,
        scale=0.10,
    )
    escalate_target = (
        "true" if escalate_dist["true"] >= 0.5 else "false"
    )

    auto_score = (
        0.50 * (1.0 if scenario.human_authorized else 0.0)
        + 0.20 * scenario.source_freshness
        + 0.15 * (1.0 - scenario.stakes)
        + 0.15 * (1.0 - scenario.contradiction)
    )
    auto_dist = _binary_distribution(
        auto_score,
        threshold=0.55,
        scale=0.10,
    )
    auto_target = "true" if auto_dist["true"] >= 0.5 else "false"

    priority_value = (
        0.45 * scenario.importance
        + 0.35 * scenario.commitment_pressure
        + 0.20 * scenario.stakes
    )
    priority_dist = _ordinal_distribution(
        priority_value,
        centers=(0.15, 0.50, 0.85),
        labels=("low", "medium", "high"),
        scale=0.16,
    )
    priority_target = max(priority_dist, key=priority_dist.get)

    source_utilities = {
        "local_cache": (
            1.8 * scenario.source_freshness
            - 1.2 * scenario.contradiction
            - 0.2 * scenario.ambiguity
        ),
        "direct_authority": (
            (1.6 if scenario.authority_available else -3.0)
            + 1.2 * (1.0 - scenario.source_freshness)
            + 0.2 * scenario.stakes
        ),
        "web_search": (
            (1.2 if scenario.external_search_allowed else -3.0)
            + 1.0 * (1.0 - scenario.source_freshness)
            + 0.5 * scenario.ambiguity
        ),
        "human": (
            1.7 * scenario.contradiction
            + 0.8 * scenario.ambiguity
            + 0.5 * scenario.stakes
            + (0.4 if not scenario.human_authorized else 0.0)
        ),
    }
    source_dist = _softmax_dict(source_utilities, temperature=0.65)
    source_target = max(source_dist, key=source_dist.get)

    targets = {
        "route_model": route_target,
        "escalate": escalate_target,
        "auto_authorize": auto_target,
        "priority": priority_target,
        "source_choice": source_target,
    }
    distributions = {
        "route_model": {
            "fast": route_dist["false"],
            "deep": route_dist["true"],
        },
        "escalate": escalate_dist,
        "auto_authorize": auto_dist,
        "priority": priority_dist,
        "source_choice": source_dist,
    }
    return targets, distributions


def render_structured_state(
    scenario: SystemOneScenario,
) -> dict[str, object]:
    return {
        "complexity": scenario.complexity,
        "ambiguity": scenario.ambiguity,
        "stakes": scenario.stakes,
        "confidence": scenario.confidence,
        "source_freshness": scenario.source_freshness,
        "contradiction": scenario.contradiction,
        "human_authorized": scenario.human_authorized,
        "authority_available": scenario.authority_available,
        "external_search_allowed": scenario.external_search_allowed,
        "commitment_pressure": scenario.commitment_pressure,
        "importance": scenario.importance,
    }


def render_raw_state(
    scenario: SystemOneScenario,
) -> str:
    # Same facts as render_structured_state, expressed as prose.
    return (
        f"The request has complexity {scenario.complexity:.6f} and ambiguity "
        f"{scenario.ambiguity:.6f}. The stakes estimate is "
        f"{scenario.stakes:.6f}. Current decision confidence is "
        f"{scenario.confidence:.6f}. Evidence freshness is "
        f"{scenario.source_freshness:.6f}, while contradiction strength is "
        f"{scenario.contradiction:.6f}. Human authorization is "
        f"{'present' if scenario.human_authorized else 'absent'}. A direct "
        f"authority source is {'available' if scenario.authority_available else 'unavailable'} "
        f"and external search is "
        f"{'allowed' if scenario.external_search_allowed else 'not allowed'}. "
        f"Commitment pressure is {scenario.commitment_pressure:.6f} and "
        f"importance is {scenario.importance:.6f}."
    )


def feature_vector(
    scenario: SystemOneScenario,
) -> tuple[float, ...]:
    return structured_feature_vector(render_structured_state(scenario))


def structured_feature_vector(
    state: dict[str, object],
) -> tuple[float, ...]:
    missing = [key for key in FEATURE_KEYS if key not in state]
    if missing:
        raise ValueError(f"structured state missing features: {missing}")
    return tuple(float(state[key]) for key in FEATURE_KEYS)


def make_pair(
    scenario: SystemOneScenario,
) -> PairedDecisionCases:
    targets, distributions = scenario_targets(scenario)
    raw = DecisionCase(
        case_id=f"{scenario.scenario_id}::raw",
        state=render_raw_state(scenario),
        questions=SYSTEM1_QUESTIONS,
        targets=targets,
        target_distributions=distributions,
        split=scenario.split,
        domain="lifeops_system1_raw_synthetic",
    )
    structured = DecisionCase(
        case_id=f"{scenario.scenario_id}::structured",
        state=render_structured_state(scenario),
        questions=SYSTEM1_QUESTIONS,
        targets=targets,
        target_distributions=distributions,
        split=scenario.split,
        domain="lifeops_system1_structured_synthetic",
    )
    return PairedDecisionCases(
        scenario=scenario,
        raw=raw,
        structured=structured,
    )


def generate_pairs(
    *,
    seed: int = 23,
    iid_count: int = 120,
    ood_per_family: int = 30,
) -> tuple[PairedDecisionCases, ...]:
    rng = random.Random(seed)
    scenarios: list[SystemOneScenario] = []

    for index in range(iid_count):
        split = (
            "train"
            if index % 10 < 7
            else "dev"
            if index % 10 == 7
            else "test"
        )
        scenarios.append(
            _sample_scenario(
                rng,
                f"iid-{index:04d}",
                split,
            )
        )

    for family in (
        "ood_compositional",
        "ood_source_failure",
        "ood_confidence_shift",
    ):
        for index in range(ood_per_family):
            scenarios.append(
                _sample_ood_scenario(
                    rng,
                    f"{family}-{index:04d}",
                    family,
                )
            )

    return tuple(make_pair(scenario) for scenario in scenarios)


@dataclass(frozen=True)
class CardinalityPair:
    option_count: int
    raw: DecisionCase
    structured: DecisionCase


def generate_cardinality_pairs(
    *,
    seed: int = 29,
    option_counts: Sequence[int] = (2, 4, 8, 16, 32, 64),
    per_count: int = 8,
) -> tuple[CardinalityPair, ...]:
    rng = random.Random(seed)
    pairs: list[CardinalityPair] = []

    for option_count in option_counts:
        if option_count < 2:
            raise ValueError("option count must be >=2")
        labels = tuple(
            f"source_{index:02d}"
            for index in range(option_count)
        )
        question = DecisionQuestion(
            question_id="source_choice",
            primitive="choice",
            labels=labels,
            instructions=(
                "Choose the source with the best combination of relevance, "
                "freshness, and risk."
            ),
        )

        for case_index in range(per_count):
            candidates = []
            utilities = {}
            for label in labels:
                relevance = rng.random()
                freshness = rng.random()
                risk = rng.random()
                utility = 1.4 * relevance + 0.9 * freshness - 0.8 * risk
                candidates.append(
                    {
                        "label": label,
                        "relevance": relevance,
                        "freshness": freshness,
                        "risk": risk,
                    }
                )
                utilities[label] = utility

            distribution = _softmax_dict(
                utilities,
                temperature=0.45,
            )
            target = max(distribution, key=distribution.get)
            base_id = f"card-{option_count:03d}-{case_index:03d}"

            structured_state = {
                "candidates": candidates,
                "objective": (
                    "maximize 1.4*relevance + 0.9*freshness - 0.8*risk"
                ),
            }
            raw_state = " ".join(
                (
                    f"{item['label']} has relevance "
                    f"{item['relevance']:.6f}, freshness "
                    f"{item['freshness']:.6f}, and risk "
                    f"{item['risk']:.6f}."
                )
                for item in candidates
            )
            targets = {"source_choice": target}
            teacher = {"source_choice": distribution}

            pairs.append(
                CardinalityPair(
                    option_count=option_count,
                    raw=DecisionCase(
                        case_id=f"{base_id}::raw",
                        state=raw_state,
                        questions=(question,),
                        targets=targets,
                        target_distributions=teacher,
                        split="cardinality",
                        domain="system1_cardinality_raw_synthetic",
                    ),
                    structured=DecisionCase(
                        case_id=f"{base_id}::structured",
                        state=structured_state,
                        questions=(question,),
                        targets=targets,
                        target_distributions=teacher,
                        split="cardinality",
                        domain="system1_cardinality_structured_synthetic",
                    ),
                )
            )
    return tuple(pairs)


def _sample_scenario(
    rng: random.Random,
    scenario_id: str,
    split: str,
) -> SystemOneScenario:
    return SystemOneScenario(
        scenario_id=scenario_id,
        split=split,
        complexity=rng.random(),
        ambiguity=rng.random(),
        stakes=rng.random(),
        confidence=rng.random(),
        source_freshness=rng.random(),
        contradiction=rng.random(),
        human_authorized=rng.random() < 0.55,
        authority_available=rng.random() < 0.80,
        external_search_allowed=rng.random() < 0.75,
        commitment_pressure=rng.random(),
        importance=rng.random(),
    )


def _sample_ood_scenario(
    rng: random.Random,
    scenario_id: str,
    family: str,
) -> SystemOneScenario:
    base = _sample_scenario(rng, scenario_id, family)

    if family == "ood_compositional":
        return SystemOneScenario(
            **{
                **base.__dict__,
                "complexity": rng.uniform(0.82, 1.0),
                "stakes": rng.uniform(0.82, 1.0),
                "source_freshness": rng.uniform(0.0, 0.30),
            }
        )
    if family == "ood_source_failure":
        return SystemOneScenario(
            **{
                **base.__dict__,
                "source_freshness": rng.uniform(0.0, 0.25),
                "authority_available": False,
                "external_search_allowed": False,
            }
        )
    if family == "ood_confidence_shift":
        return SystemOneScenario(
            **{
                **base.__dict__,
                "confidence": rng.uniform(0.82, 1.0),
                "contradiction": rng.uniform(0.82, 1.0),
            }
        )
    raise ValueError(f"unknown OOD family: {family}")


def _binary_distribution(
    value: float,
    *,
    threshold: float,
    scale: float = 0.12,
) -> dict[str, float]:
    x = (value - threshold) / scale
    p_true = 1.0 / (1.0 + math.exp(-x))
    return {
        "false": 1.0 - p_true,
        "true": p_true,
    }


def _ordinal_distribution(
    value: float,
    *,
    centers: Sequence[float],
    labels: Sequence[str],
    scale: float,
) -> dict[str, float]:
    if len(centers) != len(labels):
        raise ValueError("centers/labels differ")
    logits = {
        label: -((value - center) ** 2) / (2 * scale * scale)
        for label, center in zip(labels, centers)
    }
    return _softmax_dict(logits, temperature=1.0)


def _softmax_dict(
    logits: dict[str, float],
    *,
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = {
        label: value / temperature
        for label, value in logits.items()
    }
    maximum = max(scaled.values())
    exponentials = {
        label: math.exp(value - maximum)
        for label, value in scaled.items()
    }
    total = sum(exponentials.values())
    return {
        label: value / total
        for label, value in exponentials.items()
    }
