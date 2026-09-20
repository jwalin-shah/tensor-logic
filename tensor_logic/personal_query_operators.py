"""Deterministic operators for canonical Personal Physics queries.

These operators implement the trusted math behind the canonical query catalog.
They do not parse natural language, call external services, or invoke an LLM.
External adapters must normalize authoritative source data into these input
shapes before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .selective_executor import QueryOperatorRegistry


@dataclass(frozen=True)
class CalendarTransition:
    person: str
    from_event: str
    to_event: str
    origin: str
    destination: str
    from_end: float
    to_start: float
    gap_minutes: float


@dataclass(frozen=True)
class TravelFeasibility:
    person: str
    from_event: str
    to_event: str
    origin: str
    destination: str
    gap_minutes: float
    route_minutes: float
    slack_minutes: float
    feasible: bool


@dataclass(frozen=True)
class RelationshipAttention:
    person: str
    score: float
    importance: float
    staleness_days: float
    staleness_component: float
    open_commitments: float
    open_commitment_component: float
    weights: tuple[tuple[str, float], ...]


def calendar_transition_adapter(
    inputs: tuple[Any, ...],
    parameters: Mapping[str, Any],
) -> tuple[CalendarTransition, ...]:
    if len(inputs) != 4:
        raise ValueError("calendar_transition_adapter expects 4 inputs")
    attends, located_at, starts_at, ends_at = inputs

    if not isinstance(attends, Mapping):
        raise TypeError("attends must map person -> event sequence")
    for mapping, name in (
        (located_at, "located_at"),
        (starts_at, "starts_at"),
        (ends_at, "ends_at"),
    ):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a mapping")

    rows: list[CalendarTransition] = []
    for person, events_raw in attends.items():
        if isinstance(events_raw, str):
            raise TypeError("attends event collection cannot be a string")
        events = tuple(str(event) for event in events_raw)
        ordered = sorted(
            events,
            key=lambda event: float(starts_at[event]),
        )
        for from_event, to_event in zip(ordered, ordered[1:]):
            origin = str(located_at[from_event])
            destination = str(located_at[to_event])
            from_end = float(ends_at[from_event])
            to_start = float(starts_at[to_event])
            rows.append(
                CalendarTransition(
                    person=str(person),
                    from_event=from_event,
                    to_event=to_event,
                    origin=origin,
                    destination=destination,
                    from_end=from_end,
                    to_start=to_start,
                    gap_minutes=to_start - from_end,
                )
            )
    return tuple(rows)


def travel_feasibility(
    inputs: tuple[Any, ...],
    parameters: Mapping[str, Any],
) -> tuple[TravelFeasibility, ...]:
    if len(inputs) != 2:
        raise ValueError("travel_feasibility expects 2 inputs")
    transitions, travel_minutes = inputs
    if not isinstance(travel_minutes, Mapping):
        raise TypeError(
            "travel_minutes must map (origin,destination) -> minutes"
        )

    buffer_minutes = float(parameters.get("buffer_minutes", 0.0))
    if buffer_minutes < 0:
        raise ValueError("buffer_minutes cannot be negative")

    out: list[TravelFeasibility] = []
    for row in transitions:
        if not isinstance(row, CalendarTransition):
            raise TypeError(
                "travel_feasibility expects CalendarTransition rows"
            )
        key = (row.origin, row.destination)
        if key not in travel_minutes:
            raise ValueError(
                f"missing route minutes for {row.origin!r}->{row.destination!r}"
            )
        route = float(travel_minutes[key])
        if route < 0:
            raise ValueError("route minutes cannot be negative")
        slack = row.gap_minutes - route - buffer_minutes
        out.append(
            TravelFeasibility(
                person=row.person,
                from_event=row.from_event,
                to_event=row.to_event,
                origin=row.origin,
                destination=row.destination,
                gap_minutes=row.gap_minutes,
                route_minutes=route,
                slack_minutes=slack,
                feasible=slack >= 0.0,
            )
        )
    return tuple(out)


def open_commitment_projection(
    inputs: tuple[Any, ...],
    parameters: Mapping[str, Any],
) -> dict[str, float]:
    if len(inputs) != 2:
        raise ValueError("open_commitment_projection expects 2 inputs")
    commitment_target_person, commitment_completed = inputs
    if not isinstance(commitment_target_person, Mapping):
        raise TypeError(
            "commitment_target_person must map commitment -> person"
        )
    if not isinstance(commitment_completed, Mapping):
        raise TypeError(
            "commitment_completed must map commitment -> bool/weight"
        )

    out: dict[str, float] = {}
    for commitment, person in commitment_target_person.items():
        completed = bool(commitment_completed.get(commitment, False))
        if completed:
            continue
        person_key = str(person)
        out[person_key] = out.get(person_key, 0.0) + 1.0
    return out


def relationship_attention_score(
    inputs: tuple[Any, ...],
    parameters: Mapping[str, Any],
) -> tuple[RelationshipAttention, ...]:
    if len(inputs) != 4:
        raise ValueError("relationship_attention_score expects 4 inputs")
    (
        relationship_importance,
        last_interaction_age_days,
        open_commitment_person,
        policy_weight,
    ) = inputs

    for mapping, name in (
        (relationship_importance, "relationship_importance"),
        (last_interaction_age_days, "last_interaction_age_days"),
        (open_commitment_person, "open_commitment_person"),
        (policy_weight, "policy_weight"),
    ):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a mapping")

    importance_weight = float(policy_weight.get("importance", 0.45))
    staleness_weight = float(policy_weight.get("staleness", 0.25))
    commitment_weight = float(
        policy_weight.get("open_commitment", 0.30)
    )
    staleness_scale_days = float(
        policy_weight.get("staleness_scale_days", 30.0)
    )
    commitment_scale = float(
        policy_weight.get("commitment_scale", 2.0)
    )

    if staleness_scale_days <= 0 or commitment_scale <= 0:
        raise ValueError("normalization scales must be positive")
    if min(
        importance_weight,
        staleness_weight,
        commitment_weight,
    ) < 0:
        raise ValueError("policy weights cannot be negative")

    people = (
        set(str(person) for person in relationship_importance)
        | set(str(person) for person in last_interaction_age_days)
        | set(str(person) for person in open_commitment_person)
    )
    rows: list[RelationshipAttention] = []
    weights = (
        ("importance", importance_weight),
        ("staleness", staleness_weight),
        ("open_commitment", commitment_weight),
    )

    for person in sorted(people):
        importance = float(relationship_importance.get(person, 0.0))
        staleness_days = float(
            last_interaction_age_days.get(person, 0.0)
        )
        open_commitments = float(
            open_commitment_person.get(person, 0.0)
        )
        if importance < 0 or staleness_days < 0 or open_commitments < 0:
            raise ValueError("relationship inputs cannot be negative")

        importance_component = min(importance, 1.0)
        staleness_component = min(
            staleness_days / staleness_scale_days,
            1.0,
        )
        open_commitment_component = min(
            open_commitments / commitment_scale,
            1.0,
        )
        score = (
            importance_weight * importance_component
            + staleness_weight * staleness_component
            + commitment_weight * open_commitment_component
        )
        rows.append(
            RelationshipAttention(
                person=person,
                score=score,
                importance=importance,
                staleness_days=staleness_days,
                staleness_component=staleness_component,
                open_commitments=open_commitments,
                open_commitment_component=open_commitment_component,
                weights=weights,
            )
        )

    rows.sort(key=lambda row: (-row.score, row.person))
    return tuple(rows)


def binary_compose_relation(
    inputs: tuple[Any, ...],
    parameters: Mapping[str, Any],
) -> frozenset[tuple[str, str]]:
    """Compose two pair relations: A(x,y) and B(y,z) -> C(x,z)."""
    if len(inputs) != 2:
        raise ValueError("binary_compose expects 2 inputs")
    left, right = inputs

    left_pairs = _pair_relation(left, "left")
    right_pairs = _pair_relation(right, "right")

    by_middle: dict[str, set[str]] = {}
    for middle, target in right_pairs:
        by_middle.setdefault(middle, set()).add(target)

    return frozenset(
        (source, target)
        for source, middle in left_pairs
        for target in by_middle.get(middle, ())
    )


def build_personal_operator_registry() -> QueryOperatorRegistry:
    registry = QueryOperatorRegistry()
    registry.register(
        "calendar_transition_adapter",
        "1",
        calendar_transition_adapter,
    )
    registry.register(
        "travel_feasibility",
        "1",
        travel_feasibility,
    )
    registry.register(
        "open_commitment_projection",
        "1",
        open_commitment_projection,
    )
    registry.register(
        "relationship_attention_score",
        "1",
        relationship_attention_score,
    )
    registry.register(
        "binary_compose",
        "1",
        binary_compose_relation,
    )
    return registry


def _pair_relation(value: Any, name: str) -> frozenset[tuple[str, str]]:
    try:
        rows = frozenset(
            (str(source), str(target))
            for source, target in value
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} relation must be iterable pairs"
        ) from exc
    return rows
