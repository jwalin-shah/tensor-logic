"""Adaptable attention policies expressed as real-valued Tensor Logic.

This module is intentionally a ranking layer, not a truth layer. It consumes
normalized signals whose provenance/admission is managed elsewhere and produces
candidate attention scores with explicit per-feature contributions.

Changing a policy changes ranking, not historical world facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

from .program import Program


@dataclass(frozen=True)
class AttentionSignals:
    """Normalized [0,1] signals for one person."""

    importance: float = 0.0
    staleness: float = 0.0
    unresolved_followup: float = 0.0
    shared_project: float = 0.0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "importance",
            "staleness",
            "unresolved_followup",
            "shared_project",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0,1], got {value}"
                )


@dataclass(frozen=True)
class AttentionPolicy:
    version: str
    importance_weight: float
    staleness_weight: float
    unresolved_followup_weight: float
    shared_project_weight: float
    scope: str = "relationship_attention"

    def __post_init__(self) -> None:
        weights = self.weights
        if any(value < 0 for value in weights.values()):
            raise ValueError("attention weights must be non-negative")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"attention weights must sum to 1.0, got {total}"
            )

    @property
    def weights(self) -> dict[str, float]:
        return {
            "importance": self.importance_weight,
            "staleness": self.staleness_weight,
            "unresolved_followup": self.unresolved_followup_weight,
            "shared_project": self.shared_project_weight,
        }

    @property
    def digest(self) -> str:
        return _digest(
            {
                "version": self.version,
                "scope": self.scope,
                "weights": self.weights,
            }
        )


@dataclass(frozen=True)
class AttentionCandidate:
    person_id: str
    score: float
    contributions: dict[str, float]
    evidence_refs: tuple[str, ...]
    policy_version: str
    policy_digest: str
    candidate_only: bool = True


def score_attention_candidates(
    signals: Mapping[str, AttentionSignals],
    policy: AttentionPolicy,
) -> list[AttentionCandidate]:
    """Compile a transparent weighted attention policy into Tensor Logic."""
    if not signals:
        return []

    people = sorted(signals)
    program = Program()
    program.domain("person", people)

    relation_names = (
        "importance_signal",
        "staleness_signal",
        "unresolved_followup_signal",
        "shared_project_signal",
        "attention_score",
    )
    for relation_name in relation_names:
        program.relation(relation_name, "person")

    for person_id in people:
        item = signals[person_id]
        program.fact(
            "importance_signal",
            person_id,
            value=item.importance,
        )
        program.fact(
            "staleness_signal",
            person_id,
            value=item.staleness,
        )
        program.fact(
            "unresolved_followup_signal",
            person_id,
            value=item.unresolved_followup,
        )
        program.fact(
            "shared_project_signal",
            person_id,
            value=item.shared_project,
        )

    w = policy.weights
    program.rule(
        "attention_score(P) := "
        f"{w['importance']} * importance_signal(P) + "
        f"{w['staleness']} * staleness_signal(P) + "
        f"{w['unresolved_followup']} * "
        "unresolved_followup_signal(P) + "
        f"{w['shared_project']} * shared_project_signal(P)"
    )

    results: list[AttentionCandidate] = []
    for person_id in people:
        item = signals[person_id]
        contributions = {
            "importance": (
                policy.importance_weight * item.importance
            ),
            "staleness": (
                policy.staleness_weight * item.staleness
            ),
            "unresolved_followup": (
                policy.unresolved_followup_weight
                * item.unresolved_followup
            ),
            "shared_project": (
                policy.shared_project_weight
                * item.shared_project
            ),
        }
        score = program.query(
            "attention_score",
            person_id,
            semiring="real",
        )
        results.append(
            AttentionCandidate(
                person_id=person_id,
                score=score,
                contributions=contributions,
                evidence_refs=item.evidence_refs,
                policy_version=policy.version,
                policy_digest=policy.digest,
            )
        )

    return sorted(
        results,
        key=lambda item: (-item.score, item.person_id),
    )


DEFAULT_ATTENTION_POLICY = AttentionPolicy(
    version="attention-v1",
    importance_weight=0.25,
    staleness_weight=0.15,
    unresolved_followup_weight=0.45,
    shared_project_weight=0.15,
)


def _digest(payload) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
