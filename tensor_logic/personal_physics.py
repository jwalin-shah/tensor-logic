"""Personal Physics v0: typed, replayable reasoning over admitted world facts.

This layer intentionally separates:
- observations/evidence (outside Tensor Logic),
- admitted primitive facts (inputs to Tensor Logic),
- deterministic rules (Tensor Logic derivations), and
- candidate/rejected/retracted facts (never used as premises).

Numeric work such as time arithmetic, routing duration, or currency conversion
is performed by deterministic adapters. Adapter results become provenance-rich
primitive facts that Tensor Logic may then compose.

Negation-as-failure is deliberately not used here: UNKNOWN is not FALSE.
Negative premises must be represented by explicit admitted relations such as
not_completed_followup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

from .program import Atom, Rule
from .provenance import evaluate_with_provenance, proof_score


FACT_STATUSES = {
    "candidate",
    "admitted",
    "rejected",
    "contradicted",
    "retracted",
}


@dataclass(frozen=True)
class RelationSchema:
    name: str
    subject_type: str
    object_type: str


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    relation: str
    subject: str
    object: str
    status: str
    evidence_refs: tuple[str, ...] = ()
    source_kind: str = "unknown"
    source_ref: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    confidence: float | None = None
    admission_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    head: Atom
    body: tuple[Atom, ...]
    scope: str = "personal_physics_v0"
    version: str = "1"

    def __post_init__(self) -> None:
        if any(atom.negated for atom in self.body):
            raise ValueError(
                "Personal Physics v0 forbids negation-as-failure; "
                "use explicit negative relations"
            )

    @property
    def ast(self) -> Rule:
        return Rule(self.head, self.body)

    @property
    def digest(self) -> str:
        payload = {
            "rule_id": self.rule_id,
            "version": self.version,
            "scope": self.scope,
            "head": _atom_payload(self.head),
            "body": [_atom_payload(atom) for atom in self.body],
        }
        return _digest(payload)


@dataclass(frozen=True)
class DerivationResult:
    query: tuple[str, str, str]
    entailed: bool
    proof: dict[str, Any] | None
    world_digest: str
    derivation_digest: str | None


class PersonalPhysicsWorld:
    """Typed evidence-backed fact/rule store with deterministic proof replay."""

    def __init__(self) -> None:
        self.entities: dict[str, str] = {}
        self.relations: dict[str, RelationSchema] = {}
        self.facts: dict[str, FactRecord] = {}
        self.rules: dict[str, RuleSpec] = {}

    def register_entity(self, entity_id: str, entity_type: str) -> None:
        existing = self.entities.get(entity_id)
        if existing is not None and existing != entity_type:
            raise ValueError(
                f"entity {entity_id!r} already registered as {existing!r}"
            )
        self.entities[entity_id] = entity_type

    def register_relation(
        self,
        name: str,
        subject_type: str,
        object_type: str,
    ) -> None:
        schema = RelationSchema(name, subject_type, object_type)
        existing = self.relations.get(name)
        if existing is not None and existing != schema:
            raise ValueError(
                f"relation {name!r} already has incompatible schema"
            )
        self.relations[name] = schema

    def add_fact(self, fact: FactRecord) -> None:
        if fact.status not in FACT_STATUSES:
            raise ValueError(f"unknown fact status: {fact.status}")
        if fact.fact_id in self.facts:
            raise ValueError(f"duplicate fact_id: {fact.fact_id}")
        self._validate_ground_atom(
            fact.relation,
            fact.subject,
            fact.object,
        )
        if fact.status == "admitted" and not fact.admission_ref:
            raise ValueError(
                "admitted facts require an admission_ref"
            )
        self.facts[fact.fact_id] = fact

    def set_fact_status(
        self,
        fact_id: str,
        status: str,
        *,
        admission_ref: str | None = None,
    ) -> None:
        if status not in FACT_STATUSES:
            raise ValueError(f"unknown fact status: {status}")
        current = self.facts[fact_id]
        next_admission = (
            admission_ref
            if admission_ref is not None
            else current.admission_ref
        )
        if status == "admitted" and not next_admission:
            raise ValueError(
                "admitted facts require an admission_ref"
            )
        self.facts[fact_id] = FactRecord(
            fact_id=current.fact_id,
            relation=current.relation,
            subject=current.subject,
            object=current.object,
            status=status,
            evidence_refs=current.evidence_refs,
            source_kind=current.source_kind,
            source_ref=current.source_ref,
            valid_from=current.valid_from,
            valid_until=current.valid_until,
            confidence=current.confidence,
            admission_ref=next_admission,
            metadata=dict(current.metadata),
        )

    def add_rule(self, spec: RuleSpec) -> None:
        if spec.rule_id in self.rules:
            raise ValueError(f"duplicate rule_id: {spec.rule_id}")
        self._validate_rule_types(spec)
        self.rules[spec.rule_id] = spec

    def admitted_graph(self) -> dict[str, list[tuple[str, str]]]:
        graph: dict[str, list[tuple[str, str]]] = {}
        for fact in sorted(
            self.facts.values(),
            key=lambda item: item.fact_id,
        ):
            if fact.status != "admitted":
                continue
            graph.setdefault(fact.relation, []).append(
                (fact.subject, fact.object)
            )
        return graph

    @property
    def world_digest(self) -> str:
        payload = {
            "entities": sorted(self.entities.items()),
            "relations": [
                (
                    item.name,
                    item.subject_type,
                    item.object_type,
                )
                for item in sorted(
                    self.relations.values(),
                    key=lambda schema: schema.name,
                )
            ],
            "admitted_facts": [
                _fact_payload(fact)
                for fact in sorted(
                    self.facts.values(),
                    key=lambda item: item.fact_id,
                )
                if fact.status == "admitted"
            ],
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "digest": rule.digest,
                }
                for rule in sorted(
                    self.rules.values(),
                    key=lambda item: item.rule_id,
                )
            ],
        }
        return _digest(payload)

    def derive(
        self,
        relation: str,
        subject: str,
        object: str,
    ) -> DerivationResult:
        self._validate_ground_atom(relation, subject, object)
        graph = self.admitted_graph()
        rule_asts = [
            spec.ast
            for spec in sorted(
                self.rules.values(),
                key=lambda item: item.rule_id,
            )
        ]
        proofs = evaluate_with_provenance(
            graph,
            rule_asts,
            relation,
            subject,
            object,
            sort=True,
        )
        if not proofs:
            return DerivationResult(
                query=(relation, subject, object),
                entailed=False,
                proof=None,
                world_digest=self.world_digest,
                derivation_digest=None,
            )

        proof = self._externalize_proof(proofs[0])
        payload = {
            "query": [relation, subject, object],
            "world_digest": self.world_digest,
            "proof": proof,
        }
        return DerivationResult(
            query=(relation, subject, object),
            entailed=True,
            proof=proof,
            world_digest=self.world_digest,
            derivation_digest=_digest(payload),
        )

    def _externalize_proof(
        self,
        proof: dict[str, Any],
    ) -> dict[str, Any]:
        if "primitive" in proof:
            relation, subject, object = proof["primitive"]
            matching = sorted(
                fact.fact_id
                for fact in self.facts.values()
                if (
                    fact.status == "admitted"
                    and fact.relation == relation
                    and fact.subject == subject
                    and fact.object == object
                )
            )
            return {
                "kind": "primitive",
                "fact_ids": matching,
                "relation": relation,
                "subject": subject,
                "object": object,
            }

        rule_ast = proof["rule"]
        specs = sorted(
            (
                spec
                for spec in self.rules.values()
                if spec.ast == rule_ast
            ),
            key=lambda item: item.rule_id,
        )
        if not specs:
            raise ValueError("proof references an unknown rule")
        spec = specs[0]
        relation, subject, object = proof["head"]
        return {
            "kind": "rule",
            "rule_id": spec.rule_id,
            "rule_version": spec.version,
            "rule_digest": spec.digest,
            "scope": spec.scope,
            "conclusion": [relation, subject, object],
            "premises": [
                self._externalize_proof(child)
                for child in proof["body"]
            ],
            "proof_score": list(proof_score(proof)),
        }

    def _validate_ground_atom(
        self,
        relation: str,
        subject: str,
        object: str,
    ) -> None:
        if relation not in self.relations:
            raise ValueError(f"unknown relation: {relation}")
        if subject not in self.entities:
            raise ValueError(f"unknown subject entity: {subject}")
        if object not in self.entities:
            raise ValueError(f"unknown object entity: {object}")
        schema = self.relations[relation]
        if self.entities[subject] != schema.subject_type:
            raise ValueError(
                f"{relation} subject expects {schema.subject_type}, "
                f"got {self.entities[subject]}"
            )
        if self.entities[object] != schema.object_type:
            raise ValueError(
                f"{relation} object expects {schema.object_type}, "
                f"got {self.entities[object]}"
            )

    def _validate_rule_types(self, spec: RuleSpec) -> None:
        atoms = (spec.head, *spec.body)
        variable_types: dict[str, str] = {}
        for atom in atoms:
            if atom.relation not in self.relations:
                raise ValueError(
                    f"unknown relation in rule {spec.rule_id}: "
                    f"{atom.relation}"
                )
            schema = self.relations[atom.relation]
            if len(atom.args) != 2:
                raise ValueError(
                    "Personal Physics v0 supports binary relations only"
                )
            for variable, expected_type in zip(
                atom.args,
                (schema.subject_type, schema.object_type),
            ):
                current = variable_types.get(variable)
                if current is not None and current != expected_type:
                    raise ValueError(
                        f"variable {variable} has incompatible types "
                        f"{current} and {expected_type}"
                    )
                variable_types[variable] = expected_type


def build_personal_physics_v0() -> PersonalPhysicsWorld:
    """Return the initial typed schema + reusable deterministic rule families."""
    world = PersonalPhysicsWorld()

    relations = {
        "attends": ("Person", "Event"),
        "overlaps": ("Event", "Event"),
        "schedule_conflict": ("Event", "Event"),
        "insufficient_travel_gap": ("Event", "Event"),
        "infeasible_transition": ("Event", "Event"),
        "requested_followup": ("Person", "Person"),
        "not_completed_followup": ("Person", "Person"),
        "unresolved_followup": ("Person", "Person"),
        "works_on": ("Person", "Project"),
        "supports_goal": ("Project", "Goal"),
        "active_goal_work": ("Person", "Goal"),
        "invoked": ("ToolCall", "Tool"),
        "produced_evidence": ("ToolCall", "Evidence"),
        "supports_claim": ("Evidence", "Claim"),
        "tool_backed_claim": ("Tool", "Claim"),
    }
    for name, (subject_type, object_type) in relations.items():
        world.register_relation(name, subject_type, object_type)

    world.add_rule(
        RuleSpec(
            rule_id="R-calendar-conflict",
            head=Atom("schedule_conflict", ("E1", "E2")),
            body=(
                Atom("attends", ("P", "E1")),
                Atom("attends", ("P", "E2")),
                Atom("overlaps", ("E1", "E2")),
            ),
        )
    )
    world.add_rule(
        RuleSpec(
            rule_id="R-travel-infeasible",
            head=Atom("infeasible_transition", ("E1", "E2")),
            body=(
                Atom("attends", ("P", "E1")),
                Atom("attends", ("P", "E2")),
                Atom(
                    "insufficient_travel_gap",
                    ("E1", "E2"),
                ),
            ),
        )
    )
    world.add_rule(
        RuleSpec(
            rule_id="R-unresolved-followup",
            head=Atom("unresolved_followup", ("P1", "P2")),
            body=(
                Atom("requested_followup", ("P1", "P2")),
                Atom(
                    "not_completed_followup",
                    ("P1", "P2"),
                ),
            ),
        )
    )
    world.add_rule(
        RuleSpec(
            rule_id="R-active-goal-work",
            head=Atom("active_goal_work", ("P", "G")),
            body=(
                Atom("works_on", ("P", "Project")),
                Atom("supports_goal", ("Project", "G")),
            ),
        )
    )
    world.add_rule(
        RuleSpec(
            rule_id="R-tool-backed-claim",
            head=Atom("tool_backed_claim", ("Tool", "Claim")),
            body=(
                Atom("invoked", ("Call", "Tool")),
                Atom("produced_evidence", ("Call", "Evidence")),
                Atom("supports_claim", ("Evidence", "Claim")),
            ),
        )
    )
    return world


def admit_interval_overlap(
    world: PersonalPhysicsWorld,
    *,
    fact_id: str,
    event_a: str,
    start_a_minute: int,
    end_a_minute: int,
    event_b: str,
    start_b_minute: int,
    end_b_minute: int,
    evidence_refs: Iterable[str],
    admission_ref: str,
) -> FactRecord | None:
    """Deterministically normalize scalar calendar times into an overlap fact."""
    overlaps = (
        start_a_minute < end_b_minute
        and start_b_minute < end_a_minute
    )
    if not overlaps:
        return None
    fact = FactRecord(
        fact_id=fact_id,
        relation="overlaps",
        subject=event_a,
        object=event_b,
        status="admitted",
        evidence_refs=tuple(evidence_refs),
        source_kind="deterministic_adapter",
        source_ref="calendar_interval_overlap_v1",
        admission_ref=admission_ref,
        metadata={
            "inputs": {
                "start_a_minute": start_a_minute,
                "end_a_minute": end_a_minute,
                "start_b_minute": start_b_minute,
                "end_b_minute": end_b_minute,
            },
            "calculation": (
                "start_a < end_b and start_b < end_a"
            ),
        },
    )
    world.add_fact(fact)
    return fact


def admit_travel_gap(
    world: PersonalPhysicsWorld,
    *,
    fact_id: str,
    event_a: str,
    event_a_end_minute: int,
    event_b: str,
    event_b_start_minute: int,
    travel_minutes: int,
    evidence_refs: Iterable[str],
    admission_ref: str,
    route_call_ref: str,
) -> FactRecord | None:
    """Normalize routing output into an admitted insufficient-gap primitive."""
    available_gap = (
        event_b_start_minute - event_a_end_minute
    )
    if travel_minutes <= available_gap:
        return None
    fact = FactRecord(
        fact_id=fact_id,
        relation="insufficient_travel_gap",
        subject=event_a,
        object=event_b,
        status="admitted",
        evidence_refs=tuple(evidence_refs),
        source_kind="deterministic_adapter",
        source_ref=route_call_ref,
        admission_ref=admission_ref,
        metadata={
            "inputs": {
                "event_a_end_minute": event_a_end_minute,
                "event_b_start_minute": event_b_start_minute,
                "travel_minutes": travel_minutes,
            },
            "available_gap_minutes": available_gap,
            "calculation": "travel_minutes > available_gap_minutes",
        },
    )
    world.add_fact(fact)
    return fact


def _fact_payload(fact: FactRecord) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "relation": fact.relation,
        "subject": fact.subject,
        "object": fact.object,
        "status": fact.status,
        "evidence_refs": list(fact.evidence_refs),
        "source_kind": fact.source_kind,
        "source_ref": fact.source_ref,
        "valid_from": fact.valid_from,
        "valid_until": fact.valid_until,
        "confidence": fact.confidence,
        "admission_ref": fact.admission_ref,
        "metadata": fact.metadata,
    }


def _atom_payload(atom: Atom) -> dict[str, Any]:
    return {
        "relation": atom.relation,
        "args": list(atom.args),
        "negated": atom.negated,
    }


def _digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
