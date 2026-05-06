"""Deterministic support/stability engine over primitive geometry facts.

This is the Tensor Logic substrate slice for the support/stability V1
experiment: perfect object tables in, primitive relations extracted from
geometry, fixpoint support closure out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


EPS = 1e-9
Fact = Tuple[str, ...]


@dataclass(frozen=True)
class SupportTolerance:
    contact: float = EPS
    horizontal: float = 0.0

    def __post_init__(self) -> None:
        if self.contact < 0:
            raise ValueError("contact tolerance must be non-negative")
        if self.horizontal < 0:
            raise ValueError("horizontal tolerance must be non-negative")


@dataclass(frozen=True)
class ProofTrace:
    fact: Fact
    rule: str
    body: Tuple["ProofTrace", ...] = ()

    def lines(self, indent: int = 0) -> list[str]:
        pad = "  " * indent
        head = f"{self.fact[0]}({', '.join(self.fact[1:])})"
        out = [f"{pad}{head}  [{self.rule}]"]
        for child in self.body:
            out.extend(child.lines(indent + 1))
        return out

    def format(self) -> str:
        return "\n".join(self.lines())


@dataclass(frozen=True)
class PrimitiveRelations:
    touching: frozenset[tuple[str, str]] = frozenset()
    above: frozenset[tuple[str, str]] = frozenset()
    horiz_overlap: frozenset[tuple[str, str]] = frozenset()
    on_ground: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StabilityResult:
    labels: dict[str, str]
    stable: frozenset[str]
    falls: frozenset[str]
    supports: frozenset[tuple[str, str]]
    primitives: PrimitiveRelations
    proofs: dict[tuple[str, str], ProofTrace] = field(default_factory=dict)

    def proof_for(self, relation: str, object_id: str) -> ProofTrace:
        return self.proofs[(relation, object_id)]


def _interval(obj: dict[str, object]) -> tuple[float, float]:
    x = float(obj["x"])
    return x, x + float(obj["w"])


def _as_tolerance(tolerance: SupportTolerance | None) -> SupportTolerance:
    return tolerance if tolerance is not None else SupportTolerance()


def _overlap_interval(
    a: dict[str, object],
    b: dict[str, object],
    tolerance: float = 0.0,
) -> tuple[float, float] | None:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        if tolerance <= 0.0 or lo - hi > tolerance:
            return None
        return lo, lo
    return lo, hi


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        mlo, mhi = merged[-1]
        if lo <= mhi:
            merged[-1] = (mlo, max(mhi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _covers_width(intervals: Iterable[tuple[float, float]], left: float, right: float, tolerance: float = EPS) -> bool:
    pos = left
    for lo, hi in _merge_intervals(intervals):
        if lo > pos + tolerance:
            return False
        pos = max(pos, hi)
        if pos >= right - tolerance:
            return True
    return pos >= right - tolerance


def _removed_id(intervention: Optional[dict[str, str]]) -> str | None:
    if not intervention or intervention.get("type") == "none":
        return None
    if intervention.get("type") != "remove":
        raise ValueError(f"unknown intervention type: {intervention.get('type')!r}")
    return intervention.get("object_id")


def _objects_by_id(objects: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for obj in objects:
        oid = str(obj["id"])
        if oid in by_id:
            raise ValueError(f"duplicate object id: {oid}")
        by_id[oid] = dict(obj)
    return by_id


def extract_primitives(
    objects: Sequence[dict[str, object]],
    intervention: Optional[dict[str, str]] = None,
    tolerance: SupportTolerance | None = None,
) -> PrimitiveRelations:
    """Extract primitive relations from perfect object geometry.

    Relation argument order follows the rule sketch in the plan:
    `touching(upper, lower)` and `above(upper, lower)`.
    """

    tol = _as_tolerance(tolerance)
    by_id = _objects_by_id(objects)
    removed = _removed_id(intervention)
    if removed is not None and removed not in by_id:
        raise ValueError(f"unknown remove target: {removed}")

    active = [obj for oid, obj in by_id.items() if oid != removed]
    touching: set[tuple[str, str]] = set()
    above: set[tuple[str, str]] = set()
    horiz_overlap: set[tuple[str, str]] = set()
    on_ground: set[str] = set()

    for obj in active:
        oid = str(obj["id"])
        if abs(float(obj["y"])) <= tol.contact:
            on_ground.add(oid)

    for upper in active:
        upper_id = str(upper["id"])
        for lower in active:
            lower_id = str(lower["id"])
            if upper_id == lower_id:
                continue
            if _overlap_interval(upper, lower, tol.horizontal) is not None:
                horiz_overlap.add((upper_id, lower_id))
            if float(upper["y"]) >= float(lower["y"]) + float(lower["h"]) - tol.contact:
                above.add((upper_id, lower_id))
            lower_top = float(lower["y"]) + float(lower["h"])
            if abs(float(upper["y"]) - lower_top) <= tol.contact and _overlap_interval(upper, lower, tol.horizontal) is not None:
                touching.add((upper_id, lower_id))

    return PrimitiveRelations(
        touching=frozenset(touching),
        above=frozenset(above),
        horiz_overlap=frozenset(horiz_overlap),
        on_ground=frozenset(on_ground),
        removed=frozenset({removed} if removed is not None else set()),
    )


def infer_stability(
    objects: Sequence[dict[str, object]],
    intervention: Optional[dict[str, str]] = None,
    tolerance: SupportTolerance | None = None,
) -> StabilityResult:
    tol = _as_tolerance(tolerance)
    by_id = _objects_by_id(objects)
    removed = _removed_id(intervention)
    if removed is not None and removed not in by_id:
        raise ValueError(f"unknown remove target: {removed}")

    primitives = extract_primitives(objects, intervention, tolerance=tol)
    active_ids = [oid for oid in by_id if oid not in primitives.removed]
    stable: set[str] = set()
    supports: set[tuple[str, str]] = set()
    proofs: dict[tuple[str, str], ProofTrace] = {}

    changed = True
    while changed:
        changed = False
        for oid in active_ids:
            if oid in stable:
                continue
            obj = by_id[oid]
            if oid in primitives.on_ground:
                stable.add(oid)
                proofs[("stable", oid)] = ProofTrace(
                    ("stable", oid),
                    "stable(X) :- on_ground(X), not removed(X)",
                    (ProofTrace(("on_ground", oid), "geometry primitive"),),
                )
                changed = True
                continue

            intervals = []
            support_children: list[ProofTrace] = []
            for sid in stable:
                supporter = by_id[sid]
                if (oid, sid) not in primitives.touching:
                    continue
                if (oid, sid) not in primitives.above:
                    continue
                if (oid, sid) not in primitives.horiz_overlap:
                    continue
                overlap = _overlap_interval(obj, supporter, tol.horizontal)
                if overlap is None:
                    continue
                intervals.append(overlap)
                supports.add((sid, oid))
                support_proof = ProofTrace(
                    ("supports", sid, oid),
                    "supports(Y, X) from primitive contact with stable(Y)",
                    (
                        ProofTrace(("touching", oid, sid), "geometry primitive"),
                        ProofTrace(("above", oid, sid), "geometry primitive"),
                        ProofTrace(("horiz_overlap", oid, sid), "geometry primitive"),
                        proofs[("stable", sid)],
                    ),
                )
                proofs[("supports", f"{sid}->{oid}")] = support_proof
                support_children.append(support_proof)

            left, right = _interval(obj)
            if _covers_width(intervals, left, right, max(EPS, tol.horizontal)):
                stable.add(oid)
                proofs[("stable", oid)] = ProofTrace(
                    ("stable", oid),
                    "stable(X) :- stable supporters cover full horizontal width",
                    tuple(support_children),
                )
                changed = True

    labels: Dict[str, str] = {}
    falls: set[str] = set()
    for oid in by_id:
        if oid in stable:
            labels[oid] = "stable"
        else:
            labels[oid] = "falls"
            falls.add(oid)
            reason = "falls(X) because X is removed" if oid in primitives.removed else "falls(X) :- not stable(X)"
            proofs[("falls", oid)] = ProofTrace(("falls", oid), reason)

    return StabilityResult(
        labels=labels,
        stable=frozenset(stable),
        falls=frozenset(falls),
        supports=frozenset(supports),
        primitives=primitives,
        proofs=proofs,
    )
