"""Minimal provenance semiring for experiment #94.

The carrier is a set of proof monomials. A monomial is a frozenset of primitive
fact IDs that jointly support a derivation.

zero = no proofs
one  = one empty proof
plus = alternative derivations (set union)
times = conjunction/composition (cartesian union of monomials)

This is intentionally small and exact. It does not attempt Scallop's complete
family of differentiable provenance structures; it gives us a local algebraic
baseline to compare with external witness DAGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Monomial = frozenset[str]


@dataclass(frozen=True)
class ProofSemiring:
    proofs: frozenset[Monomial]

    @classmethod
    def zero(cls) -> "ProofSemiring":
        return cls(frozenset())

    @classmethod
    def one(cls) -> "ProofSemiring":
        return cls(frozenset({frozenset()}))

    @classmethod
    def fact(cls, fact_id: str) -> "ProofSemiring":
        if not fact_id:
            raise ValueError("fact_id cannot be empty")
        return cls(frozenset({frozenset({fact_id})}))

    def plus(self, other: "ProofSemiring") -> "ProofSemiring":
        return ProofSemiring(self.proofs | other.proofs)

    def times(self, other: "ProofSemiring") -> "ProofSemiring":
        if not self.proofs or not other.proofs:
            return ProofSemiring.zero()
        return ProofSemiring(
            frozenset(
                left | right
                for left in self.proofs
                for right in other.proofs
            )
        )

    def minimal(self) -> "ProofSemiring":
        """Drop proofs that are strict supersets of another proof."""
        kept = set(self.proofs)
        for proof in self.proofs:
            if any(
                other < proof
                for other in self.proofs
            ):
                kept.discard(proof)
        return ProofSemiring(frozenset(kept))

    def topk(
        self,
        k: int,
        *,
        fact_costs: dict[str, float] | None = None,
    ) -> tuple[Monomial, ...]:
        if k <= 0:
            raise ValueError("k must be positive")
        fact_costs = fact_costs or {}

        def cost(proof: Monomial) -> tuple[float, int, tuple[str, ...]]:
            return (
                sum(float(fact_costs.get(fact, 1.0)) for fact in proof),
                len(proof),
                tuple(sorted(proof)),
            )

        return tuple(sorted(self.proofs, key=cost)[:k])


def compose_relation_with_provenance(
    left: dict[tuple[str, str], ProofSemiring],
    right: dict[tuple[str, str], ProofSemiring],
) -> dict[tuple[str, str], ProofSemiring]:
    """Relational join/projection where provenance follows semiring algebra."""
    by_middle: dict[str, list[tuple[str, ProofSemiring]]] = {}
    for (middle, target), provenance in right.items():
        by_middle.setdefault(middle, []).append((target, provenance))

    out: dict[tuple[str, str], ProofSemiring] = {}
    for (source, middle), left_provenance in left.items():
        for target, right_provenance in by_middle.get(middle, ()):
            key = (source, target)
            contribution = left_provenance.times(right_provenance)
            out[key] = out.get(key, ProofSemiring.zero()).plus(
                contribution
            )
    return {
        key: value.minimal()
        for key, value in out.items()
    }


def facts_to_relation(
    rows: Iterable[tuple[str, str, str]],
) -> dict[tuple[str, str], ProofSemiring]:
    """Convert (source,target,fact_id) triples into a provenanced relation."""
    out: dict[tuple[str, str], ProofSemiring] = {}
    for source, target, fact_id in rows:
        key = (source, target)
        tag = ProofSemiring.fact(fact_id)
        out[key] = out.get(key, ProofSemiring.zero()).plus(tag)
    return out
