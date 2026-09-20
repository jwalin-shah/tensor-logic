"""Recursive provenanced Datalog-style closure using ProofSemiring tags."""

from __future__ import annotations

from dataclasses import dataclass

from .provenance_semiring import (
    ProofSemiring,
    compose_relation_with_provenance,
)


Relation = dict[tuple[str, str], ProofSemiring]


@dataclass(frozen=True)
class ProvenanceClosureResult:
    relation: Relation
    iterations: int
    proof_count: int


def provenanced_transitive_closure(
    edges: Relation,
    *,
    max_iterations: int | None = None,
) -> ProvenanceClosureResult:
    relation = dict(edges)
    nodes = {node for pair in relation for node in pair}
    theoretical_bound = max(1, len(nodes) * len(nodes))
    limit = max_iterations or theoretical_bound

    for iteration in range(1, limit + 1):
        composed = compose_relation_with_provenance(
            relation,
            relation,
        )
        changed = False
        for key, provenance in composed.items():
            previous = relation.get(key, ProofSemiring.zero())
            merged = previous.plus(provenance).minimal()
            if merged != previous:
                relation[key] = merged
                changed = True
        if not changed:
            return ProvenanceClosureResult(
                relation=relation,
                iterations=iteration,
                proof_count=sum(
                    len(value.proofs)
                    for value in relation.values()
                ),
            )

    raise RuntimeError(
        "provenanced closure did not converge within finite bound"
    )
