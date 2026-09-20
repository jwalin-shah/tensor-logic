"""exp103: algebraic provenance sanity benchmark.

Compare an exact proof-semiring representation with a plain external witness
list on the same two-hop relational composition. The goal is to expose proof
growth and runtime, not to claim equivalence to full Scallop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tensor_logic.provenance_semiring import (
    compose_relation_with_provenance,
    facts_to_relation,
)


def run(width: int = 2000, fanout: int = 3) -> dict:
    left_rows = []
    right_rows = []
    for i in range(width):
        person = f"p{i % 200}"
        event = f"e{i}"
        left_rows.append((person, event, f"attends:{person}:{event}"))
        for j in range(fanout):
            topic = f"t{(i + j) % 100}"
            right_rows.append((event, topic, f"topic:{event}:{topic}"))

    start = time.perf_counter()
    left = facts_to_relation(left_rows)
    right = facts_to_relation(right_rows)
    algebraic = compose_relation_with_provenance(left, right)
    semiring_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    witness: dict[tuple[str, str], list[tuple[str, str]]] = {}
    right_by_event: dict[str, list[tuple[str, str]]] = {}
    for event, topic, fact_id in right_rows:
        right_by_event.setdefault(event, []).append((topic, fact_id))
    for person, event, left_fact in left_rows:
        for topic, right_fact in right_by_event.get(event, ()):
            witness.setdefault((person, topic), []).append(
                (left_fact, right_fact)
            )
    witness_ms = (time.perf_counter() - start) * 1000.0

    semiring_proofs = sum(
        len(value.proofs)
        for value in algebraic.values()
    )
    witness_proofs = sum(len(value) for value in witness.values())

    return {
        "experiment": "exp103_provenance",
        "warning": (
            "This compares a tiny exact proof-set algebra with a Python witness "
            "map; it is not a performance comparison against Scallop."
        ),
        "width": width,
        "fanout": fanout,
        "left_facts": len(left_rows),
        "right_facts": len(right_rows),
        "derived_cells": len(algebraic),
        "semiring_proofs": semiring_proofs,
        "witness_proofs": witness_proofs,
        "semiring_ms": semiring_ms,
        "witness_ms": witness_ms,
        "proof_counts_match": semiring_proofs == witness_proofs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp103.json")
    args = parser.parse_args()
    result = run()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
