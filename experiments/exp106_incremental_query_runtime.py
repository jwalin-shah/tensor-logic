"""exp106: selective query planning and coordinate-level recomputation.

Synthetic mechanism benchmark:
1. compile one target from a registry containing unrelated views and report how
   much state is excluded from the plan;
2. mutate one sparse binary input coordinate and compare exact incremental
   maintenance against a full sparse matrix recomputation.

This is not yet an OCI performance claim; hardware-specific numbers belong in
the governed runtime manifest.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from tensor_logic.incremental_binary_view import (
    BinaryCoordinateDelta,
    IncrementalBinaryView,
)
from tensor_logic.query_runtime import (
    DerivedViewSpec,
    PrimitiveTensorSpec,
    QueryRegistry,
    SourceRequirement,
)
from tensor_logic.tensor_ops import sparse_binary_compose
from tensor_logic.world_tensor import SparseWorldTensor, TensorAxis


def _registry() -> QueryRegistry:
    r = QueryRegistry()
    primitives = {
        "attends": "calendar",
        "located_at": "calendar",
        "travel_minutes": "routes",
        "starts_at": "calendar",
        "ends_at": "calendar",
        "requested_followup": "messages",
        "commitment_completed": "tasks",
        "goal_priority": "human_policy",
        "message_sender": "messages",
        "document_project": "drive",
        "provider_readable": "provider_health",
    }
    for name, source in primitives.items():
        r.add_primitive(
            PrimitiveTensorSpec(
                name,
                (SourceRequirement(source),),
            )
        )

    r.add_view(
        DerivedViewSpec(
            "event_place",
            ("attends", "located_at"),
            "binary_compose",
            "1",
        )
    )
    r.add_view(
        DerivedViewSpec(
            "travel_pressure",
            ("event_place", "travel_minutes", "starts_at", "ends_at"),
            "travel_feasibility",
            "1",
        )
    )
    r.add_view(
        DerivedViewSpec(
            "unresolved_followup",
            ("requested_followup", "commitment_completed"),
            "followup_rule",
            "1",
        )
    )
    r.add_view(
        DerivedViewSpec(
            "attention_candidate",
            ("unresolved_followup", "goal_priority"),
            "weighted_score",
            "2",
        )
    )
    r.add_view(
        DerivedViewSpec(
            "project_correspondence",
            ("message_sender", "document_project"),
            "project_join",
            "1",
        )
    )
    return r


def _sparse_relations(
    seed: int,
    n: int,
) -> tuple[SparseWorldTensor, SparseWorldTensor]:
    rng = random.Random(seed)
    a = TensorAxis("A", "A", tuple(f"a{i}" for i in range(n)))
    b = TensorAxis("B", "B", tuple(f"b{i}" for i in range(n)))
    c = TensorAxis("C", "C", tuple(f"c{i}" for i in range(n)))

    left = SparseWorldTensor("left", (a, b), value_kind="real")
    right = SparseWorldTensor("right", (b, c), value_kind="real")

    for i in range(n):
        for offset in (0, 17):
            left.set(
                (f"a{i}", f"b{(i + offset) % n}"),
                1.0 + rng.random(),
            )
        for offset in (0, 13, 29):
            right.set(
                (f"b{i}", f"c{(i + offset) % n}"),
                1.0 + rng.random(),
            )
    return left, right


def _sparse_lookup(product):
    product = product.coalesce()
    return {
        tuple(index): float(value)
        for index, value in zip(
            product.indices().t().tolist(),
            product.values().tolist(),
        )
    }


def run(seed: int = 19, n: int = 600) -> dict:
    registry = _registry()
    plan = registry.compile("travel_pressure")

    total_primitives = len(registry.primitives)
    total_views = len(registry.views)

    left, right = _sparse_relations(seed, n)

    start = time.perf_counter()
    incremental = IncrementalBinaryView(left, right)
    initial_ms = (time.perf_counter() - start) * 1000.0

    # Add one right-side coordinate. The exact impact set is determined by the
    # left relation's support on b0, rather than the whole output tensor.
    changed = ("b0", f"c{211 % n}")
    right.set(changed, 9.0)

    start = time.perf_counter()
    cells = incremental.apply_delta(
        left,
        right,
        BinaryCoordinateDelta("right", changed),
    )
    incremental_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    full = sparse_binary_compose(left, right).coalesce()
    full_ms = (time.perf_counter() - start) * 1000.0
    full_lookup = _sparse_lookup(full)

    all_match = True
    for cell in cells:
        x, z = cell.coordinate
        key = (
            left.axes[0].position(x),
            right.axes[1].position(z),
        )
        if abs(cell.value - full_lookup.get(key, 0.0)) > 1e-5:
            all_match = False
            break

    return {
        "experiment": "exp106_incremental_query_runtime",
        "seed": seed,
        "axis_size": n,
        "query": {
            "target": plan.target,
            "selected_primitive_count": len(plan.primitive_tensors),
            "total_primitive_count": total_primitives,
            "selected_view_count": len(plan.view_order),
            "total_view_count": total_views,
            "selected_primitives": list(plan.primitive_tensors),
            "view_order": list(plan.view_order),
            "excluded_primitive_fraction": (
                1.0 - len(plan.primitive_tensors) / total_primitives
            ),
        },
        "incremental": {
            "initial_materialization_ms": initial_ms,
            "affected_output_cells": len(cells),
            "incremental_update_ms": incremental_ms,
            "full_recompute_ms": full_ms,
            "speedup_vs_full": (
                full_ms / incremental_ms
                if incremental_ms > 0
                else None
            ),
            "matches_full_on_affected_cells": all_match,
            "output_nnz": incremental.nnz,
        },
        "claim_boundary": (
            "Synthetic source-side mechanism benchmark only. Re-run identical "
            "commit on Mac and OCI before making performance claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp106.json")
    parser.add_argument("--n", type=int, default=600)
    args = parser.parse_args()

    result = run(n=args.n)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
