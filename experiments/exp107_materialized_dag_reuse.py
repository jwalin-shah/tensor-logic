"""exp107: repeated-query materialized DAG reuse.

Measures derivation counts, not hardware performance. The same deterministic
query is executed repeatedly under controlled primitive revisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensor_logic.materialized_dag import (
    MaterializedDagCache,
    execute_materialized_dag,
    primitive_artifact,
)
from tensor_logic.query_runtime import (
    DerivedViewSpec,
    PrimitiveTensorSpec,
    QueryRegistry,
)
from tensor_logic.selective_executor import QueryOperatorRegistry


def _registry() -> QueryRegistry:
    r = QueryRegistry()
    for name in ("p0", "p1", "p2", "p3", "leaf"):
        r.add_primitive(PrimitiveTensorSpec(name))
    r.add_view(DerivedViewSpec("v1", ("p0", "p1"), "add", "1"))
    r.add_view(DerivedViewSpec("v2", ("v1", "p2"), "mul", "1"))
    r.add_view(DerivedViewSpec("v3", ("v2", "p3"), "add", "1"))
    r.add_view(DerivedViewSpec("target", ("v3", "leaf"), "mul", "1"))
    return r


def run(iterations: int = 100) -> dict:
    registry = _registry()
    query = registry.compile("target")
    cache = MaterializedDagCache()

    values = {
        "p0": 2,
        "p1": 3,
        "p2": 4,
        "p3": 5,
        "leaf": 2,
    }
    revisions = {name: "0" for name in values}

    operator_calls = {"add": 0, "mul": 0}

    operators = QueryOperatorRegistry()

    def add(inputs, params):
        operator_calls["add"] += 1
        return inputs[0] + inputs[1]

    def mul(inputs, params):
        operator_calls["mul"] += 1
        return inputs[0] * inputs[1]

    operators.register("add", "1", add)
    operators.register("mul", "1", mul)

    total_hits = 0
    total_misses = 0
    outputs = []

    def loader(name):
        return primitive_artifact(
            name,
            values[name],
            revision=revisions[name],
        )

    for step in range(iterations):
        if step > 0 and step % 10 == 0:
            values["leaf"] += 1
            revisions["leaf"] = str(step)

        if step > 0 and step % 25 == 0:
            values["p0"] += 1
            revisions["p0"] = str(step)

        result = execute_materialized_dag(
            registry,
            query,
            loader,
            operators,
            cache,
        )
        total_hits += len(result.trace.cache_hits)
        total_misses += len(result.trace.cache_misses)
        outputs.append(result.output.value)

    stateless_derivations = iterations * len(query.view_order)
    materialized_derivations = (
        operator_calls["add"] + operator_calls["mul"]
    )

    return {
        "experiment": "exp107_materialized_dag_reuse",
        "iterations": iterations,
        "views_per_query": len(query.view_order),
        "stateless_derivations": stateless_derivations,
        "materialized_derivations": materialized_derivations,
        "derivations_avoided": (
            stateless_derivations - materialized_derivations
        ),
        "derivation_reduction_fraction": (
            1.0 - materialized_derivations / stateless_derivations
        ),
        "cache_hits": total_hits,
        "cache_misses": total_misses,
        "operator_calls": operator_calls,
        "cache_entries": cache.size,
        "first_output": outputs[0],
        "last_output": outputs[-1],
        "claim_boundary": (
            "Counts deterministic recomputation avoided. It does not establish "
            "token/latency savings for real RAG or LifeOps workloads."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", default="/tmp/exp107.json")
    args = parser.parse_args()

    result = run(args.iterations)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
