"""exp104: restricted fixed-point kernel vs watchdog-bounded host computation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tensor_logic.kernel_contract import (
    run_with_fuel,
    transitive_closure,
)


def run(chain_nodes: int = 120, fuel: int = 100000) -> dict:
    edges = [
        (f"n{i}", f"n{i+1}")
        for i in range(chain_nodes - 1)
    ]

    start = time.perf_counter()
    closure = transitive_closure(edges)
    closure_ms = (time.perf_counter() - start) * 1000.0

    def nonhalting(counter: int):
        return counter + 1, False

    start = time.perf_counter()
    script = run_with_fuel(
        0,
        nonhalting,
        fuel=fuel,
    )
    script_ms = (time.perf_counter() - start) * 1000.0

    return {
        "experiment": "exp104_kernel_contract",
        "chain_nodes": chain_nodes,
        "input_edges": len(edges),
        "closure_pairs": len(closure.relation),
        "closure_iterations": closure.iterations,
        "closure_ms": closure_ms,
        "restricted_converged": closure.converged,
        "script_fuel": fuel,
        "script_steps": script.steps,
        "script_halted": script.halted,
        "script_fuel_exhausted": script.fuel_exhausted,
        "script_ms": script_ms,
        "claim_boundary": (
            "The unrestricted baseline is host-language computation constrained "
            "by an external watchdog. This benchmark does not prove the Tensor "
            "Logic kernel Turing complete."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp104.json")
    args = parser.parse_args()
    result = run()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
