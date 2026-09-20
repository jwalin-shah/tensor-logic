"""exp101: ACT-R-inspired activation vs recency-only retrieval.

Synthetic sanity benchmark. It does NOT claim psychological validation. The
purpose is to test whether a recency/frequency + task-association mechanism can
recover goal-relevant older memories that a pure recency heuristic misses.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from tensor_logic.actr_memory import rank_memories


def run(seed: int = 7, cases: int = 2000) -> dict:
    rng = random.Random(seed)
    now = 1000.0
    recency_correct = 0
    activation_correct = 0
    activation_ms = 0.0

    for case_index in range(cases):
        target = f"target:{case_index}"
        distractor = f"distractor:{case_index}"

        # The distractor is newer. The target is older but repeatedly used and
        # strongly associated with the current goal.
        target_times = [
            now - rng.uniform(80.0, 180.0),
            now - rng.uniform(30.0, 80.0),
            now - rng.uniform(15.0, 40.0),
        ]
        distractor_times = [now - rng.uniform(1.0, 12.0)]

        most_recent = max(
            (
                (max(target_times), target),
                (max(distractor_times), distractor),
            )
        )[1]
        recency_correct += int(most_recent == target)

        start = time.perf_counter()
        ranked = rank_memories(
            {
                target: target_times,
                distractor: distractor_times,
            },
            now=now,
            associative={
                target: rng.uniform(1.2, 2.0),
                distractor: 0.0,
            },
            k=1,
        )
        activation_ms += (time.perf_counter() - start) * 1000.0
        activation_correct += int(ranked[0].item_id == target)

    return {
        "experiment": "exp101_actr_retrieval",
        "synthetic_warning": (
            "Goal relevance is injected by construction; this is a mechanism "
            "sanity check, not evidence of human-like memory."
        ),
        "seed": seed,
        "cases": cases,
        "recency_accuracy": recency_correct / cases,
        "activation_accuracy": activation_correct / cases,
        "activation_total_ms": activation_ms,
        "activation_us_per_case": activation_ms * 1000.0 / cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp101.json")
    args = parser.parse_args()

    result = run()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
