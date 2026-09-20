"""Experiment 84: unlabeled relation discovery -> anonymous TL predicates.

No relation names or truth values are supplied to the learner. Ground truth is
used only after fitting to permutation-match anonymous channels for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tensor_logic.research.unlabeled_relations import run_unlabeled_benchmark


def current_commit() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frames", type=int, default=400)
    parser.add_argument("--test-frames", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=11)
    parser.add_argument("--test-seed", type=int, default=22)
    parser.add_argument("--random-seed", type=int, default=33)
    parser.add_argument(
        "--out",
        default="experiments/exp84_unlabeled_relation_data/results.json",
    )
    args = parser.parse_args()

    result = run_unlabeled_benchmark(
        train_frames=args.train_frames,
        test_frames=args.test_frames,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        random_seed=args.random_seed,
    )
    result["provenance"] = {
        "commit": current_commit(),
        "experiment": "exp84_unlabeled_relations",
        "learner_receives_relation_labels": False,
        "evaluation_only_semantic_alignment": True,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    random_f1 = result["methods"]["random"]["mean_matched_f1"]
    kmeans_f1 = result["methods"]["kmeans"]["mean_matched_f1"]
    pca_f1 = result["methods"]["pca"]["mean_matched_f1"]
    delta = result["pca_minus_random"]

    print("exp84: unlabeled relation discovery")
    print(f"random matched F1: {random_f1:.4f}")
    print(f"kmeans matched F1: {kmeans_f1:.4f}")
    print(f"PCA matched F1:    {pca_f1:.4f}")
    print(f"PCA - random:      {delta:+.4f}")
    print("PCA evaluator mapping:")
    for match in result["methods"]["pca"]["matches"]:
        polarity = " inverted" if match["inverted"] else ""
        print(
            f"  {match['channel']} -> {match['relation']}: "
            f"F1={match['f1']:.4f}{polarity}"
        )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
