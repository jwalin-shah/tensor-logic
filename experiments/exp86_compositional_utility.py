"""Experiment 86: latent representation utility beyond isolated relation recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tensor_logic.research.compositional_utility import (
    run_compositional_utility_benchmark,
)


SOURCE_ISSUE_79_COMMIT = "a3ac7097c320e0193d7bde547ea7d567f7072d56"


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
    parser.add_argument("--train-frames", type=int, default=180)
    parser.add_argument("--induction-frames", type=int, default=80)
    parser.add_argument("--heldout-frames", type=int, default=80)
    parser.add_argument(
        "--out",
        default="experiments/exp86_compositional_utility_data/results.json",
    )
    args = parser.parse_args()

    result = run_compositional_utility_benchmark(
        train_frames=args.train_frames,
        induction_frames=args.induction_frames,
        heldout_frames=args.heldout_frames,
    )
    result["provenance"] = {
        "commit": current_commit(),
        "source_issue_79_commit": SOURCE_ISSUE_79_COMMIT,
        "experiment": "exp86_compositional_utility",
        "semantic_labels_visible_to_representation_fit": False,
        "downstream_hidden_targets_visible_to_representation_fit": False,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("exp86: compositional utility benchmark")
    print("isolated recovery order:")
    print("  " + " > ".join(result["isolated_recovery_order"]))
    print("compositional utility order:")
    print("  " + " > ".join(result["compositional_utility_order"]))
    print(
        "PCA/k-means ordering disagreement: "
        f"{result['pca_kmeans_ordering_disagreement']}"
    )

    for name, metrics in result["representations"].items():
        print(
            f"{name:>12}: isolated={metrics['isolated_matched_f1']:.4f} "
            f"composition={metrics['mean_compositional_f1']:.4f} "
            f"counterfactual={metrics['mean_counterfactual_retraction_accuracy']:.4f} "
            f"brier={metrics['matched_brier']:.4f} "
            f"accepted_rules={metrics['accepted_rule_count']} "
            f"noise_fail={metrics['earliest_noise_failure']}"
        )
        for target_name, target in metrics["targets"].items():
            print(
                f"    {target_name}: body={target['candidate']} "
                f"heldout_f1={target['heldout_f1']:.4f} "
                f"accepted={target['accepted']} "
                f"retraction={target['counterfactual_retraction_accuracy']:.4f}"
            )

    print("next hypothesis:")
    print(f"  {result['next_hypothesis_from_measured_failure']}")
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
