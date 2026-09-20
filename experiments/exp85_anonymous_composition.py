"""Experiment 85: Tensor Logic composition over anonymous learned predicates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from tensor_logic.research.anonymous_composition import (
    run_anonymous_composition_benchmark,
)


SOURCE_ISSUE_78_COMMIT = "3ef1423b72d89f8feda2221abdadca0a18f71233"


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
    parser.add_argument("--train-frames", type=int, default=200)
    parser.add_argument("--induction-frames", type=int, default=80)
    parser.add_argument("--heldout-frames", type=int, default=80)
    parser.add_argument(
        "--out",
        default="experiments/exp85_anonymous_composition_data/results.json",
    )
    args = parser.parse_args()

    result = run_anonymous_composition_benchmark(
        train_frames=args.train_frames,
        induction_frames=args.induction_frames,
        heldout_frames=args.heldout_frames,
    )
    result["provenance"] = {
        "commit": current_commit(),
        "source_issue_78_commit": SOURCE_ISSUE_78_COMMIT,
        "experiment": "exp85_anonymous_composition",
        "semantic_names_visible_to_conjecturer": False,
        "evaluator_mapping_used_for_proposal_generation": False,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("exp85: anonymous predicate composition")
    for name, condition in result["conditions"].items():
        print(
            f"{name:>8}: body={condition['candidate']} "
            f"induction_f1={condition['induction_f1']:.4f} "
            f"heldout_f1={condition['heldout_f1']:.4f} "
            f"accepted={condition['accepted']}"
        )

    print("noise sweep:")
    for noise, condition in result["noise_sweep"].items():
        print(
            f"  {noise}: heldout_f1={condition['heldout_f1']:.4f} "
            f"accepted={condition['accepted']} "
            f"body={condition['candidate']}"
        )

    false_rule = result["false_rule_control"]
    print(
        "false-rule control: "
        f"body={false_rule['body']} "
        f"heldout_f1={false_rule['heldout_f1']:.4f} "
        f"accepted={false_rule['accepted']} "
        f"counterexample={false_rule['counterexample']}"
    )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
