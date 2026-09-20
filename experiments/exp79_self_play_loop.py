"""exp79: Self-play rule factory loop (SGS pattern).

The reusable conjecturer / solver / guide machinery now lives in
tensor_logic.research.rule_induction. This file remains the experiment entry
point and compatibility surface for the original easy/medium/hard modes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from tensor_logic.research.constants import (
    CONTACTS,
    QUERY_TARGETS,
    schema_with_distractors,
)
from tensor_logic.research.rule_induction import (
    Config,
    Conjecturer,
    Guide,
    RuleInductionLoop,
    RuleKB,
    Solver,
    candidate_prediction_cache,
    candidate_relation_names,
    cross_world_generalization,
    run_adversarial,
    run_self_play,
    verdict,
)

# Historical compatibility for callers that imported KB from this experiment.
KB = RuleKB

CONTACTS_WITH_DISTRACTORS = schema_with_distractors(CONTACTS)

EASY = Config(
    name="easy",
    schema=CONTACTS,
    n_pos=10,
    n_neg=10,
    noise=0.0,
    n_entities=50,
    max_steps=20,
    min_equiv=0.95,
    f1_threshold=0.85,
    n_attempts=1,
    vote_threshold=0.85,
    min_support=1,
)

MEDIUM = Config(
    name="medium",
    schema=CONTACTS_WITH_DISTRACTORS,
    n_pos=5,
    n_neg=5,
    noise=0.0,
    n_entities=50,
    max_steps=20,
    min_equiv=0.95,
    f1_threshold=0.85,
    n_attempts=5,
    vote_threshold=0.75,
    min_support=2,
)

HARD = Config(
    name="hard",
    schema=CONTACTS_WITH_DISTRACTORS,
    n_pos=5,
    n_neg=5,
    noise=0.20,
    n_entities=50,
    max_steps=20,
    min_equiv=0.95,
    f1_threshold=0.75,
    n_attempts=10,
    vote_threshold=0.50,
    min_support=3,
    exclusive=True,
)

VERY_HARD = Config(
    name="very_hard",
    schema=CONTACTS_WITH_DISTRACTORS,
    n_pos=3,
    n_neg=3,
    noise=0.20,
    n_entities=50,
    max_steps=20,
    min_equiv=0.95,
    f1_threshold=0.75,
    n_attempts=15,
    vote_threshold=0.40,
    min_support=3,
)


def print_mode_result(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Mode: {result['mode'].upper()}")
    print(f"{'=' * 60}")
    print(
        "Queries answered: "
        f"{result['answered_at_0']} -> {result['answered_after']} "
        f"/ {result['total_queries']}"
    )
    print(
        f"Rules induced:    {result['rules_induced']}  |  "
        f"Steps taken: {result['steps_taken']}"
    )
    print()
    print(
        f"{'Step':>4}  {'Target':<22}  {'Outcome':<20}  "
        f"{'F1':>5}  {'Equiv':>5}  {'Gen':>5}  Body"
    )
    print("-" * 95)
    for entry in result["step_log"]:
        body_str = " o ".join(entry["induced"]) if entry["induced"] else "-"
        gen_str = ""
        if entry.get("gen_scores"):
            gen_mean = (
                sum(entry["gen_scores"].values())
                / len(entry["gen_scores"])
            )
            gen_str = f"{gen_mean:.2f}"
        print(
            f"{entry['step']:>4}  {entry['target']:<22}  "
            f"{entry['outcome']:<20}  {entry['f1']:>5.3f}  "
            f"{entry['equiv']:>5.3f}  {gen_str:>5}  {body_str}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path("experiments") / "exp79_data" / "results.json"),
    )
    parser.add_argument(
        "--mode",
        choices=["all", "easy", "medium", "hard", "very_hard"],
        default="all",
    )
    args = parser.parse_args()

    configs = {
        "easy": EASY,
        "medium": MEDIUM,
        "hard": HARD,
        "very_hard": VERY_HARD,
    }
    modes = list(configs) if args.mode == "all" else [args.mode]

    print("exp79: self-play rule factory loop - reusable research module")
    print(f"Seed={args.seed}  Queries={len(QUERY_TARGETS)}")

    all_results: dict[str, dict] = {}
    overall_pass = True

    for mode_name in modes:
        cfg = configs[mode_name]
        started = time.perf_counter()
        result = run_self_play(cfg, seed=args.seed)
        result["wall_s"] = round(time.perf_counter() - started, 2)
        print_mode_result(result)

        if mode_name == "very_hard":
            ok = True
            message = "~ EXPECTED FAIL (identifiability boundary)"
        else:
            ok, message = verdict(result, cfg)
        print(f"\n{message}  ({result['wall_s']}s)")
        overall_pass = overall_pass and ok
        all_results[mode_name] = result

    if args.mode in ("all", "hard", "very_hard"):
        adversarial = run_adversarial(HARD, seed=args.seed)
        if adversarial["falsified"]:
            overall_pass = False
        all_results["adversarial"] = adversarial

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL PASS' if overall_pass else 'FAILURES DETECTED'}")
    print(f"{'=' * 60}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
