"""
exp81 sweep: load model once, run all mode/target combos sequentially.
Usage: python3 exp81_sweep.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from exp81_optimize_rule_induction import run_exp81, Exp81Config

RUNS = [
    dict(target="grandparent", mode="hard",    max_steps=30),
    dict(target="grandparent", mode="hard_v2", max_steps=30),
    dict(target="uncle",       mode="hard_v2", max_steps=30),
    dict(target="great_uncle", mode="hard_v2", max_steps=50),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    print(f"\nModel: {args.model}")
    print("=" * 80)

    results = []
    for run in RUNS:
        cfg = Exp81Config(
            target=run["target"],
            mode=run["mode"],
            max_steps=run["max_steps"],
            seed=args.seed,
            model=args.model,
        )
        print(f"\n>>> {cfg.target} / {cfg.mode}  (max_steps={cfg.max_steps})")
        result = run_exp81(cfg)
        results.append(result)

        print(f"  Steps to F1=1.0 : {result['steps_to_f1_1']}")
        print(f"  Brute-force tmpl : {result['brute_force_template_count']}")
        print(f"  Accepted F1      : {result['accepted_f1']:.3f}")
        print(f"  Gold covered     : {result['gold_covered_rate']:.0%}")
        print(f"  Pruning rate     : {result['proposer_pruning_rate']:.0%}")
        print(f"  Failure mode     : {result['failure_mode']}")
        if result['steps_to_f1_1']:
            ratio = result['steps_to_f1_1'] / result['brute_force_template_count']
            print(f"  Ratio            : {ratio:.3f}  → PASS")
        else:
            print(f"  → FAIL")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"{'target':<16} {'mode':<10} {'steps':>6} {'templates':>10} {'F1':>6} {'failure'}")
    print("-" * 70)
    for r in results:
        steps = str(r['steps_to_f1_1']) if r['steps_to_f1_1'] else "DNF"
        print(f"{r['target']:<16} {r['mode']:<10} {steps:>6} {r['brute_force_template_count']:>10} {r['accepted_f1']:>6.3f}  {r['failure_mode'] or 'PASS'}")


if __name__ == "__main__":
    main()
