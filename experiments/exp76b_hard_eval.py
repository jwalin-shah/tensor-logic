"""
exp76b: build experiments/exp76_data/eval_hard.jsonl.

Stresses three axes the in-distribution eval doesn't:
  1. Held-out names (none appear in exp76a's NAMES)
  2. Larger graphs (16-20 people, ~2x the in-distribution size)
  3. Paraphrased questions (template variants not seen in training)

If (C) sft+tool stays at 100% on this set, the harness routing is genuinely
robust. If it drops, the gap tells us which axis to dig into next (compare to
a half-hard variant — names+graph only, original templates — for ablation).
"""

import json
import random
from pathlib import Path

from exp76a_rule_traces import (
    RULE_BODIES,
    derive_siblings,
    parent_set,
    sibling_set,
    EVALUATORS,
    gold_tag,
)

# Held-out names — no overlap with exp76a's NAMES.
HARD_NAMES = [
    "ulysses", "victor", "wendy", "xavier", "yara", "zane",
    "amos", "blair", "cleo", "dax", "elsa", "finn",
    "gus", "hugo", "iggy", "june", "kira", "luca",
    "mona", "nico", "opal", "piper",
]

# Multiple paraphrases per rule. None match exp76a's QUERY_PHRASING templates.
HARD_PHRASINGS = {
    "grandparent": [
        "Is {x} the grandparent of {y}?",
        "Does {x} have {y} as a grandchild?",
        "Would you call {x} a grandfather or grandmother of {y}?",
    ],
    "uncle": [
        "Is {y} a niece or nephew of {x}?",
        "Would {x} be considered an uncle to {y}?",
        "Is {x} uncle to {y}?",
    ],
    "cousin": [
        "Are {x} and {y} cousins?",
        "Is {x} a first cousin of {y}?",
        "Would {y} be {x}'s cousin?",
    ],
    "great_uncle": [
        "Is {x} a great-uncle to {y}?",
        "Would {y} call {x} their great-uncle?",
        "Is {y} the grandniece or grandnephew of {x}?",
    ],
}


def gen_family_hard(rng):
    """Larger graphs: 16-20 people."""
    n_people = rng.randint(16, 20)
    people = rng.sample(HARD_NAMES, n_people)
    parents = []
    n_roots = rng.randint(2, 3)
    for i in range(n_roots, len(people)):
        possible = people[max(0, i - 5):i]
        p = rng.choice(possible)
        parents.append([p, people[i]])
    return parents, people


def make_hard_trace(rng, rule_type, want_positive):
    for _ in range(60):
        parents, people = gen_family_hard(rng)
        siblings = derive_siblings(parents)
        P, S = parent_set(parents), sibling_set(siblings)
        x, y = rng.sample(people, 2)
        gold = EVALUATORS[rule_type](x, y, P, S)
        if gold == want_positive:
            phrasing = rng.choice(HARD_PHRASINGS[rule_type])
            return {
                "graph": {"parent": parents, "sibling": siblings},
                "query": phrasing.format(x=x, y=y),
                "rule_type": rule_type,
                "subject": x,
                "object": y,
                "gold_tool_call": gold_tag(rule_type, x, y),
                "gold_answer": "yes" if gold else "no",
            }
    # Fallback — accept whatever the last iteration produced.
    phrasing = rng.choice(HARD_PHRASINGS[rule_type])
    return {
        "graph": {"parent": parents, "sibling": siblings},
        "query": phrasing.format(x=x, y=y),
        "rule_type": rule_type,
        "subject": x,
        "object": y,
        "gold_tool_call": gold_tag(rule_type, x, y),
        "gold_answer": "yes" if gold else "no",
    }


def main():
    rng = random.Random(76)
    out_path = Path(__file__).parent / "exp76_data" / "eval_hard.jsonl"
    rules = list(RULE_BODIES.keys())
    n = 200
    rule_counts = {}
    ans_counts = {"yes": 0, "no": 0}

    with out_path.open("w") as f:
        for i in range(n):
            rule_type = rules[i % len(rules)]
            want_pos = (i // len(rules)) % 2 == 0
            trace = make_hard_trace(rng, rule_type, want_pos)
            f.write(json.dumps(trace) + "\n")
            rule_counts[rule_type] = rule_counts.get(rule_type, 0) + 1
            ans_counts[trace["gold_answer"]] += 1

    print(f"wrote {out_path} ({n} traces)")
    print(f"  rules: {sorted(rule_counts.items())}")
    print(f"  answers: {ans_counts}")


if __name__ == "__main__":
    main()
