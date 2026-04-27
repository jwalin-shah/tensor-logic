"""
exp76c: rebuild train.jsonl with paraphrased queries to test whether linguistic
diversity in training closes the OOD gap exposed by eval_hard.jsonl.

exp76b showed (C) sft+tool drops from 100% (in-dist) to 72.5% (paraphrased
queries), and tool-call validity drops to 79%. Hypothesis: the original SFT
overfit to a single query template per rule. If we train on diverse phrasings
(disjoint from eval_hard's), the model should learn the rule SEMANTICS rather
than the template, and (C) on eval_hard should jump.

Critical design point: TRAIN_PHRASINGS must be disjoint from exp76b's
HARD_PHRASINGS — else we measure memorization, not generalization.

Output: experiments/exp76_data/train_paraphrased.jsonl (2000 traces, same
graphs/answers as a fresh exp76a-style run, but each query randomly drawn
from a 5-paraphrase pool).
"""

import json
import random
from pathlib import Path

from exp76a_rule_traces import (
    NAMES,
    RULE_BODIES,
    derive_siblings,
    parent_set,
    sibling_set,
    EVALUATORS,
    gold_tag,
    gen_family,
)

# Disjoint from exp76a's QUERY_PHRASING (single template per rule) AND from
# exp76b's HARD_PHRASINGS (held-out eval). 5 paraphrases per rule for breadth.
TRAIN_PHRASINGS = {
    "grandparent": [
        "Is {x} a grandparent of {y}?",                       # original template, kept
        "Is {x} grandfather or grandmother of {y}?",
        "Is {y} a grandchild of {x}?",
        "Tell me whether {x} is a grandparent of {y}.",
        "Among the people listed, is {x} the grandparent of {y}?",
    ],
    "uncle": [
        "Is {x} an uncle of {y}?",                            # original template, kept
        "Is {x} the uncle of {y}?",
        "Is {y} a niece or nephew of {x}, through their parent's sibling?",
        "Tell me whether {x} qualifies as an uncle of {y}.",
        "Does {x} count as an uncle of {y}?",
    ],
    "cousin": [
        "Is {x} a cousin of {y}?",                            # original template, kept
        "Is {y} a cousin of {x}?",
        "Tell me whether {x} and {y} are cousins.",
        "Are {x} and {y} related as cousins?",
        "Does {x} share a cousin relationship with {y}?",
    ],
    "great_uncle": [
        "Is {x} a great-uncle of {y}?",                       # original template, kept
        "Is {x} a great uncle of {y}?",
        "Tell me whether {x} is a great-uncle of {y}.",
        "Is {x} the great-uncle of {y}?",
        "Does {x} qualify as a great-uncle of {y}?",
    ],
}


def make_paraphrased_trace(rng, rule_type, want_positive):
    for _ in range(40):
        parents, people = gen_family(rng)
        siblings = derive_siblings(parents)
        P, S = parent_set(parents), sibling_set(siblings)
        x, y = rng.sample(people, 2)
        gold = EVALUATORS[rule_type](x, y, P, S)
        if gold == want_positive:
            phrasing = rng.choice(TRAIN_PHRASINGS[rule_type])
            return {
                "graph": {"parent": parents, "sibling": siblings},
                "query": phrasing.format(x=x, y=y),
                "rule_type": rule_type,
                "subject": x,
                "object": y,
                "gold_tool_call": gold_tag(rule_type, x, y),
                "gold_answer": "yes" if gold else "no",
            }
    phrasing = rng.choice(TRAIN_PHRASINGS[rule_type])
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
    out_path = Path(__file__).parent / "exp76_data" / "train_paraphrased.jsonl"
    rules = list(RULE_BODIES.keys())
    n = 2000
    rule_counts = {}
    ans_counts = {"yes": 0, "no": 0}
    phrasing_counts = {}

    with out_path.open("w") as f:
        for i in range(n):
            rule_type = rules[i % len(rules)]
            want_pos = (i // len(rules)) % 2 == 0
            trace = make_paraphrased_trace(rng, rule_type, want_pos)
            f.write(json.dumps(trace) + "\n")
            rule_counts[rule_type] = rule_counts.get(rule_type, 0) + 1
            ans_counts[trace["gold_answer"]] += 1
            phrasing_counts[trace["query"].split("{")[0][:40]] = (
                phrasing_counts.get(trace["query"].split("{")[0][:40], 0) + 1
            )

    print(f"wrote {out_path} ({n} traces)")
    print(f"  rules: {sorted(rule_counts.items())}")
    print(f"  answers: {ans_counts}")


if __name__ == "__main__":
    main()
