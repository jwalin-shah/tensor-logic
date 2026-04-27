"""
exp76a (step 1/2): generate (graph, query, gold-<tl_rule>, gold-answer)
traces for multi-relation rule-chain SFT.

Tests whether a small instruct LM can learn to PICK THE RIGHT RULE from a
menu of 4 multi-relation rules, not just emit a single closure tag like
exp60d. Each trace's gold tool call is one of:

  grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
  uncle(X, Y)       :- sibling(X, P), parent(P, Y).
  cousin(X, Y)      :- parent(P, X), sibling(P, Q), parent(Q, Y).
  great_uncle(X, Y) :- sibling(X, P), parent(P, Q), parent(Q, Y).

The substrate side is exp65's rule-chain harness — already 9/9 on these
exact rule shapes. exp76 is a pure LM-side question: can SFT teach the
model to route the query to the right rule?

Output:
  experiments/exp76_data/{train,eval}.jsonl

Trace format:
  {
    "graph": {"parent": [...], "sibling": [...]},
    "query": "Is alice a grandparent of dave?",
    "rule_type": "grandparent",
    "gold_tool_call": "<tl_rule head=\"grandparent(X, Y)\" body=\"parent(X, Z), parent(Z, Y)\"></tl_rule>",
    "subject": "alice",
    "object": "dave",
    "gold_answer": "yes"
  }

Note: the gold_tool_call is the RULE DEFINITION; the SLM emits the same
tag for every grandparent query (the rule is fixed). The (subject, object)
fields are passed alongside; exp60d's harness extension (in this same
patch) will read them to query the head tensor returned by exp65's
evaluate_rule.
"""

import json
import random
from pathlib import Path

NAMES = [
    "alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry",
    "iris", "jack", "kate", "leo", "mia", "noah", "olive", "pete",
]


# ---- Rule definitions (rule body → fixed gold tag) ----

RULE_BODIES = {
    "grandparent": ("grandparent(X, Y)", "parent(X, Z), parent(Z, Y)"),
    "uncle":       ("uncle(X, Y)",       "sibling(X, P), parent(P, Y)"),
    "cousin":      ("cousin(X, Y)",      "parent(P, X), sibling(P, Q), parent(Q, Y)"),
    "great_uncle": ("great_uncle(X, Y)", "sibling(X, P), parent(P, Q), parent(Q, Y)"),
}

QUERY_PHRASING = {
    "grandparent": "Is {x} a grandparent of {y}?",
    "uncle":       "Is {x} an uncle of {y}?",
    "cousin":      "Is {x} a cousin of {y}?",
    "great_uncle": "Is {x} a great-uncle of {y}?",
}


def gold_tag(rule_type: str, subject: str, obj: str) -> str:
    """Self-contained rule tag: head + body + concrete query (subj, obj).

    Strict superset of exp65's <tl_rule> format — subject/object are new
    optional attrs the exp76 harness routes on. Keeps the LM's emission
    end-to-end (pick rule + extract entities), parallel to exp60d's
    <tl_closure>.
    """
    head, body = RULE_BODIES[rule_type]
    return (
        f'<tl_rule head="{head}" body="{body}" '
        f'subject="{subject}" object="{obj}"></tl_rule>'
    )


# ---- Graph generation ----

def gen_family(rng, n_people=None):
    """Random family tree: assign each person a parent from earlier ones."""
    if n_people is None:
        n_people = rng.randint(8, 14)
    people = rng.sample(NAMES, n_people)
    parents = []  # [parent, child]
    n_roots = rng.randint(2, 3)
    for i in range(n_roots, len(people)):
        possible = people[max(0, i - 5):i]  # bias toward recent
        p = rng.choice(possible)
        parents.append([p, people[i]])
    return parents, people


def derive_siblings(parents):
    """Two people are siblings iff they share a parent."""
    by_parent = {}
    for p, c in parents:
        by_parent.setdefault(p, []).append(c)
    siblings = []
    for kids in by_parent.values():
        for i, a in enumerate(kids):
            for b in kids[i + 1:]:
                siblings.append([a, b])
                siblings.append([b, a])  # symmetric
    return siblings


# ---- Ground-truth rule evaluators (pure Python, no torch) ----

def parent_set(parents):
    """{(p, c) tuples}."""
    return set(tuple(pc) for pc in parents)


def sibling_set(siblings):
    return set(tuple(s) for s in siblings)


def is_grandparent(x, y, P, _S):
    return any((x, z) in P and (z, y) in P for z in {a for a, _ in P} | {b for _, b in P})


def is_uncle(x, y, P, S):
    # sibling(x, p), parent(p, y)
    return any((x, p) in S and (p, y) in P for p in {a for a, _ in P} | {b for _, b in P})


def is_cousin(x, y, P, S):
    # parent(p, x), sibling(p, q), parent(q, y), p != q
    parents_of_x = [p for p, c in P if c == x]
    for p in parents_of_x:
        for (a, b) in S:
            if a == p and a != b and (b, y) in P:
                return True
    return False


def is_great_uncle(x, y, P, S):
    # sibling(x, p), parent(p, q), parent(q, y)
    for (a, b) in S:
        if a == x:
            for q in {bb for _, bb in P}:
                if (b, q) in P and (q, y) in P:
                    return True
    return False


EVALUATORS = {
    "grandparent": is_grandparent,
    "uncle":       is_uncle,
    "cousin":      is_cousin,
    "great_uncle": is_great_uncle,
}


# ---- Trace generation ----

def make_trace(rng, rule_type, want_positive=None):
    """Generate one trace for a given rule_type.

    If want_positive is True, retry until we find an (x, y) pair where the
    rule fires; if False, until we find a pair where it doesn't. None: any.
    """
    for _ in range(40):  # bounded retries
        parents, people = gen_family(rng)
        siblings = derive_siblings(parents)
        P, S = parent_set(parents), sibling_set(siblings)
        x, y = rng.sample(people, 2)
        gold = EVALUATORS[rule_type](x, y, P, S)
        if want_positive is None or gold == want_positive:
            return {
                "graph": {"parent": parents, "sibling": siblings},
                "query": QUERY_PHRASING[rule_type].format(x=x, y=y),
                "rule_type": rule_type,
                "subject": x,
                "object": y,
                "gold_tool_call": gold_tag(rule_type, x, y),
                "gold_answer": "yes" if gold else "no",
            }
    # Fallback: return whatever we have
    return {
        "graph": {"parent": parents, "sibling": siblings},
        "query": QUERY_PHRASING[rule_type].format(x=x, y=y),
        "rule_type": rule_type,
        "subject": x,
        "object": y,
        "gold_tool_call": gold_tag(rule_type, x, y),
        "gold_answer": "yes" if gold else "no",
    }


def main():
    rng = random.Random(7)
    out_dir = Path(__file__).parent / "exp76_data"
    out_dir.mkdir(exist_ok=True)

    rules = list(RULE_BODIES.keys())  # 4 rule types
    splits = [("train", 2000), ("eval", 400)]

    for name, n in splits:
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for i in range(n):
                rule_type = rules[i % len(rules)]
                # 50/50 positive vs negative within each rule type
                want_pos = (i // len(rules)) % 2 == 0
                trace = make_trace(rng, rule_type, want_positive=want_pos)
                f.write(json.dumps(trace) + "\n")
        # Distribution
        rule_counts = {}
        ans_counts = {"yes": 0, "no": 0}
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                rule_counts[d["rule_type"]] = rule_counts.get(d["rule_type"], 0) + 1
                ans_counts[d["gold_answer"]] += 1
        print(f"  {name}.jsonl: {n} traces")
        print(f"    rules: {sorted(rule_counts.items())}")
        print(f"    answers: {ans_counts}")

    print()
    print(f"Wrote train + eval JSONL files to {out_dir}/")


if __name__ == "__main__":
    main()
