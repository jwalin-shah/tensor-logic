"""
exp60a (step 1/4 of the TL-as-tool integration line):
synthetic kinship-graph trace generator.

Goal: produce (graph, query, gold-tool-call, gold-answer) tuples that an
LM can later be SFT'd on (in exp60d, deferred until LM access). Each
trace is fully self-contained — graph is in the prompt, the gold answer
is computable from the graph by deterministic TL closure, and the gold
tool-call format is fixed.

This step has no LM dependency. It produces a JSONL file of training and
eval traces. The downstream pieces (exp60b harness, exp60c deterministic
baseline, exp60d SFT) consume this file.

Trace format (JSONL, one example per line):
{
  "graph": {"parent": [["alice","bob"], ["bob","carol"], ...]},
  "query": "Is carol a descendant of alice?",
  "hops": 2,
  "gold_tool_call": "<tl_closure relation=\"parent\">carol descendant alice</tl_closure>",
  "gold_answer": "yes"
}

Why kinship?
  - Domingos's TL paper uses kinship as the canonical example: ancestor
    is parent's transitive closure, uncle is parent ∘ sibling, etc.
  - Hops are unambiguous and tunable (we control train ∈ {1,2,3}, test
    ∈ {1..5}) — direct test of compositional generalization.
  - Closure has an exact ground-truth answer for every query, so any
    LM error is unambiguously the LM's, not the substrate's.

Sweep: NAMES = 12 people, GRAPHS = 1000 train + 200 eval, 4 query types
(ancestor, descendant, sibling-via, cousin-via).
"""

import json
import random
from pathlib import Path

NAMES = [
    "alice", "bob", "carol", "dave", "eve", "frank",
    "grace", "henry", "iris", "jack", "kate", "leo",
]


def gen_random_family(rng, n_people=None, max_children=3):
    """Generate a small random family tree as a parent relation.

    Returns: dict {"parent": [[parent, child], ...]}.
    Tree is a random forest over a subset of NAMES, max 4 generations deep.
    """
    if n_people is None:
        n_people = rng.randint(6, len(NAMES))
    people = rng.sample(NAMES, n_people)
    rng.shuffle(people)
    parents = []
    # Random tree: assign each person (except first 1-2 roots) a parent
    # from someone earlier in the sequence.
    n_roots = rng.randint(1, 2)
    for i in range(n_roots, len(people)):
        possible_parents = people[:i]
        # Bias toward shallow trees to make multi-hop queries non-trivial
        weights = [max_children - min(max_children - 1, sum(1 for p, c in parents if p == pp)) for pp in possible_parents]
        if sum(weights) <= 0:
            weights = [1] * len(possible_parents)
        parent = rng.choices(possible_parents, weights=weights, k=1)[0]
        parents.append([parent, people[i]])
    return {"parent": parents}, people


def parent_dict(parents_list):
    """Convert parent relation list to dict[child] = parent."""
    return {c: p for p, c in parents_list}


def ancestors_of(person, parents):
    """Iterative chain-up via parent relation."""
    pdict = parent_dict(parents)
    out = []
    cur = pdict.get(person)
    seen = set()
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = pdict.get(cur)
    return out


def descendants_of(person, parents):
    """All descendants via DFS."""
    children = {}
    for p, c in parents:
        children.setdefault(p, []).append(c)
    out = []
    stack = list(children.get(person, []))
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(children.get(x, []))
    return out


def hops_between(a, b, parents):
    """Min hops from a to b along the parent relation in EITHER direction.
    None if unrelated."""
    # try a → b (a descendant of b)
    pdict = parent_dict(parents)
    cur = a
    seen = {a: 0}
    hops = 0
    while cur in pdict:
        cur = pdict[cur]
        hops += 1
        if cur == b:
            return hops
        seen[cur] = hops
    # try b → a (b descendant of a)
    cur = b
    hops = 0
    while cur in pdict:
        cur = pdict[cur]
        hops += 1
        if cur == a:
            return hops
    return None


def make_trace(rng, target_hops=None):
    """Generate one (graph, query, gold) trace.

    Query types:
      - ancestor: "is X an ancestor of Y?"
      - descendant: "is X a descendant of Y?"
    """
    graph, people = gen_random_family(rng)
    parents = graph["parent"]

    # Try to get a query at a target hop count if requested
    candidates = []
    for x in people:
        for y in people:
            if x == y:
                continue
            h = hops_between(x, y, parents)
            if h is None:
                # negative example — unrelated
                candidates.append((x, y, 0, False))
            else:
                candidates.append((x, y, h, True))
    rng.shuffle(candidates)

    if target_hops is not None:
        matching = [c for c in candidates if c[2] == target_hops]
        if matching:
            x, y, hops, related = rng.choice(matching)
        else:
            x, y, hops, related = candidates[0]
    else:
        x, y, hops, related = candidates[0]

    # Choose query phrasing
    direction = rng.choice(["ancestor", "descendant"])
    if direction == "ancestor":
        # "is X an ancestor of Y?"  → True iff X in ancestors_of(Y)
        gold = (x in ancestors_of(y, parents))
        query = f"Is {x} an ancestor of {y}?"
    else:
        gold = (x in descendants_of(y, parents))
        query = f"Is {x} a descendant of {y}?"

    tool_call = (
        f'<tl_closure relation="parent" '
        f'query="{direction}" subject="{x}" object="{y}">'
        f'</tl_closure>'
    )
    answer = "yes" if gold else "no"
    return {
        "graph": graph,
        "query": query,
        "hops": hops,
        "related": related,
        "gold_tool_call": tool_call,
        "gold_answer": answer,
    }


def main():
    rng = random.Random(42)
    out_dir = Path(__file__).parent / "exp60_data"
    out_dir.mkdir(exist_ok=True)

    splits = [
        ("train", 1000, [1, 2, 3]),
        ("eval", 200, [1, 2, 3, 4, 5]),
    ]
    for name, n, hop_targets in splits:
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for i in range(n):
                target = hop_targets[i % len(hop_targets)]
                trace = make_trace(rng, target_hops=target)
                f.write(json.dumps(trace) + "\n")
        # Hop distribution
        hops_counter = {}
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                hops_counter[d["hops"]] = hops_counter.get(d["hops"], 0) + 1
        print(f"  {name}.jsonl: {n} traces, hops distribution = {sorted(hops_counter.items())}")

    print()
    print(f"Wrote training + eval JSONL files to {out_dir}/")
    print("Sample trace:")
    with (out_dir / "train.jsonl").open() as f:
        sample = json.loads(f.readline())
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
