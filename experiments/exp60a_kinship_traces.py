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

# Larger pool for the --hard eval set: 40 names, supports 30+-person graphs
# and hop chains of length 6-10.
NAMES_HARD = NAMES + [
    "mia", "noah", "olive", "pete", "quinn", "ruth", "sam", "tess",
    "umar", "vera", "wade", "xena", "yusuf", "zara", "amy", "ben",
    "cleo", "dan", "ella", "finn", "gus", "hana", "ivy", "jonah",
    "luna", "milo", "nina", "owen",
]


def add_distractors(graph: dict, people, rng) -> dict:
    """Add married_to / friend_of relations to the graph as distractors.

    These relations have NOTHING to do with the ancestor/descendant query —
    a correctly SFT'd model should always emit `relation="parent"` and
    ignore these. If accuracy drops on --hard, it's diagnostic: the model
    is confusing relations.
    """
    n = len(people)
    n_marriages = max(1, n // 5)
    n_friendships = max(2, n // 3)
    married = []
    for _ in range(n_marriages):
        a, b = rng.sample(people, 2)
        married.append([a, b])
    friends = []
    for _ in range(n_friendships):
        a, b = rng.sample(people, 2)
        friends.append([a, b])
    graph["married_to"] = married
    graph["friend_of"] = friends
    return graph


def gen_random_family(rng, n_people=None, max_children=3, name_pool=None, chain_bias=False):
    """Generate a small random family tree as a parent relation.

    Returns: dict {"parent": [[parent, child], ...]}.
    Tree is a random forest over a subset of `name_pool` (default: NAMES).

    chain_bias=True biases parent selection toward the most-recent person,
    producing long linear chains needed for hops 6-10 in the --hard set.
    """
    pool = name_pool or NAMES
    if n_people is None:
        n_people = rng.randint(6, len(pool))
    people = rng.sample(pool, n_people)
    rng.shuffle(people)
    parents = []
    # Random tree: assign each person (except first 1-2 roots) a parent
    # from someone earlier in the sequence.
    n_roots = rng.randint(1, 2)
    for i in range(n_roots, len(people)):
        possible_parents = people[:i]
        if chain_bias:
            # Heavily prefer the most-recent ancestor → long linear chains
            weights = [(j + 1) ** 3 for j in range(len(possible_parents))]
        else:
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


def make_trace(rng, target_hops=None, hard=False):
    """Generate one (graph, query, gold) trace.

    Query types:
      - ancestor: "is X an ancestor of Y?"
      - descendant: "is X a descendant of Y?"

    hard=True uses NAMES_HARD (40 names), 25-35 person graphs with linear
    chains (chain_bias=True), and adds married_to / friend_of distractor
    relations the model must ignore.
    """
    if hard:
        n_people = rng.randint(25, 35)
        graph, people = gen_random_family(
            rng, n_people=n_people, max_children=2,
            name_pool=NAMES_HARD, chain_bias=True,
        )
        graph = add_distractors(graph, people, rng)
    else:
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", action="store_true",
                    help="Generate harder eval set: 25-35 person graphs, "
                         "hops 6-10, distractor relations.")
    ap.add_argument("--out", default=None,
                    help="Output filename (default: train+eval for normal, "
                         "eval_hard.jsonl for --hard).")
    ap.add_argument("--n", type=int, default=200,
                    help="Number of traces (only used with --hard).")
    args = ap.parse_args()

    rng = random.Random(42 if not args.hard else 1337)
    out_dir = Path(__file__).parent / "exp60_data"
    out_dir.mkdir(exist_ok=True)

    if args.hard:
        out_path = out_dir / (args.out or "eval_hard.jsonl")
        hop_targets = [6, 7, 8, 9, 10]
        with out_path.open("w") as f:
            for i in range(args.n):
                target = hop_targets[i % len(hop_targets)]
                trace = make_trace(rng, target_hops=target, hard=True)
                f.write(json.dumps(trace) + "\n")
        hops_counter = {}
        with out_path.open() as f:
            for line in f:
                d = json.loads(line)
                hops_counter[d["hops"]] = hops_counter.get(d["hops"], 0) + 1
        print(f"  {out_path.name}: {args.n} HARD traces, hops = {sorted(hops_counter.items())}")
        print(f"  graphs: 25-35 people, distractor relations: married_to, friend_of")
        return

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
        hops_counter = {}
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                hops_counter[d["hops"]] = hops_counter.get(d["hops"], 0) + 1
        print(f"  {name}.jsonl: {n} traces, hops distribution = {sorted(hops_counter.items())}")

    print()
    print(f"Wrote training + eval JSONL files to {out_dir}/")


if __name__ == "__main__":
    main()
