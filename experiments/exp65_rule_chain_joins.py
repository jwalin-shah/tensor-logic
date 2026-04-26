"""
exp65: extend the TL tool-call protocol to multi-relation joins.

exp60b's harness handled single-relation transitive closure
(`<tl_closure relation="parent">`). exp65 extends to Datalog-class
rules (`<tl_rule head="..." body="...">`) where the head relation is
derived from a join of body relations via TL einsum.

This makes the OPENHUMAN_TL_MEMO §1 multi-hop relational claim
operational, not just theoretical:
  uncle(X, Y)       :- sibling(X, P), parent(P, Y).
  grandparent(X, Y) :- parent(X, P), parent(P, Y).
  cousin(X, Y)      :- parent(P, X), sibling(P, Q), parent(Q, Y).

Encoding:
  Each binary relation is a tensor R[i, j] over a shared name index.
  Rule body `b1(X, Z), b2(Z, Y)` → `head[X, Y] = (B1 @ B2 > 0)`.
  Multi-step bodies → repeated einsum, contracting on shared index.

What this proves:
  - The exp60b tool-call machinery extends from single-relation closure
    to multi-relation joins with no architectural change.
  - The OPENHUMAN_TL_MEMO §1 family of queries (uncle/cousin/etc.) is
    a compositional special case of the same TL substrate.
  - Adding a new relation (or a new rule) costs zero training — the
    rule is a single einsum chain over existing tensors.

What this does NOT prove:
  - That an SLM can learn to emit `<tl_rule>` tags (that's exp60d).
  - That the rule chain stays performant at large name-set size — for
    n=10k, a single binary relation tensor is 10k×10k = 400 MB dense.
    For real openhuman scale, this needs the exp63 sparse substrate.
"""

import re
from pathlib import Path

import torch

# ---- Tag parser ----

# <tl_rule head="rel(X, Y)" body="rel1(X, Z), rel2(Z, Y)"></tl_rule>
RULE_RE = re.compile(
    r'<tl_rule\s+head="(?P<head>[^"]+)"\s+body="(?P<body>[^"]+)"\s*></tl_rule>'
)

# Atom: rel(VAR, VAR)
ATOM_RE = re.compile(r'(?P<rel>\w+)\s*\(\s*(?P<a>\w+)\s*,\s*(?P<b>\w+)\s*\)')


def parse_rule(s: str):
    m = RULE_RE.search(s)
    if not m:
        return None
    head_atom = ATOM_RE.search(m.group("head"))
    body_atoms = list(ATOM_RE.finditer(m.group("body")))
    if not head_atom or not body_atoms:
        return None
    head = (head_atom.group("rel"), head_atom.group("a"), head_atom.group("b"))
    body = [(a.group("rel"), a.group("a"), a.group("b")) for a in body_atoms]
    return head, body


def build_relation(name_to_idx, pairs):
    n = len(name_to_idx)
    R = torch.zeros(n, n)
    for src, dst in pairs:
        if src in name_to_idx and dst in name_to_idx:
            R[name_to_idx[src], name_to_idx[dst]] = 1.0
    return R


def evaluate_rule(graph: dict, rule):
    """Compute the head-relation tensor by joining body atoms.

    Pure forward chaining: variables shared across atoms are contracted
    via einsum; the head's two variables become the output dimensions.
    """
    head, body = rule
    head_rel, head_X, head_Y = head

    # Collect all names from all relations
    all_names = set()
    for rel_name in graph:
        for src, dst in graph[rel_name]:
            all_names.add(src)
            all_names.add(dst)
    all_names = sorted(all_names)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    n = len(all_names)

    # Build relation tensors
    rel_tensors = {
        rel_name: build_relation(name_to_idx, graph[rel_name])
        for rel_name in graph
    }

    # Walk body atoms; for each shared variable, contract.
    # Approach: assign axes to each variable; build an einsum string.
    # body: list of (rel, var1, var2). head: (rel, head_X, head_Y).
    var_to_axis = {}
    operands = []
    operand_strs = []
    for atom_rel, av1, av2 in body:
        if atom_rel not in rel_tensors:
            return None, f"unknown body relation: {atom_rel}"
        for v in [av1, av2]:
            if v not in var_to_axis:
                var_to_axis[v] = chr(ord("a") + len(var_to_axis))
        operands.append(rel_tensors[atom_rel])
        operand_strs.append(var_to_axis[av1] + var_to_axis[av2])
    if head_X not in var_to_axis or head_Y not in var_to_axis:
        return None, f"head variables {head_X}, {head_Y} not in body"
    out_str = var_to_axis[head_X] + var_to_axis[head_Y]
    eq = ",".join(operand_strs) + "->" + out_str
    R = torch.einsum(eq, *operands)
    head_tensor = (R > 0).float()
    return (head_tensor, all_names, name_to_idx, head_rel), None


def query_relation(head_result, subj, obj):
    head_tensor, all_names, name_to_idx, head_rel = head_result
    if subj not in name_to_idx or obj not in name_to_idx:
        return False
    return bool(head_tensor[name_to_idx[subj], name_to_idx[obj]] > 0)


def main():
    # Build a small extended-family graph with parent + sibling primitives.
    graph = {
        "parent": [
            ["alice", "bob"],
            ["alice", "carol"],
            ["bob", "dave"],
            ["bob", "eve"],
            ["carol", "frank"],
            ["carol", "grace"],
        ],
        "sibling": [
            # explicitly enumerate sibling pairs (could also be derived;
            # left as primitive for simplicity)
            ["bob", "carol"], ["carol", "bob"],
            ["dave", "eve"], ["eve", "dave"],
            ["frank", "grace"], ["grace", "frank"],
        ],
    }

    cases = [
        # (rule_str, query_subj, query_obj, expected, description)
        (
            '<tl_rule head="grandparent(X, Y)" body="parent(X, Z), parent(Z, Y)"></tl_rule>',
            "alice", "dave", True, "alice → bob → dave",
        ),
        (
            '<tl_rule head="grandparent(X, Y)" body="parent(X, Z), parent(Z, Y)"></tl_rule>',
            "alice", "frank", True, "alice → carol → frank",
        ),
        (
            '<tl_rule head="grandparent(X, Y)" body="parent(X, Z), parent(Z, Y)"></tl_rule>',
            "bob", "frank", False, "bob is not grandparent of frank (cousin)",
        ),
        (
            '<tl_rule head="uncle(X, Y)" body="sibling(X, P), parent(P, Y)"></tl_rule>',
            "carol", "dave", True, "carol is sibling of bob (parent of dave)",
        ),
        (
            '<tl_rule head="uncle(X, Y)" body="sibling(X, P), parent(P, Y)"></tl_rule>',
            "bob", "frank", True, "bob is sibling of carol (parent of frank)",
        ),
        (
            '<tl_rule head="uncle(X, Y)" body="sibling(X, P), parent(P, Y)"></tl_rule>',
            "alice", "dave", False, "alice is not sibling of bob, just parent",
        ),
        (
            '<tl_rule head="cousin(X, Y)" body="parent(P, X), sibling(P, Q), parent(Q, Y)"></tl_rule>',
            "dave", "frank", True, "dave's parent (bob) sibling carol → frank",
        ),
        (
            '<tl_rule head="cousin(X, Y)" body="parent(P, X), sibling(P, Q), parent(Q, Y)"></tl_rule>',
            "dave", "eve", False, "dave and eve share parent bob — not cousins",
        ),
        (
            '<tl_rule head="great_uncle(X, Y)" body="sibling(X, P), parent(P, GP), parent(GP, Y)"></tl_rule>',
            "alice", "dave", False, "alice has no sibling in this graph",
        ),
    ]

    print("exp65: multi-relation join tool-call extension")
    print("=" * 72)
    print()
    print(f"{'rule':<60}{'query':<28}{'expected':<10}{'got':<10}{'OK?':<6}")
    print("-" * 116)
    n_pass = 0
    for rule_str, subj, obj, expected, desc in cases:
        rule = parse_rule(rule_str)
        if rule is None:
            print(f"{'PARSE FAIL':<60}")
            continue
        head_name = rule[0][0]
        head_result, err = evaluate_rule(graph, rule)
        if err:
            print(f"{head_name:<60}{f'{subj}→{obj}':<28}{str(expected):<10}{'ERR':<10}{'FAIL':<6}  ({err})")
            continue
        got = query_relation(head_result, subj, obj)
        ok = (got == expected)
        n_pass += ok
        print(f"{head_name+': '+desc:<60}{f'{subj}→{obj}':<28}{str(expected):<10}{str(got):<10}{('PASS' if ok else 'FAIL'):<6}")
    print()
    print(f"Result: {n_pass}/{len(cases)} cases passed.")
    if n_pass == len(cases):
        print("Multi-relation join harness works. The TL tool-call protocol")
        print("now supports rule chains, not just single-relation closure.")
        print("The same einsum-chain machinery would handle any Datalog-class")
        print("rule expressed as binary-relation joins.")
    else:
        print("Some cases failed — check rule semantics.")


if __name__ == "__main__":
    main()
