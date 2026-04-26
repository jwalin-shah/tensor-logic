"""
exp60b (step 2/4 of the TL-as-tool integration line):
tool-call interception harness.

Given a string that may contain `<tl_closure relation="..." query="..."
subject="..." object="...">` tags, this module:
  1. parses the tag,
  2. evaluates the query against the relation (using TL transitive
     closure, exp44/53 machinery),
  3. returns a structured result the caller can splice back into the
     model's context (or compare to a gold answer).

This step has no LM dependency. The harness is pure plumbing — it would
be called by an LM-serving loop in exp60d, but is fully testable on its
own with hand-written or rule-emitted tool calls.

Why a tag-based protocol (vs OpenAI-style function call JSON)?
  - Easy to parse with a small regex; no JSON schema validation required.
  - Easy to constrain a small instruct LM to emit, including with
    grammar-constrained decoding (Outlines / Instructor).
  - Maps cleanly to the OPENHUMAN_TL_MEMO architecture: <tl_closure>
    is one of a small set of tags an SLM is allowed to emit.

Returns (per call): {"answer": "yes"|"no", "trace": [...]} where trace is
a derivation chain (list of (step, fact) pairs) for auditability — the
TL substrate's natural provenance, per OPENHUMAN_TL_MEMO §4.
"""

import re
from typing import Any

import torch

TAG_RE = re.compile(
    r'<tl_closure\s+relation="(?P<rel>[^"]+)"\s+'
    r'query="(?P<q>[^"]+)"\s+'
    r'subject="(?P<subj>[^"]+)"\s+'
    r'object="(?P<obj>[^"]+)"\s*>\s*</tl_closure>'
)


def parse_tool_call(s: str):
    """Return list of dicts for every <tl_closure ...> tag in `s`."""
    return [
        {
            "relation": m.group("rel"),
            "query": m.group("q"),
            "subject": m.group("subj"),
            "object": m.group("obj"),
        }
        for m in TAG_RE.finditer(s)
    ]


def closure_matrix(adjacency: torch.Tensor, max_iters: int = 20) -> torch.Tensor:
    """Boolean transitive closure via TL fixpoint:
       R ← ((R @ A + R) > 0) until fixpoint.
    """
    n = adjacency.shape[0]
    R = adjacency.clone()
    for _ in range(max_iters):
        R_new = ((R @ adjacency + R) > 0).float()
        if torch.equal(R_new, R):
            break
        R = R_new
    return R


def derivation_chain(adjacency: torch.Tensor, names, src_idx, tgt_idx, max_hops=10):
    """Reconstruct a single chain src → ... → tgt for provenance.

    Pure BFS using the adjacency matrix; not the closure tensor itself.
    Returns list of names along the path, or [] if no path.
    """
    n = adjacency.shape[0]
    visited = {src_idx: None}
    queue = [src_idx]
    while queue:
        u = queue.pop(0)
        if u == tgt_idx:
            # reconstruct path
            chain = [tgt_idx]
            while visited[chain[-1]] is not None:
                chain.append(visited[chain[-1]])
            return [names[i] for i in reversed(chain)]
        for v in range(n):
            if adjacency[u, v] > 0 and v not in visited:
                visited[v] = u
                queue.append(v)
    return []


def build_adjacency(graph_relation, names):
    """Build adjacency matrix from a list of [src, dst] pairs."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    A = torch.zeros(n, n)
    for src, dst in graph_relation:
        if src in name_to_idx and dst in name_to_idx:
            A[name_to_idx[src], name_to_idx[dst]] = 1.0
    return A


def evaluate_call(graph: dict, call: dict) -> dict:
    """Execute one tool call against the graph. Return answer + trace."""
    rel_name = call["relation"]
    if rel_name not in graph:
        return {"answer": "no", "trace": [], "error": f"unknown relation: {rel_name}"}
    adjacency_pairs = graph[rel_name]
    # Collect all names mentioned in the relation OR the query
    all_names = sorted({n for pair in adjacency_pairs for n in pair} |
                       {call["subject"], call["object"]})
    A = build_adjacency(adjacency_pairs, all_names)
    R = closure_matrix(A)

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    if call["subject"] not in name_to_idx or call["object"] not in name_to_idx:
        return {"answer": "no", "trace": []}

    subj = name_to_idx[call["subject"]]
    obj = name_to_idx[call["object"]]

    # query semantics:
    #  - "ancestor": subject ancestor of object → R[subj, obj] (parent chain)
    #  - "descendant": subject descendant of object → R[obj, subj]
    if call["query"] == "ancestor":
        related = bool(R[subj, obj] > 0)
        if related:
            chain = derivation_chain(A, all_names, subj, obj)
        else:
            chain = []
    elif call["query"] == "descendant":
        related = bool(R[obj, subj] > 0)
        if related:
            chain = derivation_chain(A, all_names, obj, subj)
        else:
            chain = []
    else:
        return {"answer": "no", "trace": [], "error": f"unknown query: {call['query']}"}

    return {
        "answer": "yes" if related else "no",
        "trace": chain,
    }


def evaluate_string(graph: dict, response_text: str) -> list:
    """Find every <tl_closure ...> in response_text, evaluate each, return
    list of {"call": ..., "result": ...} dicts."""
    calls = parse_tool_call(response_text)
    return [{"call": c, "result": evaluate_call(graph, c)} for c in calls]


def main():
    """Smoke test: hand-write a few tool calls + verify against ground truth."""
    graph = {
        "parent": [
            ["alice", "bob"],
            ["bob", "carol"],
            ["carol", "dave"],
            ["dave", "eve"],
        ],
    }
    cases = [
        # (call_string, expected)
        ('<tl_closure relation="parent" query="ancestor" subject="alice" object="dave"></tl_closure>', "yes"),
        ('<tl_closure relation="parent" query="ancestor" subject="alice" object="eve"></tl_closure>', "yes"),
        ('<tl_closure relation="parent" query="descendant" subject="eve" object="alice"></tl_closure>', "yes"),
        ('<tl_closure relation="parent" query="ancestor" subject="dave" object="alice"></tl_closure>', "no"),
        ('<tl_closure relation="parent" query="descendant" subject="alice" object="eve"></tl_closure>', "no"),
    ]
    print("exp60b: tool-call harness smoke test")
    print()
    print(f"{'expected':<10}{'got':<10}{'trace':<40}{'OK?':<5}")
    print("-" * 70)
    n_pass = 0
    for call_str, expected in cases:
        results = evaluate_string(graph, call_str)
        got = results[0]["result"]["answer"]
        trace = " → ".join(results[0]["result"]["trace"]) or "—"
        ok = (got == expected)
        n_pass += ok
        print(f"{expected:<10}{got:<10}{trace:<40}{'PASS' if ok else 'FAIL':<5}")
    print()
    print(f"Result: {n_pass}/{len(cases)} cases passed.")
    if n_pass == len(cases):
        print("Tool-call harness is wired correctly. Ready for downstream use in exp60c/d.")


if __name__ == "__main__":
    main()
