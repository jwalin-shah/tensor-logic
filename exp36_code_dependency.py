"""
exp36: code-dependency reasoning with GPT-2.

Three-way comparison on synthetic call-graph transitive-closure queries:
  (A) LM alone — no context
  (B) LM + raw facts — all calls(a,b) lines in prompt
  (C) LM + tensor-logic tool — transitive closure pre-computed, injected

Scoring: log P("Yes") vs log P("No") on each query (GPT-2 is base, not chat-tuned).
Metric: accuracy by hop depth.
"""

import random
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SEED = 0
N_FUNCS = 20
N_QUERIES_PER_DEPTH = 40
DEPTHS = [1, 2, 3, 4]
EDGE_PROB = 0.18  # sparse-ish DAG

random.seed(SEED)
torch.manual_seed(SEED)


def build_dag(n: int, p: float):
    names = [f"f{i:02d}" for i in range(n)]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):  # DAG: only forward edges (i -> j, i<j)
            if random.random() < p:
                edges.add((names[i], names[j]))
    return names, edges


def transitive_closure(names, edges):
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    M = torch.zeros(n, n)
    for a, b in edges:
        M[idx[a], idx[b]] = 1.0
    # boolean transitive closure via repeated squaring
    R = M.clone()
    for _ in range(n):
        R_new = ((R @ R + R) > 0).float()
        if torch.equal(R_new, R):
            break
        R = R_new
    return R, idx


def shortest_hop(R_step, idx, a, b, max_hops=8):
    """Return min hop count from a to b, or 0 if unreachable. R_step = adjacency."""
    n = R_step.shape[0]
    reach = torch.zeros(n)
    reach[idx[a]] = 1.0
    power = R_step.clone()
    for k in range(1, max_hops + 1):
        if power[idx[a], idx[b]] > 0:
            return k
        power = ((power @ R_step) > 0).float()
    return 0


def make_queries(names, edges, R, idx, depths, n_per_depth):
    """Sample yes/no queries balanced per hop depth."""
    n = len(names)
    M = torch.zeros(n, n)
    for a, b in edges:
        M[idx[a], idx[b]] = 1.0

    by_depth = {d: [] for d in depths}
    no_pool = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = names[i], names[j]
            if R[i, j] > 0:
                d = shortest_hop(M, idx, a, b)
                if d in by_depth:
                    by_depth[d].append((a, b, True))
            else:
                no_pool.append((a, b, False))

    queries = []
    rng = random.Random(SEED + 1)
    for d in depths:
        pos = by_depth[d]
        rng.shuffle(pos)
        rng.shuffle(no_pool)
        k = min(n_per_depth // 2, len(pos), len(no_pool))
        for q in pos[:k]:
            queries.append((d, q))
        for q in no_pool[:k]:
            queries.append((d, q))
    return queries


def facts_text(edges):
    return "\n".join(f"{a} calls {b}." for a, b in sorted(edges))


def closure_text(name, R, names, idx):
    i = idx[name]
    reach = [names[j] for j in range(len(names)) if R[i, j] > 0]
    if not reach:
        return f"{name} transitively calls: (nothing)."
    return f"{name} transitively calls: {', '.join(reach)}."


def query_text(a, b):
    return f"Question: does {a} (directly or indirectly) call {b}? Answer (Yes or No):"


@torch.no_grad()
def score_yes_no(model, tok, prompt, device):
    """Return (logP(' Yes'), logP(' No'))."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    logits = model(ids).logits[0, -1]
    logp = F.log_softmax(logits, dim=-1)
    return logp[yes_id].item(), logp[no_id].item()


def evaluate(model, tok, device, queries, build_prompt):
    correct_by_depth = {}
    total_by_depth = {}
    for depth, (a, b, label) in queries:
        prompt = build_prompt(a, b)
        lp_yes, lp_no = score_yes_no(model, tok, prompt, device)
        pred = lp_yes > lp_no
        ok = (pred == label)
        correct_by_depth[depth] = correct_by_depth.get(depth, 0) + int(ok)
        total_by_depth[depth] = total_by_depth.get(depth, 0) + 1
    return {d: correct_by_depth[d] / total_by_depth[d] for d in total_by_depth}


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    names, edges = build_dag(N_FUNCS, EDGE_PROB)
    print(f"DAG: {len(names)} nodes, {len(edges)} edges")
    R, idx = transitive_closure(names, edges)
    print(f"transitive closure density: {R.sum().item()/(len(names)**2):.3f}")

    queries = make_queries(names, edges, R, idx, DEPTHS, N_QUERIES_PER_DEPTH)
    print(f"queries: {len(queries)} total, by depth: { {d: sum(1 for dd,_ in queries if dd==d) for d in DEPTHS} }")

    print("loading GPT-2…")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    facts = facts_text(edges)

    def prompt_alone(a, b):
        return query_text(a, b)

    def prompt_facts(a, b):
        return f"Call graph:\n{facts}\n\n{query_text(a, b)}"

    def prompt_tool(a, b):
        return f"Call graph:\n{facts}\n\nTool output: {closure_text(a, R, names, idx)}\n\n{query_text(a, b)}"

    print("\n=== A: LM alone ===")
    res_a = evaluate(model, tok, device, queries, prompt_alone)
    print(res_a)

    print("\n=== B: LM + raw facts ===")
    res_b = evaluate(model, tok, device, queries, prompt_facts)
    print(res_b)

    print("\n=== C: LM + tensor-logic tool ===")
    res_c = evaluate(model, tok, device, queries, prompt_tool)
    print(res_c)

    print("\n=== summary (accuracy by hop depth) ===")
    print(f"{'depth':<8}{'A:alone':<12}{'B:facts':<12}{'C:tool':<12}")
    for d in DEPTHS:
        print(f"{d:<8}{res_a.get(d,0):<12.3f}{res_b.get(d,0):<12.3f}{res_c.get(d,0):<12.3f}")
    overall = lambda r: sum(r.values()) / len(r)
    print(f"{'avg':<8}{overall(res_a):<12.3f}{overall(res_b):<12.3f}{overall(res_c):<12.3f}")


if __name__ == "__main__":
    main()
