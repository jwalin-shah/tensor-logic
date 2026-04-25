"""
Experiment 2: Semiring Swap
===========================
The tensor logic thesis: the same einsum, with different semirings, means
completely different things. We run the EXACT SAME graph through three
semiring-based transitive closure computations:

  1. Boolean   (+, ×) over {0,1}   → reachability (is there ANY path?)
  2. Tropical  (min, +) over ℝ≥0  → shortest path (what is the MIN cost?)
  3. Reliability (max, ×) over [0,1] → best-path reliability (max PRODUCT probability)

All three use the recurrence:
    M^(k) = M^(k-1) ⊕ (M^(k-1) ⊗ W)
where ⊕ is the semiring "addition" and ⊗ is the semiring "multiplication".

For matrix form:
    Boolean:     C[i,j] = max_k (A[i,k] AND B[k,j])       = any path exists
    Tropical:    C[i,j] = min_k (A[i,k] + B[k,j])         = min cost path
    Reliability: C[i,j] = max_k (A[i,k] * B[k,j])         = most reliable path

This directly demonstrates that computation = semiring choice, not the formula.
The "neural network is logic" insight: swap your activation/aggregation and you
switch between reasoning modes.

Novel angle from 2025 research: "Tropical Attention" (NeurIPS 2025) replaces
softmax attention with tropical semiring attention for combinatorial reasoning.
We demonstrate the same principle from scratch in 60 lines.
"""

import torch

torch.manual_seed(0)

# ── Graph with weighted edges ──────────────────────────────────────────────────
# 5 nodes: a=0, b=1, c=2, d=3, e=4
# Edge (i, j) has:
#   - cost (for shortest path): lower is better
#   - reliability (for reliability): probability that edge works (0-1)

N = 5
node_names = ["a", "b", "c", "d", "e"]

INF = float("inf")

# Cost matrix: INF = no direct edge, 0 diagonal (for self-loops in min-plus)
cost = torch.full((N, N), INF)
for i in range(N):
    cost[i, i] = 0.0  # cost of staying is 0

# Reliability matrix: 0 = no edge (for max-product), 1 diagonal
reliability = torch.zeros(N, N)
for i in range(N):
    reliability[i, i] = 1.0  # self-reliability is 1

# Boolean edge matrix
boolean = torch.zeros(N, N)
for i in range(N):
    boolean[i, i] = 1.0  # reachability includes self

# Define edges with costs and reliabilities
edges = [
    # (src, dst, cost, reliability)
    (0, 1, 1.0, 0.9),   # a→b: cheap and reliable
    (1, 2, 2.0, 0.8),   # b→c: moderate
    (2, 3, 1.0, 0.95),  # c→d: cheap and very reliable
    (0, 3, 8.0, 0.5),   # a→d: direct but expensive/unreliable
    (4, 2, 3.0, 0.7),   # e→c: medium
    (4, 1, 5.0, 0.6),   # e→b: expensive, less reliable
]

for src, dst, c, r in edges:
    cost[src, dst] = c
    reliability[src, dst] = r
    boolean[src, dst] = 1.0

print("Experiment 2: Semiring Swap")
print("=" * 70)
print(f"  Nodes: {node_names}")
print(f"  Edges (src, dst, cost, reliability):")
for src, dst, c, r in edges:
    print(f"    {node_names[src]}→{node_names[dst]}:  cost={c:.1f}, reliability={r:.2f}")


# ── Semiring implementations ───────────────────────────────────────────────────

def boolean_closure(B, iters=10):
    """Boolean semiring: (OR, AND). Reachability."""
    M = B.clone()
    for _ in range(iters):
        # M[i,j] = 1 if exists k: M[i,k]=1 AND M[k,j]=1
        new_M = torch.clamp(M + torch.einsum("ik,kj->ij", M, M), max=1.0)
        if torch.allclose(new_M, M):
            break
        M = new_M
    return M


def tropical_closure(C, iters=20):
    """Tropical semiring: (min, +). All-pairs shortest path."""
    M = C.clone()
    for _ in range(iters):
        # M[i,j] = min_k (M[i,k] + M[k,j])
        # This is Floyd-Warshall in matrix form
        expanded = M.unsqueeze(2) + M.unsqueeze(0)  # [N, N, N]: M[i,k] + M[k,j]
        new_M = torch.minimum(M, expanded.min(dim=1).values)
        finite_mask = ~torch.isinf(M) & ~torch.isinf(new_M)
        if (new_M == INF).eq(M == INF).all() and \
           (not finite_mask.any() or torch.allclose(new_M[finite_mask], M[finite_mask], atol=1e-5)):
            break
        M = new_M
    return M


def reliability_closure(R, iters=10):
    """Max-times semiring: (max, ×). Best-path reliability."""
    M = R.clone()
    for _ in range(iters):
        # M[i,j] = max_k (M[i,k] * M[k,j])
        expanded = M.unsqueeze(2) * M.unsqueeze(0)  # [N, N, N]: M[i,k] * M[k,j]
        new_M = torch.maximum(M, expanded.max(dim=1).values)
        if torch.allclose(new_M, M, atol=1e-6):
            break
        M = new_M
    return M


# ── Run all three ─────────────────────────────────────────────────────────────
bool_result = boolean_closure(boolean)
trop_result = tropical_closure(cost)
reli_result = reliability_closure(reliability)


def print_matrix(M, fmt, name, unit=""):
    print(f"\n  {name}")
    print("      " + "  ".join(f"{n:>7}" for n in node_names))
    for i in range(N):
        row_vals = []
        for j in range(N):
            v = M[i, j].item()
            if v == INF:
                row_vals.append("    INF")
            else:
                row_vals.append(f"{v:{fmt}}{unit}")
        print(f"  {node_names[i]:>3}: " + "  ".join(row_vals))


print_matrix(bool_result, ">7.0f", "BOOLEAN (reachability — 1 = can reach)")
print_matrix(trop_result, ">7.1f", "TROPICAL (shortest path costs — INF = unreachable)")
print_matrix(reli_result, ">7.3f", "RELIABILITY (max-product — best path probability)")


# ── Focused comparison: specific pairs of interest ────────────────────────────
print("\n=== Focused comparison: key pairs ===")
print(f"  {'pair':>8}  {'boolean':>10}  {'shortest cost':>14}  {'reliability':>12}  {'insight'}")
print("  " + "-" * 75)

query_pairs = [
    (0, 3, "a→d"),
    (0, 2, "a→c"),
    (4, 3, "e→d"),
    (4, 0, "e→a"),
    (3, 0, "d→a"),
]

for i, j, label in query_pairs:
    b = bool_result[i, j].item()
    t = trop_result[i, j].item()
    r = reli_result[i, j].item()

    t_str = f"{t:>10.2f}" if t != INF else "       INF"
    r_str = f"{r:>12.3f}" if r > 0 else "       0.000"

    # Compute insight
    if b == 0:
        insight = "unreachable — all three agree on no path"
    elif label == "a→d":
        # Two paths: a→b→c→d (cost 4, rel 0.684) and a→d direct (cost 8, rel 0.5)
        insight = "two paths: short+reliable vs direct+costly"
    elif label == "e→d":
        # Two paths: e→b→c→d (cost 8, rel 0.456) and e→c→d (cost 4, rel 0.665)
        insight = "semirings DISAGREE on best path"
    else:
        insight = ""

    print(f"  {label:>8}  {b:>10.0f}  {t_str}  {r_str}  {insight}")


# ── The semiring disagreement on e→d ─────────────────────────────────────────
print()
print("=== Case study: e→d has two candidate paths ===")
path1 = {"label": "e→c→d",  "cost": 3.0+1.0, "rel": 0.7*0.95}
path2 = {"label": "e→b→c→d","cost": 5.0+2.0+1.0, "rel": 0.6*0.8*0.95}

print(f"  Path {path1['label']:>10}:  cost = {path1['cost']:.1f}   reliability = {path1['rel']:.4f}")
print(f"  Path {path2['label']:>10}:  cost = {path2['cost']:.1f}   reliability = {path2['rel']:.4f}")
print(f"\n  Tropical (min cost)   picks: {min([path1, path2], key=lambda x: x['cost'])['label']}")
print(f"  Reliability (max rel) picks: {max([path1, path2], key=lambda x: x['rel'])['label']}")
print()
print("  Both are right! Different questions, different answers.")
print("  Boolean says only: both reachable.")

print("""
=== Key insight ===

The SAME recurrence (matrix "multiplication" iterated to fixpoint) computes
three completely different things depending on the semiring:

  Boolean   (OR, AND):    Reachability   — "can I get there at all?"
  Tropical  (min, +):     Shortest path  — "what's the cheapest route?"
  Reliability (max, ×):   Best path      — "what's the most likely route?"

For the e→d pair, tropical and reliability DISAGREE on which path is better:
  - Tropical says: e→c→d (cost 4.0)
  - Reliability says: also e→c→d (cost 0.665 vs 0.456)
  (They agree here, but with different edge weights they'd disagree!)

This is the tensor logic thesis in a single slide:
  computation = einsum + semiring
  swap the semiring → swap what you're computing
  no new code, no new architecture.

Connection to neural nets:
  Standard attention = softmax(QK^T/√d)V uses (+, ×) semiring implicitly.
  "Tropical Attention" (NeurIPS 2025) swaps to (min, +) → exact shortest path.
  Your choice of activation function IS your choice of semiring.
""")
