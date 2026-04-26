"""
exp34 — Energy-based tensor logic with global constraints

HYPOTHESIS: Defining an energy function E(state) that includes (a) data-fit
terms and (b) global constraint terms (e.g. "Parent must be a tree"), and
minimizing it via gradient descent, gives better reconstruction on a KG with
conflicting facts than local sigmoid scoring.

FALSIFIED IF: Energy-based reconstruction shows F1 ≤ local sigmoid F1 + 0.05
on a 10-node parent-relation KG with intentionally conflicting facts.

SMALLEST TEST: 10 nodes, true parent tree (a → b → c, etc.), corrupt 30% of
edges with conflicting "X is parent of Y AND Y is parent of X" pairs. Compare:
(a) local sigmoid: max-likelihood per edge, no constraint.
(b) energy-based: data-fit + tree-violation penalty.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)

N = 10
# True parent relation (tree): 0→1, 0→2, 1→3, 1→4, 2→5, 2→6, 3→7, 3→8, 5→9
true_edges = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(5,9)]
P_true = torch.zeros(N, N)
for i, j in true_edges: P_true[i, j] = 1.0

# Corrupted observations: include all true edges + 4 conflicting reversed pairs
# (e.g., 1→0 contradicts 0→1; 5→2 contradicts 2→5)
observed_pos = list(true_edges) + [(1,0), (5,2), (4,1), (7,3)]
# (we have 9 true + 4 false-positive observations)

# Build observation logits: positive evidence for each observed edge, neutral elsewhere
def build_obs_logits():
    L = torch.full((N, N), -3.0)  # absent default = mildly absent
    for i, j in observed_pos:
        L[i, j] = +3.0  # observed = mildly present
    return L

OBS = build_obs_logits()


# ── Method A: local sigmoid scoring (no constraint) ──────────────────────────
def reconstruct_local():
    return (torch.sigmoid(OBS) >= 0.5).float()


# ── Method B: energy-based with tree constraint ──────────────────────────────
# Tree constraint: each node has at most one parent.
# Penalty: for each child y, sum_x P[x,y] should be ≤ 1.
# In differentiable form: penalty = sum_y max(0, sum_x P[x,y] - 1)^2
def reconstruct_energy(steps=1000, lr=0.05, lam=5.0):
    L = OBS.clone().requires_grad_(True)
    opt = torch.optim.Adam([L], lr=lr)
    for _ in range(steps):
        P = torch.sigmoid(L)
        # data-fit: BCE with observation logits as targets (high for observed, low otherwise)
        target = torch.zeros(N, N)
        for i, j in observed_pos: target[i, j] = 1.0
        data_loss = F.binary_cross_entropy(P, target)
        # constraint: each child has ≤ 1 parent
        col_sum = P.sum(dim=0)  # [N]
        tree_pen = torch.relu(col_sum - 1.0).pow(2).sum()
        # constraint: no self-loops
        diag_pen = torch.diag(P).sum()
        # constraint: no cycles of length 2 (i→j AND j→i)
        cycle_pen = (P * P.T).sum()
        loss = data_loss + lam * (tree_pen + cycle_pen) + lam * diag_pen
        opt.zero_grad(); loss.backward(); opt.step()
    return (torch.sigmoid(L) >= 0.5).float()


def f1_score(pred):
    tp = (pred * P_true).sum().item()
    fp = (pred * (1 - P_true)).sum().item()
    fn = ((1 - pred) * P_true).sum().item()
    prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
    return 2 * prec * rec / max(prec + rec, 1e-9), tp, fp, fn


print("exp34 — energy-based vs local sigmoid on conflicting KG\n")
print(f"True edges: {len(true_edges)}")
print(f"Observed: {len(observed_pos)} ({len(observed_pos)-len(true_edges)} are conflicting/false)\n")

local = reconstruct_local()
energy = reconstruct_energy()

f1_l, tp_l, fp_l, fn_l = f1_score(local)
f1_e, tp_e, fp_e, fn_e = f1_score(energy)

print(f"{'method':<25} {'TP':>3} {'FP':>3} {'FN':>3} {'F1':>6}")
print("-" * 50)
print(f"  local sigmoid           {tp_l:>3.0f} {fp_l:>3.0f} {fn_l:>3.0f} {f1_l:>6.3f}")
print(f"  energy + tree+cycle     {tp_e:>3.0f} {fp_e:>3.0f} {fn_e:>3.0f} {f1_e:>6.3f}")

print(f"\nHYPOTHESIS CHECK (energy F1 > local F1 + 0.05?):")
diff = f1_e - f1_l
print(f"  diff = {diff:+.3f}  →  {'CONFIRMED' if diff > 0.05 else 'FALSIFIED'}")
