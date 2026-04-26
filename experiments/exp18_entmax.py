"""
Experiment 18: Entmax-α — The Goldilocks Fix
=============================================
Problem recap:
  exp1/exp9: sigmoid floor — sigmoid(0)=0.5, hallucinations locked at 0.41
  exp14: sparsemax fixes floor BUT concentration kills multi-destination rows
          (simplex forces all mass to one destination per row → coverage=0.50)

Entmax-α (Peters et al., 2019):
  α=1.0  →  sigmoid (independent per-entry, floor=0.5)
  α=1.5  →  entmax-1.5 (sparse, CAN output 0, no simplex concentration)
  α=2.0  →  sparsemax (hard simplex projection, one winner per row)

Why α=1.5 is the Goldilocks point:
  - CAN output exact 0.0 (no floor problem)
  - Does NOT force all mass to one destination (multiple true edges possible)
  - Differentiable (gradient exists)
  - Feb 2026: used in differentiable symbolic planning, +26 accuracy over sigmoid

Key test: node with 4 true outgoing edges (a has children c,d,e,f)
  sigmoid: all 4 get ~0.5 even without evidence (floor)
  sparsemax: at most 1 gets nonzero mass per row (concentration)
  entmax-1.5: all 4 can get nonzero mass, zero for non-edges ✓
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ── Entmax-1.5 implementation ─────────────────────────────────────────────────
def entmax15(z, dim=-1, n_iter=50):
    """
    Entmax with α=1.5. Uses bisection on the dual variable.
    Peters et al., 2019. More efficient than general entmax-α.
    Output: sparse probability distribution, can have exact zeros,
            does NOT force all mass to one entry.
    """
    z = z.float()
    # Shift for numerical stability
    z = z - z.max(dim=dim, keepdim=True).values

    # Entmax-1.5: p_i = max(0, τ - z_i)^(-1) ... actually:
    # p_i = max(0, (z_i - τ) / 1) via the 1.5-entmax formula
    # The exact formula: p_i ∝ max(z_i - τ, 0)^(1/(α-1)) = max(z_i - τ, 0)^2
    # for α=1.5 → exponent = 1/(1.5-1) = 2

    def _p_from_tau(tau):
        return torch.clamp(z - tau, min=0.0) ** 2

    def _sum_from_tau(tau):
        return _p_from_tau(tau).sum(dim=dim, keepdim=True)

    # Bisection to find τ such that sum(p) = 1
    tau_lo = (z.max(dim=dim, keepdim=True).values - 1.0)
    tau_hi = z.max(dim=dim, keepdim=True).values - (1.0 / z.shape[dim]) ** 0.5

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) / 2.0
        s = _sum_from_tau(tau_mid)
        tau_lo = torch.where(s > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(s <= 1.0, tau_mid, tau_hi)

    tau = (tau_lo + tau_hi) / 2.0
    p = _p_from_tau(tau)
    # Renormalize to exactly sum to 1 (bisection may not be perfect)
    p = p / p.sum(dim=dim, keepdim=True).clamp(min=1e-9)
    return p


# ── Test graph: a has 4 children, b has 1 child, e is isolated ───────────────
N = 7
node_names = ["a","b","c","d","e","f","g"]
# a→c, a→d, a→e, a→f  (a has 4 children)
# b→g                   (b has 1 child)
# (e, f, g are leaves)
edge_pairs = [(0,2),(0,3),(0,4),(0,5), (1,6)]
E = torch.zeros(N,N)
for i,j in edge_pairs: E[i,j] = 1.0

# True closure: a→c,d,e,f; b→g; no transitive paths (depth=1 only)
true_closure = set(edge_pairs)
TC_true = torch.zeros(N,N)
for i,j in true_closure: TC_true[i,j] = 1.0


def run_fixpoint_sigmoid(T, iters=60):
    if T < 1e-6:
        P = E.clone()
        for _ in range(iters):
            new = ((E + torch.einsum("xy,yz->xz", P, E)) > 0).float()
            if torch.allclose(new, P): break
            P = new
        return P
    P = E.clone()
    for _ in range(iters):
        logits = E + torch.einsum("xy,yz->xz", P, E)
        new = torch.sigmoid(logits / T)
        if torch.allclose(new, P, atol=1e-5): break
        P = new
    return P


def run_fixpoint_sparsemax(scale, iters=60):
    from exp14_sparsemax import sparsemax
    P = E.clone()
    for _ in range(iters):
        logits = (E + torch.einsum("xy,yz->xz", P, E)) * scale
        new = sparsemax(logits, dim=-1)
        if torch.allclose(new, P, atol=1e-5): break
        P = new
    return P


def run_fixpoint_entmax15(scale, iters=60):
    P = E.clone()
    for _ in range(iters):
        logits = (E + torch.einsum("xy,yz->xz", P, E)) * scale
        new = entmax15(logits, dim=-1)
        if torch.allclose(new, P, atol=1e-5): break
        P = new
    return P


def metrics(P, threshold=0.5):
    pred = (P >= threshold).float()
    tp  = (pred * TC_true).sum().item()
    fp  = (pred * (1-TC_true)).sum().item()
    fn  = ((1-pred) * TC_true).sum().item()
    pr  = tp / max(tp+fp, 1e-9)
    re  = tp / max(tp+fn, 1e-9)
    f1  = 2*pr*re / max(pr+re, 1e-9)
    nonzero_rate = (P > 1e-6).float().mean().item()
    true_zeros = (P[TC_true==0] < 1e-6).sum().item()
    total_neg  = int((TC_true==0).sum())
    return pr, re, f1, nonzero_rate, true_zeros, total_neg


print("Experiment 18: Entmax-1.5 — Goldilocks Fix for Tensor Logic")
print("=" * 65)
print(f"  Graph: {N} nodes, edges {edge_pairs}")
print(f"  Key challenge: node 'a' has 4 true outgoing edges")
print(f"  sigmoid floor problem + sparsemax concentration problem")
print(f"  True closure: {len(true_closure)} pairs\n")


# ── Main comparison ───────────────────────────────────────────────────────────
print("  Activation comparison at threshold=0.5:")
print(f"  {'Method':<35}  {'P':>5}  {'R':>5}  {'F1':>5}  {'density':>8}  {'true-zeros':>12}")
print("  " + "-" * 80)

# Step (ground truth)
P_step = run_fixpoint_sigmoid(0.0)
pr, re, f1, dens, tz, tot = metrics(P_step, 0.5)
print(f"  {'Step (T→0, ground truth)':<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {dens:>7.2%}  {tz}/{tot}")

# Sigmoid variants
for T in [1.0, 0.5, 0.1]:
    P = run_fixpoint_sigmoid(T)
    pr, re, f1, dens, tz, tot = metrics(P, 0.5)
    print(f"  {'sigmoid T='+str(T):<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {dens:>7.2%}  {tz}/{tot}")

print()
# Sparsemax variants
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
for scale in [1.0, 5.0, 10.0]:
    try:
        P = run_fixpoint_sparsemax(scale)
        pr, re, f1, dens, tz, tot = metrics(P, 0.5)
        print(f"  {'sparsemax scale='+str(scale):<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {dens:>7.2%}  {tz}/{tot}")
    except Exception as ex:
        print(f"  sparsemax scale={scale}: {ex}")

print()
# Entmax-1.5 variants
for scale in [1.0, 2.0, 5.0, 10.0]:
    P = run_fixpoint_entmax15(scale)
    pr, re, f1, dens, tz, tot = metrics(P, 0.5)
    print(f"  {'entmax-1.5 scale='+str(scale):<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {dens:>7.2%}  {tz}/{tot}")


# ── Show output matrix for best methods ──────────────────────────────────────
print()
print("  Output matrix — row 'a' (has 4 true children: c,d,e,f):")
print("  " + "  ".join(f"{n:>7}" for n in node_names))
print("  True:     " + "  ".join(f"{'1' if E[0,j]>0 else '0':>7}" for j in range(N)))

P_sig05 = run_fixpoint_sigmoid(0.5)
print("  sig T=0.5:" + "  ".join(f"{P_sig05[0,j].item():>7.3f}" for j in range(N)))

try:
    P_smax  = run_fixpoint_sparsemax(5.0)
    print("  smax s=5: " + "  ".join(f"{P_smax[0,j].item():>7.3f}" for j in range(N)))
except:
    pass

P_e15_5 = run_fixpoint_entmax15(5.0)
print("  entmax5:  " + "  ".join(f"{P_e15_5[0,j].item():>7.3f}" for j in range(N)))

P_e15_2 = run_fixpoint_entmax15(2.0)
print("  entmax2:  " + "  ".join(f"{P_e15_2[0,j].item():>7.3f}" for j in range(N)))


# ── The three problems: floor, concentration, true-zero ──────────────────────
print()
print("  Diagnostic: floor / concentration / true-zero ability")
print(f"  {'Method':<35}  {'floor (no-edge min)':>20}  {'conc. (max mass 1 row)':>22}  {'exact-0 count':>14}")
print("  " + "-" * 95)

def diagnose(P, label):
    no_edge_vals = P[TC_true == 0]
    floor_min = no_edge_vals.min().item()
    floor_mean = no_edge_vals.mean().item()
    # Concentration: for each row, what fraction of mass on top entry?
    row_max = P.max(dim=-1).values
    row_sum = P.sum(dim=-1).clamp(min=1e-9)
    conc = (row_max / row_sum).mean().item()
    exact_zeros = (no_edge_vals < 1e-6).sum().item()
    print(f"  {label:<35}  min={floor_min:.4f} mean={floor_mean:.4f}   {conc:>15.3f}         {exact_zeros}/{len(no_edge_vals)}")

diagnose(P_sig05, "sigmoid T=0.5")
try:
    diagnose(P_smax, "sparsemax scale=5")
except:
    pass
diagnose(P_e15_5, "entmax-1.5 scale=5")
diagnose(P_e15_2, "entmax-1.5 scale=2")
diagnose(P_step,  "step (ground truth)")


# ── Sweep α to show the transition ───────────────────────────────────────────
print()
print("  Effect of α on entmax (from sigmoid to sparsemax):")
print("  (using simple single-step application, not fixpoint)")
print(f"  {'α':<8}  {'floor_min':>10}  {'coverage':>10}  {'true-zeros':>12}  {'description'}")
print("  " + "-" * 65)

# For this demo, apply entmax directly to E (logits = E * 5.0)
logits = E * 5.0
for alpha_name, P_row in [
    ("sigmoid", torch.sigmoid(logits)),
    ("entmax α≈1.2", None),   # not implemented, skip
    ("entmax-1.5", entmax15(logits, dim=-1)),
    ("sparsemax", None),
]:
    if P_row is None:
        continue
    no_edge = P_row[TC_true == 0]
    floor_min = no_edge.min().item()
    cov = (P_row[TC_true == 1] >= 0.5).float().mean().item()
    tz  = (no_edge < 1e-6).sum().item()
    desc = ("has floor" if floor_min > 0.1 else "near-0 possible") + \
           (", concentrated" if P_row.max(dim=-1).values.mean() > 0.7 else "")
    print(f"  {alpha_name:<8}  {floor_min:>10.4f}  {cov:>10.3f}  {tz:>5}/{len(no_edge)}      {desc}")


# ── Entmax-1.5 on the original exp1 graph ────────────────────────────────────
print()
print("=" * 65)
print("  Revisiting exp1 graph with entmax-1.5 (a→b→c→d, e→c):")
N2 = 5
node_names2 = ["a","b","c","d","e"]
edge_pairs2 = [(0,1),(1,2),(2,3),(4,2)]
E2 = torch.zeros(N2,N2)
for i,j in edge_pairs2: E2[i,j] = 1.0
true_closure2 = {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(4,2),(4,3)}
TC2 = torch.zeros(N2,N2)
for i,j in true_closure2: TC2[i,j] = 1.0

def run_e15_exp1(scale, iters=60):
    P = E2.clone()
    for _ in range(iters):
        logits = (E2 + torch.einsum("xy,yz->xz", P, E2)) * scale
        new = entmax15(logits, dim=-1)
        if torch.allclose(new, P, atol=1e-5): break
        P = new
    return P

def metrics2(P, threshold=0.5):
    pred = (P >= threshold).float()
    tp  = (pred * TC2).sum().item()
    fp  = (pred * (1-TC2)).sum().item()
    fn  = ((1-pred) * TC2).sum().item()
    pr  = tp / max(tp+fp, 1e-9)
    re  = tp / max(tp+fn, 1e-9)
    f1  = 2*pr*re / max(pr+re, 1e-9)
    tz  = (P[TC2==0] < 1e-6).sum().item()
    return pr, re, f1, tz

print(f"\n  True closure: {len(true_closure2)} pairs")
print(f"  {'Method':<35}  {'P':>5}  {'R':>5}  {'F1':>5}  {'true-zeros':>12}")
print("  " + "-" * 65)

# Step
P_step2 = E2.clone()
for _ in range(30):
    new = ((E2 + torch.einsum("xy,yz->xz", P_step2, E2)) > 0).float()
    if torch.allclose(new, P_step2): break
    P_step2 = new
pr, re, f1, tz = metrics2(P_step2)
print(f"  {'step (ground truth)':<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {tz}/{int((TC2==0).sum())}")

# Sigmoid
P_sig_exp1 = E2.clone()
for _ in range(60):
    logits = E2 + torch.einsum("xy,yz->xz", P_sig_exp1, E2)
    new = torch.sigmoid(logits / 0.5)
    if torch.allclose(new, P_sig_exp1, atol=1e-5): break
    P_sig_exp1 = new
pr, re, f1, tz = metrics2(P_sig_exp1)
print(f"  {'sigmoid T=0.5 (exp1 broken)':<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {tz}/{int((TC2==0).sum())}")

for scale in [2.0, 5.0, 10.0]:
    P = run_e15_exp1(scale)
    pr, re, f1, tz = metrics2(P)
    print(f"  {'entmax-1.5 scale='+str(scale):<35}  {pr:.3f}  {re:.3f}  {f1:.3f}  {tz}/{int((TC2==0).sum())}")


print("""
=== Key Insights ===

1. Entmax-1.5 DOES fix the floor problem:
   sigmoid floor: 0.5 on all non-edge entries → 0/17 true zeros
   entmax-1.5:    0.0 on non-edge entries    → 15-16/17 true zeros
   This is a genuine improvement: absent evidence is now representable.

2. Entmax-1.5 does NOT fully solve the multi-destination threshold problem.
   When a node has N true destinations with equal evidence, each gets ≈1/N.
   With 4 true children, each gets 0.25 — below the 0.5 threshold.
   Result: recall=0.200 for 4-child node, identical to sparsemax.
   The normalization constraint is unavoidable for any row-normalized activation.

3. The actual tradeoff revealed by the data:
   sigmoid:      floor=0.5, recall=1.0, precision=0.102, F1=0.185
   sparsemax:    floor=0.0, recall=0.200, precision=1.000, F1=0.333
   entmax-1.5:   floor=0.0, recall=0.200, precision=1.000, F1=0.333 (= sparsemax here)
   step (T→0):   floor=0.0, recall=1.000, precision=1.000, F1=1.000
   The ONLY method that achieves F1=1.000 is the step function.

4. Why normalized activations fail for logic:
   Logical predicates are INDEPENDENT per pair. Whether (a,c) is true is
   independent of whether (a,d) is true. Row normalization imposes a zero-sum
   competition that doesn't match the semantics. Sigmoid is independent (good)
   but has a floor (bad). Step is independent and has no floor (best for inference).

5. The correct architecture split:
   INFERENCE:  use step function (T→0). It's exact, it's fast, it's binary.
   LEARNING:   use sigmoid with T-annealing — smooth gradients for training,
               snap to step at inference. This is the lesson from exp9+exp14+exp18.
   The floor problem at training time is real but manageable (use threshold ≥ 0.6).

6. Why entmax-1.5 still matters despite not improving F1 here:
   - For MULTI-CLASS outputs (which class is entity X?), normalization is correct.
   - For ATTENTION mechanisms in transformer-style rule application, entmax-1.5
     achieves sparse attention (some inputs contribute exactly zero), which matters.
   - The Feb 2026 +26 accuracy result was for CONSTRAINT SELECTION (pick which
     rule to apply), not multi-label edge prediction. Different task structure.
""")
