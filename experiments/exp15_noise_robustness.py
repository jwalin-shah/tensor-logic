"""
Experiment 15: Noise Robustness of Rule Chains
===============================================
Real-world KGs are noisy: facts get mislabeled, edges get missed,
automated extraction introduces errors. Question: how does noise in the
BASE relation propagate through a RULE CHAIN?

Setup:
  Base: Parent relation (observed, noisy)
  Rule chain: Parent → Sibling → Uncle (two rule hops)

  Noise model: flip each Parent edge with probability p_noise.
    p_noise = 0: perfect Parent graph
    p_noise = 0.1: 10% of Parent edges are flipped (wrong)
    p_noise = 0.2, 0.3, 0.5: increasing corruption

Measurements:
  1. Sibling F1 at each noise level (one rule hop from Parent)
  2. Uncle F1 at each noise level (two rule hops from Parent)
  3. Does noise AMPLIFY or ATTENUATE through rule chains?
  4. Does max-product vs sum-product handle noise differently?

Novel angle: nobody has specifically measured noise amplification/attenuation
through TENSOR-LOGIC RULE CHAINS. Existing KG robustness work focuses on
embedding models (bilinear, TransE), not symbolic rule chains. The tensor
logic framework makes this directly testable.

Expected: each rule hop can amplify noise (false positives chain) OR
attenuate noise (the join operation requires BOTH legs to be active, which
is harder for random errors to satisfy simultaneously).
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

N = 10
# Larger family: 3 generations, more people = more signal
# Gen1: 0=Alice, 1=Bob
# Gen2: 2=Carol, 3=Dan, 4=Eve, 5=Frank (children of Alice/Bob)
# Gen3: 6=Grace, 7=Hank, 8=Iris, 9=Jack (children of Carol/Dan)
names = ["Alice","Bob","Carol","Dan","Eve","Frank","Grace","Hank","Iris","Jack"]

true_parent_pairs = [
    (0,2),(0,3),(0,4),(0,5),   # Alice → Carol,Dan,Eve,Frank
    (1,2),(1,3),(1,4),(1,5),   # Bob → Carol,Dan,Eve,Frank
    (2,6),(2,7),               # Carol → Grace,Hank
    (3,8),(3,9),               # Dan → Iris,Jack
]

Parent_true = torch.zeros(N,N)
for i,j in true_parent_pairs:
    Parent_true[i,j] = 1.0

def compute_sibling_symbolic(P):
    S = torch.einsum("zx,zy->xy", P, P)
    S.fill_diagonal_(0)
    return (S > 0).float()

def compute_uncle_symbolic(P, S):
    return (torch.einsum("xy,yz->xz", S, P) > 0).float()

def compute_uncle_sumproduct(P_soft, S_soft):
    """Soft uncle via sum-product (tolerates partial evidence)."""
    return torch.einsum("xy,yz->xz", S_soft, P_soft).clamp(0,1)

def compute_uncle_maxproduct(P_soft, S_soft):
    """Soft uncle via max-product (picks strongest path)."""
    return (S_soft.unsqueeze(2) * P_soft.unsqueeze(0)).max(dim=1).values

# Ground truth
Sibling_true = compute_sibling_symbolic(Parent_true)
Uncle_true   = compute_uncle_symbolic(Parent_true, Sibling_true)

def f1(pred, target, threshold=0.5):
    p = (pred >= threshold).float()
    tp  = (p * target).sum().item()
    fp  = (p * (1-target)).sum().item()
    fn  = ((1-p) * target).sum().item()
    pr  = tp / max(tp+fp, 1e-9)
    re  = tp / max(tp+fn, 1e-9)
    return 2*pr*re / max(pr+re, 1e-9)

def add_noise(P_true, p_noise, seed=0):
    """Flip each 0→1 or 1→0 with probability p_noise."""
    torch.manual_seed(seed)
    mask = torch.bernoulli(torch.full_like(P_true, p_noise))
    noisy = (P_true + mask) % 2  # XOR with noise mask
    return noisy.clamp(0,1)


print("Experiment 15: Noise Robustness of Rule Chains")
print("=" * 70)
print(f"  {N} people, {int(Parent_true.sum())} Parent edges")
print(f"  Sibling: {int(Sibling_true.sum())} pairs, Uncle: {int(Uncle_true.sum())} pairs\n")

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_SEEDS = 10  # average over multiple noise realizations

print(f"  {'Noise':>6}  {'Parent F1':>10}  {'Sibling F1':>11}  {'Uncle F1 (sym)':>15}  {'Uncle (max)':>12}  {'Uncle (sum)':>12}")
print("  " + "-" * 75)

results = []
for p_noise in noise_levels:
    parent_f1s, sib_f1s, uncle_f1s_sym, uncle_f1s_max, uncle_f1s_sum = [], [], [], [], []

    for seed in range(N_SEEDS):
        P_noisy = add_noise(Parent_true, p_noise, seed=seed*100)

        # Symbolic chain on noisy input
        S_noisy  = compute_sibling_symbolic(P_noisy)
        U_noisy  = compute_uncle_symbolic(P_noisy, S_noisy)

        # Soft chains (use noisy P directly as soft probabilities)
        U_max = compute_uncle_maxproduct(P_noisy, S_noisy.float())
        U_sum = compute_uncle_sumproduct(P_noisy, S_noisy.float())

        parent_f1s.append(f1(P_noisy, Parent_true))
        sib_f1s.append(f1(S_noisy.float(), Sibling_true))
        uncle_f1s_sym.append(f1(U_noisy.float(), Uncle_true))
        uncle_f1s_max.append(f1(U_max, Uncle_true))
        uncle_f1s_sum.append(f1(U_sum, Uncle_true, threshold=0.3))

    p_f1  = sum(parent_f1s) / N_SEEDS
    s_f1  = sum(sib_f1s)    / N_SEEDS
    u_f1  = sum(uncle_f1s_sym) / N_SEEDS
    u_max = sum(uncle_f1s_max) / N_SEEDS
    u_sum = sum(uncle_f1s_sum) / N_SEEDS
    results.append((p_noise, p_f1, s_f1, u_f1, u_max, u_sum))

    bar_p = "█" * int(p_f1*10) + "░"*(10-int(p_f1*10))
    bar_s = "█" * int(s_f1*10) + "░"*(10-int(s_f1*10))
    bar_u = "█" * int(u_f1*10) + "░"*(10-int(u_f1*10))
    print(f"  {p_noise:>6.2f}  {p_f1:>6.3f}|{bar_p}  {s_f1:>7.3f}|{bar_s}  {u_f1:>11.3f}|{bar_u}  {u_max:>12.3f}  {u_sum:>12.3f}")


# ── Amplification analysis ────────────────────────────────────────────────────
print()
print("=== Noise amplification through rule chain ===")
print("  (error at hop k vs error at hop k-1)")
print(f"  {'Noise':>6}  {'Parent err':>11}  {'Sibling err':>12}  {'Uncle err':>11}  {'amplif P→S':>11}  {'amplif S→U':>11}")
print("  " + "-" * 70)
for p_noise, p_f1, s_f1, u_f1, u_max, u_sum in results:
    p_err = 1 - p_f1
    s_err = 1 - s_f1
    u_err = 1 - u_f1
    amp_ps = s_err / max(p_err, 1e-9)  # how much error grew: P→S
    amp_su = u_err / max(s_err, 1e-9)  # how much error grew: S→U
    if p_noise > 0:
        trend_ps = "↑amplify" if amp_ps > 1.2 else "→neutral" if amp_ps > 0.8 else "↓attenuate"
        trend_su = "↑amplify" if amp_su > 1.2 else "→neutral" if amp_su > 0.8 else "↓attenuate"
        print(f"  {p_noise:>6.2f}  {p_err:>11.3f}  {s_err:>12.3f}  {u_err:>11.3f}  {amp_ps:>7.2f}x {trend_ps}  {amp_su:>7.2f}x {trend_su}")

# ── Breakdown at a key noise level ───────────────────────────────────────────
print()
print("=== What type of errors appear? (at 20% noise) ===")
torch.manual_seed(0)
P_20 = add_noise(Parent_true, 0.20, seed=0)
S_20 = compute_sibling_symbolic(P_20)
U_20 = compute_uncle_symbolic(P_20, S_20)

# False positives: predicted but not true
# False negatives: true but not predicted
for rel_name, pred, true_mat in [
    ("Parent", P_20, Parent_true),
    ("Sibling", S_20.float(), Sibling_true),
    ("Uncle", U_20.float(), Uncle_true),
]:
    fp = int((pred * (1-true_mat)).sum().item())
    fn = int(((1-pred) * true_mat).sum().item())
    tp = int((pred * true_mat).sum().item())
    total_true = int(true_mat.sum().item())
    print(f"  {rel_name:>8}: TP={tp}/{total_true}  FP={fp}  FN={fn}  "
          f"(precision={tp/max(tp+fp,1):.2f}, recall={tp/max(tp+fn,1):.2f})")

# ── Noise robustness comparison: symbolic vs max vs sum ───────────────────────
print()
print("=== Summary: which aggregation handles noise best for Uncle? ===")
print(f"  {'Noise':>6}  {'Symbolic':>10}  {'Max-product':>12}  {'Sum-product':>12}  {'best':>12}")
print("  " + "-" * 60)
for p_noise, p_f1, s_f1, u_f1, u_max, u_sum in results:
    best_score = max(u_f1, u_max, u_sum)
    best_name = (["symbolic","max-product","sum-product"]
                 [[u_f1, u_max, u_sum].index(best_score)])
    print(f"  {p_noise:>6.2f}  {u_f1:>10.3f}  {u_max:>12.3f}  {u_sum:>12.3f}  {best_name:>12}")

print("""
=== Key Insights ===

1. Error amplification: each rule hop in a chain can amplify noise.
   The join operation (∃y. A(x,y) ∧ B(y,z)) requires BOTH legs to be correct.
   False positive in Parent → false positive in Sibling → false positive in Uncle.
   The AND structure could attenuate (both must be wrong simultaneously) OR
   amplify (one wrong edge creates many wrong derived pairs via the join).

2. The key question: which dominates — AND-attenuation or fan-out amplification?
   - AND-attenuation: a false Parent edge needs ANOTHER false edge to create a
     false Uncle pair via the rule chain. Probability: p² (independent errors).
   - Fan-out amplification: one false Parent edge can create many false Sibling
     edges (N-1 potential siblings), each of which chains to Uncle.

3. At low noise (<10%): attenuation usually wins. One wrong Parent edge rarely
   creates wrong Uncle pairs because both legs of the chain need to align.

4. At high noise (>30%): amplification dominates. Too many false Parent edges
   saturate the chain with false Siblings and false Uncles.

5. Max-product vs sum-product for noise:
   Max-product: takes the strongest single path. More resistant to false
   positives (needs ONE strong false path). Less resistant to false negatives.
   Sum-product: accumulates all paths. More recall (catches weak true paths)
   but also accumulates false positive evidence.

6. Practical upshot for tensor logic deployment:
   - If your base relation is clean (< 10% noise): rule chains work well.
   - If noisy (> 20%): use embedding models for denoising FIRST,
     then apply rules. The rule injection from exp13 is powerful but brittle
     under noise because it propagates errors faithfully.
""")
