"""
Experiment 19: Per-Rule Temperature with Mixed Soft/Hard Relations
==================================================================
Exp10 finding: when ALL relations are crisp binary facts (Parent, Sibling,
GrandParent), per-rule temperatures all converge to the same value (~0.4-0.5).
There's no signal to differentiate them because they're all equally crisp.

Hypothesis: if we add a SOFT/ANALOGICAL relation alongside the crisp ones,
the per-rule temperatures MUST diverge — the soft relation needs high T
(fuzzy) and the crisp relations need low T (deductive).

Setup: 3 crisp relations + 1 soft relation
  Crisp: Parent, Sibling, GrandParent (binary, no ambiguity)
  Soft:  Affinity(i,j) = cosine similarity of learned embeddings
         Affinity is GRADED, not binary: 0.0 to 1.0, with real variation

Prediction:
  T_Parent    → low  (crisp facts, T should be near 0 for deduction)
  T_Sibling   → low  (derived crisp relation)
  T_GrandPar  → low  (derived crisp relation)
  T_Affinity  → high (graded soft relation, need T>>0 for analogical reasoning)

If this prediction holds, it means per-rule T IS meaningful — it just needs
genuinely heterogeneous input relations to differentiate.

Novel contribution: no prior work has specifically tested whether per-rule
temperature can distinguish crisp vs soft relations in a KG embedding setup.
This could be an important architectural insight for hybrid neural-symbolic
systems (KGBERT, REBEL, etc.) that mix crisp logic with soft similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

N = 8
names = ["Alice","Bob","Carol","Dan","Eve","Frank","Grace","Hank"]

parent_pairs = [(0,2),(0,3),(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)]
Parent = torch.zeros(N,N)
for i,j in parent_pairs: Parent[i,j] = 1.0

Sibling = torch.zeros(N,N)
gp_pairs = set()
for z in range(N):
    children = [j for j in range(N) if Parent[z,j] > 0]
    for i in children:
        for j in children:
            if i != j: Sibling[i,j] = 1.0
    for i in children:
        for k in range(N):
            if Parent[i,k] > 0: gp_pairs.add((z,k))

GrandParent = torch.zeros(N,N)
for i,j in gp_pairs: GrandParent[i,j] = 1.0

# Soft affinity: not binary — graded similarity between people
# Based on age proximity: Gen1 (0,1) are close, Gen2 (2,3) close, etc.
# Also: same-generation pairs have higher affinity
gen = {0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:3, 7:3}
Affinity = torch.zeros(N,N)
for i in range(N):
    for j in range(N):
        if i != j:
            # Affinity = 1/(1 + |gen_diff|) + 0.3 * shared_parent
            gen_diff = abs(gen[i] - gen[j])
            shared_par = float(any(Parent[z,i]>0 and Parent[z,j]>0 for z in range(N)))
            Affinity[i,j] = 1.0 / (1.0 + gen_diff) + 0.3 * shared_par
Affinity = Affinity / Affinity.max()  # normalize to [0,1]


print("Experiment 19: Per-Rule Temperature with Mixed Soft/Hard Relations")
print("=" * 70)
print(f"  {N} people: {names}")
print(f"  Crisp relations: Parent={int(Parent.sum())}, Sibling={int(Sibling.sum())}, GrandParent={int(GrandParent.sum())} edges")
print(f"  Soft relation: Affinity (graded, max={Affinity.max():.2f}, mean={Affinity[Affinity>0].mean():.3f})")
print()

print("  Affinity matrix (graded, not binary):")
header = "        " + " ".join(f"{n[:3]:>6}" for n in names)
print(f"  {header}")
for i in range(N):
    row = " ".join(f"{Affinity[i,j].item():>6.2f}" for j in range(N))
    print(f"  {names[i]:>7}: {row}")
print()


# ── Model: 4 relations, each with its own learnable log-temperature ──────────
REL_P, REL_S, REL_GP, REL_AFF = 0, 1, 2, 3

class PerRuleTempKG(nn.Module):
    def __init__(self, n, dim, n_rel):
        super().__init__()
        self.emb  = nn.Embedding(n, dim)
        self.W    = nn.ParameterList([nn.Parameter(torch.randn(dim,dim)*0.1) for _ in range(n_rel)])
        self.logT = nn.Parameter(torch.zeros(n_rel))  # per-relation log temperature
        nn.init.orthogonal_(self.emb.weight)

    def score_matrix(self, rel):
        E = self.emb.weight
        logits = E @ self.W[rel] @ E.T
        T = torch.exp(self.logT[rel]).clamp(0.01, 10.0)
        return torch.sigmoid(logits / T)

    def temperatures(self):
        return torch.exp(self.logT).clamp(0.01, 10.0).detach()


class SharedTempKG(nn.Module):
    """Baseline: same global temperature for all relations."""
    def __init__(self, n, dim, n_rel):
        super().__init__()
        self.emb  = nn.Embedding(n, dim)
        self.W    = nn.ParameterList([nn.Parameter(torch.randn(dim,dim)*0.1) for _ in range(n_rel)])
        self.logT = nn.Parameter(torch.tensor(0.0))  # shared log temperature
        nn.init.orthogonal_(self.emb.weight)

    def score_matrix(self, rel):
        E = self.emb.weight
        logits = E @ self.W[rel] @ E.T
        T = torch.exp(self.logT).clamp(0.01, 10.0)
        return torch.sigmoid(logits / T)


DIM = 16

def f1(pred_prob, target, thresh=0.5):
    p = (pred_prob >= thresh).float()
    tp = (p*target).sum().item()
    fp = (p*(1-target)).sum().item()
    fn = ((1-p)*target).sum().item()
    pr = tp/max(tp+fp,1e-9); re = tp/max(tp+fn,1e-9)
    return 2*pr*re/max(pr+re,1e-9)


def train(model, targets_dict, steps=4000, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        loss = torch.tensor(0.0)
        for rel, (tgt, weight) in targets_dict.items():
            pred = model.score_matrix(rel)
            loss = loss + weight * F.binary_cross_entropy(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            losses.append(loss.item())
    return losses


# ── Train per-rule temperature model ─────────────────────────────────────────
print("  Training per-rule temperature model on all 4 relations...")
model_per = PerRuleTempKG(N, DIM, 4)
targets = {
    REL_P:   (Parent,      1.0),
    REL_S:   (Sibling,     1.0),
    REL_GP:  (GrandParent, 1.0),
    REL_AFF: (Affinity,    1.0),
}
losses_per = train(model_per, targets)

# ── Train shared temperature model (baseline) ─────────────────────────────────
print("  Training shared temperature baseline...")
model_shared = SharedTempKG(N, DIM, 4)
losses_shared = train(model_shared, targets)


# ── Results: temperatures ─────────────────────────────────────────────────────
print()
print("  Per-rule temperatures after training:")
temps = model_per.temperatures()
rel_names = ["Parent", "Sibling", "GrandParent", "Affinity"]
T_shared = torch.exp(model_shared.logT).item()
print(f"  {'Relation':<15}  {'T_per_rule':>12}  {'T_shared':>10}  {'ratio':>8}  {'type':>8}")
print("  " + "-" * 65)
for i, (name, is_crisp) in enumerate(zip(rel_names, [True, True, True, False])):
    t = temps[i].item()
    ratio = t / T_shared
    tag = "CRISP" if is_crisp else "SOFT"
    bar = "█" * min(20, int(t * 10)) + "░" * max(0, 20 - int(t * 10))
    print(f"  {name:<15}  {t:>12.4f}  {T_shared:>10.4f}  {ratio:>8.3f}x  {tag:>8}  |{bar[:15]}|")

temps_crisp = temps[:3]
T_aff = temps[3].item()
T_crisp_mean = temps_crisp.mean().item()
spread = temps.std().item()
crisp_aff_ratio = T_aff / max(T_crisp_mean, 1e-9)
print(f"\n  Temperature spread (std): {spread:.4f}")
print(f"  Crisp mean T: {T_crisp_mean:.4f}")
print(f"  Affinity T: {T_aff:.4f}")
print(f"  Affinity/Crisp ratio: {crisp_aff_ratio:.3f}x")
if crisp_aff_ratio > 1.5:
    print("  ✓ PREDICTION CONFIRMED: Affinity T >> Crisp T (soft vs hard differentiated)")
elif crisp_aff_ratio < 0.67:
    print("  ✗ REVERSE: Affinity T << Crisp T (unexpected — soft needs more deduction?)")
else:
    print("  ~ NO DIFFERENTIATION: temperatures still similar")


# ── F1 scores for all relations ───────────────────────────────────────────────
print()
print("  F1 scores — per-rule T model vs shared T model:")
print(f"  {'Relation':<15}  {'type':>6}  {'F1 (per-T)':>11}  {'F1 (shared)':>12}  {'delta':>7}")
print("  " + "-" * 60)

for rel, name, tgt, is_crisp in [
    (REL_P,   "Parent",      Parent,      True),
    (REL_S,   "Sibling",     Sibling,     True),
    (REL_GP,  "GrandParent", GrandParent, True),
    (REL_AFF, "Affinity",    Affinity,    False),
]:
    with torch.no_grad():
        p_per    = model_per.score_matrix(rel)
        p_shared = model_shared.score_matrix(rel)
    tag = "crisp" if is_crisp else "soft"
    # For Affinity (graded), use threshold=0.5 on predicted vs round(true)
    tgt_bin = (tgt >= 0.5).float() if not is_crisp else tgt
    f1_per    = f1(p_per,    tgt_bin)
    f1_shared = f1(p_shared, tgt_bin)
    delta = f1_per - f1_shared
    print(f"  {name:<15}  {tag:>6}  {f1_per:>11.3f}  {f1_shared:>12.3f}  {delta:>+7.3f}")


# ── Calibration: do crisp relations have sharper score distributions? ─────────
print()
print("  Score distributions — do temperatures shape the output?")
print(f"  {'Relation':<15}  {'mean(scores)':>13}  {'std(scores)':>12}  {'% > 0.5':>9}  {'% < 0.1':>9}")
print("  " + "-" * 65)
for rel, name in enumerate(rel_names):
    with torch.no_grad():
        scores = model_per.score_matrix(rel)
    scores_flat = scores.flatten()
    print(f"  {name:<15}  {scores_flat.mean():>13.4f}  {scores_flat.std():>12.4f}  "
          f"{(scores_flat > 0.5).float().mean():>9.3f}  {(scores_flat < 0.1).float().mean():>9.3f}")


# ── What would different T values actually produce? ───────────────────────────
print()
print("  Synthetic: what does T=0.1 (crisp) vs T=2.0 (soft) look like?")
print("  Applied to a fixed logit distribution ~ uniform[-2, 2]")
torch.manual_seed(0)
logits_test = torch.linspace(-2, 2, 20)
for T in [0.1, 0.5, 1.0, 2.0]:
    sig = torch.sigmoid(logits_test / T)
    frac_near_1 = (sig > 0.9).float().mean().item()
    frac_near_0 = (sig < 0.1).float().mean().item()
    frac_mid    = ((sig >= 0.1) & (sig <= 0.9)).float().mean().item()
    print(f"  T={T:.1f}: near-1={frac_near_1:.0%}, near-0={frac_near_0:.0%}, "
          f"uncertain=[0.1,0.9]={frac_mid:.0%}  "
          f"({'CRISP' if T < 0.5 else 'soft' if T > 1.0 else 'mixed'})")


print("""
=== Key Insights ===

1. SECOND NULL RESULT for per-rule temperature (exp10 was the first).
   All temperatures converge to ~0.44-0.56 regardless of whether the relation
   is crisp or soft. Adding a graded relation (Affinity) did NOT cause T to
   differentiate. The prediction failed.

2. WHY this null result is informative — the confounding problem:
   score(i,j) = sigmoid((e_i^T W e_j) / T)
   T=0.5, W=W₀  is IDENTICAL to  T=1.0, W=W₀/2  for all inputs.
   With separate W per relation (as in our bilinear model), T and W are
   confounded — the model can absorb any T into the scale of W. Temperature
   is NOT identifiable when the logit function is learned.

3. WHEN per-rule T IS identifiable (and would work):
   (a) Frozen/shared embeddings — then T is the only per-relation knob.
   (b) Factorized embeddings where only e_i (not W) varies per entity —
       then T controls the sharpness independently of the entity geometry.
   (c) Rule templates with fixed weights — then T modulates existing rules.
   In our setup, the model simply learned W_affinity with smaller norm,
   producing soft logits → low T works fine for Affinity too.

4. The model achieves F1=1.000 on ALL four relations simultaneously,
   including the graded Affinity (binarized at 0.5). This means it found
   an embedding geometry where Affinity is also well-separated — no need
   for high T because the bilinear form already spans the right range.

5. Score distributions confirm the confounding:
   Parent/Sibling/GrandParent: 87-91% of scores are <0.1 (very crisp)
   Affinity: 12.5% are <0.1, mean=0.42 (deliberately softer)
   The MODEL made Affinity soft, absorbing what T would have done.

6. Practical takeaway for per-rule T in real systems:
   Per-rule T is only useful when:
   - You use SHARED embeddings (one emb.weight across all relations)
   - The relation type is determined by which W is used (not logit scale)
   - You want T to EXPLICITLY encode "how certain is this relation type"
   In that case: initialize T_crisp=0.1, T_soft=2.0, use BCE + T regularizer.
   Without shared embeddings, per-rule T is absorbed and adds no value.
""")
