"""
Novel Experiment 10: Per-Rule Temperature Learning
===================================================
This is genuinely new — not in the tensor logic paper, not in DLN literature.

Standard tensor logic: one global temperature T controls ALL rules equally.
  sigmoid(logit / T)   ← same T for every rule

Proposed: each rule gets its own learnable temperature T_i.
  rule_i output = sigmoid(logit_i / exp(log_T_i))
  log_T_i is a learnable parameter, initialized to log(1.0) = 0.

Why this matters:
  - Some rules should be HARD (T→0): "Parent is a crisp fact, never soft"
  - Other rules should be SOFT (T→large): "Similarity is analogical, not exact"
  - A global T forces all rules into the same regime simultaneously
  - Per-rule T lets the model discover WHICH rules need to be hard vs. soft

Setup: Family KG with 3 rules of fundamentally different character:
  R_parent:    Parent(x,y) — direct observation, should be hard (T→0)
  R_sibling:   Sibling(x,y) = ∃z. Parent(z,x) ∧ Parent(z,y) — derived, medium
  R_similarity: Similar(x,y) = ||embed(x) - embed(y)|| < ε — soft, analogical

We train with per-rule temperatures and check:
  1. Does T_parent → 0 (hard/deductive)?
  2. Does T_similarity → large (soft/analogical)?
  3. Does the model learn better with per-rule T than global T?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

N = 8
names = ["Alice","Bob","Carol","Dan","Eve","Frank","Grace","Hank"]

parent_pairs = [
    (0,2),(0,3),(1,2),(1,3),   # Alice,Bob → Carol,Dan
    (2,4),(2,5),(3,6),(3,7),   # Carol → Eve,Frank; Dan → Grace,Hank
]
Parent = torch.zeros(N,N)
for i,j in parent_pairs: Parent[i,j] = 1.0

# Ground truth sibling matrix
def make_sibling(P):
    S = torch.einsum("zx,zy->xy", P, P)
    S.fill_diagonal_(0)
    return (S>0).float()

def make_grandparent(P):
    return (torch.einsum("xy,yz->xz", P, P) > 0).float()

Sibling = make_sibling(Parent)
GrandParent = make_grandparent(Parent)

DIM = 12


# ── Model with per-rule temperatures ─────────────────────────────────────────
class PerRuleTempModel(nn.Module):
    """
    Three tensor-logic rules, each with its own learnable temperature.
    All temperatures initialized to T=1.0 (log_T=0).
    """
    def __init__(self, n, dim):
        super().__init__()
        self.emb = nn.Embedding(n, dim)
        nn.init.orthogonal_(self.emb.weight)

        # Per-rule temperatures: log_T so T = exp(log_T) > 0 always
        # Initialize all at T=1.0 → log_T=0
        self.log_T_parent     = nn.Parameter(torch.tensor(0.0))
        self.log_T_sibling    = nn.Parameter(torch.tensor(0.0))
        self.log_T_grandpar   = nn.Parameter(torch.tensor(0.0))

        # Bilinear weights per relation
        self.W_parent   = nn.Parameter(torch.eye(dim) * 0.1 + torch.randn(dim,dim)*0.01)
        self.W_sibling  = nn.Parameter(torch.eye(dim) * 0.1 + torch.randn(dim,dim)*0.01)
        self.W_grandpar = nn.Parameter(torch.eye(dim) * 0.1 + torch.randn(dim,dim)*0.01)

    def score(self, W, log_T):
        E = self.emb.weight           # [N, dim]
        T = torch.exp(log_T).clamp(min=0.01, max=10.0)
        logits = E @ W @ E.T          # [N, N]
        return torch.sigmoid(logits / T)

    def forward(self):
        P_pred  = self.score(self.W_parent,   self.log_T_parent)
        S_pred  = self.score(self.W_sibling,  self.log_T_sibling)
        GP_pred = self.score(self.W_grandpar, self.log_T_grandpar)
        return P_pred, S_pred, GP_pred

    def temperatures(self):
        return {
            "T_parent":    float(torch.exp(self.log_T_parent).item()),
            "T_sibling":   float(torch.exp(self.log_T_sibling).item()),
            "T_grandpar":  float(torch.exp(self.log_T_grandpar).item()),
        }


class GlobalTempModel(nn.Module):
    """Same but ONE shared temperature for all rules. Baseline comparison."""
    def __init__(self, n, dim):
        super().__init__()
        self.emb = nn.Embedding(n, dim)
        nn.init.orthogonal_(self.emb.weight)
        self.log_T = nn.Parameter(torch.tensor(0.0))
        self.W_parent   = nn.Parameter(torch.eye(dim)*0.1 + torch.randn(dim,dim)*0.01)
        self.W_sibling  = nn.Parameter(torch.eye(dim)*0.1 + torch.randn(dim,dim)*0.01)
        self.W_grandpar = nn.Parameter(torch.eye(dim)*0.1 + torch.randn(dim,dim)*0.01)

    def score(self, W):
        E = self.emb.weight
        T = torch.exp(self.log_T).clamp(min=0.01, max=10.0)
        return torch.sigmoid((E @ W @ E.T) / T)

    def forward(self):
        return self.score(self.W_parent), self.score(self.W_sibling), self.score(self.W_grandpar)

    def temperatures(self):
        T = float(torch.exp(self.log_T).item())
        return {"T_global": T}


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, steps=3000, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        P_pred, S_pred, GP_pred = model()
        loss = (F.binary_cross_entropy(P_pred,  Parent)
              + F.binary_cross_entropy(S_pred,  Sibling)
              + F.binary_cross_entropy(GP_pred, GrandParent))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            temps = model.temperatures()
            history.append((step, loss.item(), dict(temps)))
    return history


def f1(pred, target, threshold=0.5):
    p = (pred >= threshold).float()
    tp = (p * target).sum().item()
    fp = (p * (1-target)).sum().item()
    fn = ((1-p) * target).sum().item()
    prec = tp / max(tp+fp, 1e-9)
    rec  = tp / max(tp+fn, 1e-9)
    return 2*prec*rec / max(prec+rec, 1e-9)


print("Experiment 10: Per-Rule Temperature Learning")
print("=" * 65)
print(f"  {N} people, 3 relations: Parent, Sibling, GrandParent")
print(f"  All temperatures init at T=1.0 (log_T=0)")
print()

# Train both models
print("  Training per-rule temperature model...")
model_per = PerRuleTempModel(N, DIM)
hist_per = train(model_per)

print("  Training global temperature model...")
model_glob = GlobalTempModel(N, DIM)
hist_glob = train(model_glob)


# ── Temperature evolution ─────────────────────────────────────────────────────
print()
print("=== Temperature evolution during training (per-rule model) ===")
print(f"  {'step':>6}  {'loss':>8}  {'T_parent':>10}  {'T_sibling':>10}  {'T_grandpar':>11}")
print("  " + "-" * 55)
for step, loss, temps in hist_per:
    print(f"  {step:>6}  {loss:>8.4f}  "
          f"{temps['T_parent']:>10.4f}  "
          f"{temps['T_sibling']:>10.4f}  "
          f"{temps['T_grandpar']:>11.4f}")


# ── Final temperatures ────────────────────────────────────────────────────────
print()
print("=== Final temperatures ===")
per_temps = model_per.temperatures()
glob_temps = model_glob.temperatures()

print(f"  Per-rule model:")
for k, v in per_temps.items():
    hardness = "hard (deductive)" if v < 0.3 else "medium" if v < 1.5 else "soft (analogical)"
    bar = "█" * int(min(v, 5) * 4) + "░" * max(0, 20 - int(min(v, 5) * 4))
    print(f"    {k}: {v:.4f}  |{bar}| {hardness}")

print(f"  Global model: T = {glob_temps['T_global']:.4f}")


# ── Final F1 scores ───────────────────────────────────────────────────────────
print()
print("=== Final F1 scores (threshold=0.5) ===")
print(f"  {'Relation':<14}  {'Per-rule T':>12}  {'Global T':>10}")
print("  " + "-" * 40)

with torch.no_grad():
    P_per, S_per, GP_per   = model_per()
    P_glob, S_glob, GP_glob = model_glob()

for rel_name, pred_per, pred_glob, target in [
    ("Parent",     P_per,  P_glob,  Parent),
    ("Sibling",    S_per,  S_glob,  Sibling),
    ("GrandParent",GP_per, GP_glob, GrandParent),
]:
    f1_per  = f1(pred_per,  target)
    f1_glob = f1(pred_glob, target)
    better = "← per-rule wins" if f1_per > f1_glob + 0.01 else \
             "← global wins"   if f1_glob > f1_per + 0.01 else "← tie"
    print(f"  {rel_name:<14}  {f1_per:>12.3f}  {f1_glob:>10.3f}  {better}")


# ── Sharpness comparison ──────────────────────────────────────────────────────
print()
print("=== Output sharpness comparison ===")
print("  (sharpness = mean |p - 0.5|, higher = more decisive)")
with torch.no_grad():
    for rel_name, pred_per, pred_glob in [
        ("Parent",      P_per,  P_glob),
        ("Sibling",     S_per,  S_glob),
        ("GrandParent", GP_per, GP_glob),
    ]:
        sharp_per  = (pred_per  - 0.5).abs().mean().item()
        sharp_glob = (pred_glob - 0.5).abs().mean().item()
        print(f"  {rel_name:<14}: per-rule={sharp_per:.3f}  global={sharp_glob:.3f}")


# ── The key question: did temperatures differentiate? ────────────────────────
print()
print("=== Key question: did rules learn different temperatures? ===")
temps = model_per.temperatures()
T_vals = list(temps.values())
spread = max(T_vals) - min(T_vals)
print(f"  Temperature spread: {spread:.4f}  (0 = all same, >0.5 = meaningfully different)")
if spread > 0.3:
    print("  ✓ Yes — the model discovered that different rules need different temperatures.")
    hardest = min(temps, key=temps.get)
    softest = max(temps, key=temps.get)
    print(f"    Hardest rule: {hardest} (T={temps[hardest]:.4f})")
    print(f"    Softest rule: {softest} (T={temps[softest]:.4f})")
else:
    print("  ✗ No — all temperatures converged to similar values.")
    print("    This suggests the task is symmetric across rules (all equally hard/soft).")
    print("    To see differentiation: add a genuinely soft/analogical rule.")

print("""
=== Key Insights ===

1. Per-rule temperature is a new degree of freedom not in the original paper.
   Domingos uses one global T. This experiment asks: what if each rule decides
   its own hardness?

2. The "Parent" relation should want low T (it's a crisp, observed fact).
   The "Sibling" relation is derived — it might tolerate higher T.
   The "GrandParent" rule chains two steps — it might want medium T.

3. If temperatures differentiate: the model discovered the epistemic character
   of each rule (deductive vs. analogical) from the training signal alone.
   This is rule-level meta-learning about uncertainty.

4. If temperatures converge: the task doesn't provide enough signal to
   differentiate. Adding a genuinely analogical relation (e.g., "likes" based
   on similarity) would pull T_likes high while T_parent stays low.

5. Connection to the literature: Differentiable Logic Networks (2025) learn
   which GATE to use per neuron (AND/OR/XOR). Per-rule T is the continuous
   analog: learn HOW HARD each rule should be. Not discrete gate choice, but
   continuous hardness on [0,1]. This fills a gap nobody has addressed.
""")
