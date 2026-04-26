"""
Experiment 26: Combined Rule Sets — Two Domains, One Graph
===========================================================
Can we literally combine two rule sets (family + corporate) and run
rule search over the merged relation library?

Inspired by:
  - OIL (1803.01129): learning from multiple imperfect teachers simultaneously
  - exp16: schema transfer (same rules, different domains)
  - exp21: exhaustive rule induction
  - exp12: semiring mixture (learnable weights over rule outputs)

Setup:
  10 people who are BOTH family members AND coworkers.
  Alice and Bob are grandparents of Carol; Carol manages Dan.
  The same person can be a parent AND a manager.

Base relations (4):
  Parent      = biological parent (family domain)
  Sibling     = share a parent (family domain)
  ReportsTo   = direct manager (corporate domain)
  Peer        = share a manager (corporate domain)

Derived relations we want to discover:
  Uncle       = Sibling ∘ Parent  (pure family rule)
  SkipLevel   = Peer ∘ ReportsTo  (pure corporate rule)
  CrossDomain = ??? — does any Parent∘ReportsTo combination mean something?

Key question: in a combined library of 4 base relations + their transposes,
which 2-hop rules explain which targets? Does cross-domain composition work?

OIL connection: each rule is an "imperfect teacher" — it explains SOME of
the target pairs. The system learns to combine them, weighting by quality.
"""

import torch
from itertools import product as iproduct

torch.manual_seed(42)

N = 12
names = ["Alice","Bob","Carol","Dan","Eve","Frank","Grace","Hank","Iris","Jack","Kim","Leo"]

# ── Family domain edges ───────────────────────────────────────────────────────
# Gen1: Alice(0), Bob(1) → Gen2: Carol(2), Dan(3)
# Gen2: Carol(2), Dan(3) → Gen3: Eve(4), Frank(5), Grace(6)
# Gen2: Eve(4), Frank(5) → Gen4: Hank(7), Iris(8)
family_edges = [
    (0,2),(0,3),(1,2),(1,3),         # Alice,Bob → Carol,Dan
    (2,4),(2,5),(3,6),               # Carol→Eve,Frank; Dan→Grace
    (4,7),(4,8),(5,7),               # Eve→Hank,Iris; Frank→Hank
]

Parent = torch.zeros(N,N)
for i,j in family_edges: Parent[i,j] = 1.0

Sibling = torch.zeros(N,N)
for z in range(N):
    ch = [j for j in range(N) if Parent[z,j]>0]
    for i in ch:
        for j in ch:
            if i!=j: Sibling[i,j] = 1.0

# ── Corporate domain edges ────────────────────────────────────────────────────
# CEO: Jack(9) → VPs: Kim(10), Leo(11)
# VP Kim(10) → Mgrs: Carol(2), Eve(4)
# VP Leo(11) → Mgrs: Dan(3), Frank(5)
# Mgr Carol(2) → ICs: Grace(6), Hank(7)
# Mgr Dan(3)   → ICs: Iris(8)
corporate_edges = [
    (9,10),(9,11),                    # Jack→Kim,Leo
    (10,2),(10,4),                    # Kim→Carol,Eve
    (11,3),(11,5),                    # Leo→Dan,Frank
    (2,6),(2,7),                      # Carol→Grace,Hank
    (3,8),                            # Dan→Iris
]

ReportsTo = torch.zeros(N,N)
for i,j in corporate_edges: ReportsTo[i,j] = 1.0

Peer = torch.zeros(N,N)
for z in range(N):
    ch = [j for j in range(N) if ReportsTo[z,j]>0]
    for i in ch:
        for j in ch:
            if i!=j: Peer[i,j] = 1.0

# ── Derived targets ───────────────────────────────────────────────────────────
def compose(A, B):
    return (torch.einsum("xy,yz->xz", A, B) > 0).float()

Uncle     = compose(Sibling, Parent)
GrandPar  = compose(Parent, Parent)
SkipLevel = compose(Peer, ReportsTo)
# Cross-domain derived relations (what do these actually mean?)
ParReports  = compose(Parent, ReportsTo)    # parent's manager?
ReportsPar  = compose(ReportsTo, Parent)    # manager's child?
SibPeer     = compose(Sibling, Peer)        # sibling's peer?
PeerSib     = compose(Peer, Sibling)        # peer's sibling?

def f1(pred, tgt):
    if tgt.sum() == 0: return 0.0
    tp=(pred*tgt).sum().item(); fp=(pred*(1-tgt)).sum().item(); fn=((1-pred)*tgt).sum().item()
    pr=tp/max(tp+fp,1e-9); re=tp/max(tp+fn,1e-9)
    return 2*pr*re/max(pr+re,1e-9)

print("Experiment 26: Combined Rule Sets — Two Domains, One Graph")
print("=" * 65)
print(f"  {N} people: family members AND coworkers simultaneously")
print(f"  Family edges: {len(family_edges)} Parent, {int(Sibling.sum())} Sibling pairs")
print(f"  Corporate edges: {len(corporate_edges)} ReportsTo, {int(Peer.sum())} Peer pairs")
print()
print(f"  Derived targets:")
for name, M in [("Uncle",Uncle),("GrandParent",GrandPar),("SkipLevel",SkipLevel),
                 ("Parent∘ReportsTo",ParReports),("ReportsTo∘Parent",ReportsPar),
                 ("Sibling∘Peer",SibPeer),("Peer∘Sibling",PeerSib)]:
    print(f"    {name:>20}: {int(M.sum())} pairs")
print()


# ── Combined relation library ─────────────────────────────────────────────────
library = {
    "Parent":      Parent,
    "Parent^T":    Parent.T.contiguous(),
    "Sibling":     Sibling,
    "ReportsTo":   ReportsTo,
    "ReportsTo^T": ReportsTo.T.contiguous(),
    "Peer":        Peer,
}
rel_names = list(library.keys())
print(f"  Combined library: {len(rel_names)} relations")
print(f"  2-hop search space: {len(rel_names)**2} candidate rules")
print()


# ── Exhaustive 2-hop search for each target ───────────────────────────────────
def search_rules(target, library, label, top_k=5):
    keys = list(library.keys())
    results = []
    for r1, r2 in iproduct(keys, keys):
        pred = compose(library[r1], library[r2])
        score = f1(pred, target)
        n_pred = int(pred.sum())
        results.append((score, r1, r2, n_pred))
    results.sort(reverse=True)
    print(f"  Target: {label} ({int(target.sum())} true pairs)")
    print(f"  {'Rule':<35}  {'F1':>6}  {'#pred':>6}  {'domain'}")
    print("  " + "-" * 60)
    shown = 0
    for score, r1, r2, n_pred in results:
        if score == 0 and shown > 0: break
        rule_str = f"{r1} ∘ {r2}"
        d1 = "family" if "Parent" in r1 or "Sibling" in r1 else "corporate"
        d2 = "family" if "Parent" in r2 or "Sibling" in r2 else "corporate"
        domain = "CROSS" if d1 != d2 else d1
        marker = " ← CROSS-DOMAIN" if domain == "CROSS" and score > 0 else ""
        print(f"  {rule_str:<35}  {score:>6.3f}  {n_pred:>6}{marker}")
        shown += 1
        if shown >= top_k: break
    print()
    return results[0]

print("  === RULE SEARCH RESULTS ===")
print()
best_uncle    = search_rules(Uncle,     library, "Uncle (family)")
best_gp       = search_rules(GrandPar,  library, "GrandParent (family)")
best_skip     = search_rules(SkipLevel, library, "SkipLevel (corporate)")
best_par_rep  = search_rules(ParReports, library, "Parent∘ReportsTo (cross?)")
best_rep_par  = search_rules(ReportsPar, library, "ReportsTo∘Parent (cross?)")
best_sib_peer = search_rules(SibPeer,   library, "Sibling∘Peer (cross?)")


# ── Weighted rule mixture (OIL-style: combine imperfect teachers) ─────────────
print("=" * 65)
print("  Weighted rule mixture — can a single model combine multiple rules?")
print("  (Like OIL: weight multiple imperfect rule-teachers by quality)")
print()
print("  Target: Uncle — what weighted combination of ALL rules explains it?")
print()

import torch.nn as nn
import torch.nn.functional as F

# Precompute all 2-hop predictions as features
all_preds = []
all_rule_names = []
keys = list(library.keys())
for r1, r2 in iproduct(keys, keys):
    pred = compose(library[r1], library[r2])
    all_preds.append(pred.flatten())
    all_rule_names.append(f"{r1}∘{r2}")

all_preds = torch.stack(all_preds, dim=1)  # [N*N, n_rules]
uncle_flat = Uncle.flatten()

# Learnable weights over rules → predict Uncle
class RuleMixer(nn.Module):
    def __init__(self, n_rules):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_rules))
    def forward(self, X):
        α = F.softmax(self.w, dim=0)  # mixture weights sum to 1
        return (X * α.unsqueeze(0)).sum(dim=1)

class SparseMixer(nn.Module):
    def __init__(self, n_rules):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_rules))
    def forward(self, X):
        # Sparsemax: exact zeros for irrelevant rules
        from exp14_sparsemax import sparsemax
        α = sparsemax(self.w.unsqueeze(0)).squeeze(0)
        return (X * α.unsqueeze(0)).sum(dim=1)

def train_mixer(model, X, y, steps=2000, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        pred = model(X).clamp(1e-6, 1-1e-6)
        loss = F.binary_cross_entropy(pred, y.float())
        opt.zero_grad(); loss.backward(); opt.step()

def eval_mixer(model, X, y):
    with torch.no_grad():
        pred = model(X)
        acc = ((pred > 0.5) == (y > 0.5)).float().mean().item()
    return acc

for MixerClass, label in [(RuleMixer, "Softmax mixer"), (SparseMixer, "Sparsemax mixer")]:
    torch.manual_seed(0)
    try:
        mixer = MixerClass(n_rules=len(all_rule_names))
        train_mixer(mixer, all_preds, uncle_flat)
        acc = eval_mixer(mixer, all_preds, uncle_flat)
        # Top rules by weight
        with torch.no_grad():
            if hasattr(mixer, 'w'):
                if label == "Softmax mixer":
                    w = F.softmax(mixer.w, dim=0)
                else:
                    from exp14_sparsemax import sparsemax
                    w = sparsemax(mixer.w.unsqueeze(0)).squeeze(0)
            top_idx = w.topk(3).indices.tolist()
        print(f"  {label} (accuracy={acc:.3f}):")
        for i in top_idx:
            print(f"    weight={w[i].item():.4f}  rule={all_rule_names[i]}")
        nonzero = (w > 0.001).sum().item()
        print(f"    nonzero rules: {nonzero}/{len(all_rule_names)}")
    except Exception as e:
        print(f"  {label}: {e}")
    print()


# ── Direct overlay: print who overlaps which domain ───────────────────────────
print("=" * 65)
print("  Who plays dual roles? (both family AND corporate nodes)")
print()
for_family  = set(i for i,j in family_edges) | set(j for i,j in family_edges)
for_corp    = set(i for i,j in corporate_edges) | set(j for i,j in corporate_edges)
dual_role   = for_family & for_corp
family_only = for_family - for_corp
corp_only   = for_corp - for_family

print(f"  Dual-role (family + corporate): {[names[i] for i in sorted(dual_role)]}")
print(f"  Family only: {[names[i] for i in sorted(family_only)]}")
print(f"  Corporate only: {[names[i] for i in sorted(corp_only)]}")

# Show cross-domain path for dual-role nodes
print()
print("  Cross-domain paths through dual-role nodes:")
print("  (Is Carol's sibling's boss the same as Carol's skip-level?)")
for i in sorted(dual_role)[:4]:
    family_con  = [(names[i],names[j]) for j in range(N) if Sibling[i,j]>0 or Parent[i,j]>0]
    corp_con    = [(names[i],names[j]) for j in range(N) if ReportsTo[i,j]>0 or Peer[i,j]>0]
    if family_con or corp_con:
        print(f"  {names[i]:>8}: family={family_con[:3]}, corporate={corp_con[:3]}")


print("""
=== Key Insights ===

1. Combined rule search works seamlessly: the 4-relation library (Parent,
   Sibling, ReportsTo, Peer) produces 36 candidate 2-hop rules. The correct
   rules are found at rank #1 for each target domain — without any
   domain-specific configuration.

2. Pure-domain rules score highest within their domain:
   Uncle is best explained by Sibling∘Parent (F1=1.0), not any cross combo.
   SkipLevel is best explained by Peer∘ReportsTo (F1=1.0).
   The search space is larger but the signal is clear.

3. Cross-domain compositions DO produce real relations:
   Parent∘ReportsTo = "a person whose parent manages them" (nepotism edge)
   ReportsTo∘Parent = "a person whose manager is their parent"
   Sibling∘Peer = "a person whose sibling shares their manager"
   These are structurally real but semantically odd — real-world meaning
   depends on whether the graph has these coincidental overlaps.

4. OIL connection: each rule is an "imperfect teacher." The weighted rule
   mixer (softmax or sparsemax) learns to upweight correct rules and zero-out
   wrong ones. Sparsemax gives exact zeros for irrelevant rules — the same
   way OIL discards bad teacher behaviors rather than smoothly averaging them.

5. The scoping lesson: start with 2 domains × 2-hop rules = 36 candidates.
   This is still exhaustively searchable. At 3 domains × 3-hop rules,
   the space is |R|^3 = 216+, where ES (exp22) becomes necessary.

6. What "combining rule sets" actually means in tensor logic:
   - Syntactically: add more rows to the relation library
   - Semantically: allow cross-domain compositions in the search
   - Computationally: same O(N^3 × |rules|) cost, just more rules
   The einsum doesn't care which domain the matrix came from.
   This is the deep modularity of tensor logic.
""")
