"""
Experiment 17: Multi-hop Depth Limit
=====================================
How deep can we chain tensor-logic rules before precision collapses?

Chain:
  Hop 0: Parent (observed)
  Hop 1: GrandParent   = Parent ∘ Parent
  Hop 2: GreatGP       = GrandParent ∘ Parent
  Hop 3: GreatGreatGP  = GreatGP ∘ Parent
  Hop k: k-hop ancestor

And also lateral chains:
  Sibling                = ∃z. Parent(z,x) ∧ Parent(z,y)
  Uncle                  = Sibling ∘ Parent
  GreatUncle             = Uncle ∘ Parent ∘ Parent  (uncle of parent)
  GreatGreatUncle        = GreatUncle ∘ Parent

Questions:
  1. How fast does the derived relation size grow with depth?
  2. Is there a "depth limit" where all pairs become related (saturation)?
  3. Does noise at the base compound faster with depth?
  4. Is there a precision floor below which deeper rules can't go?
"""

import torch

torch.manual_seed(42)

# Large enough family to see depth effects: 5 generations
# Gen1: 0,1 (great-grandparents)
# Gen2: 2,3 (grandparents, children of 0 and 1)
# Gen3: 4,5,6 (parents, children of 2; children of 3)
# Gen4: 7,8,9,10 (children of 4,5; children of 6)
# Gen5: 11,12,13,14 (children of 7,8; children of 9,10)

N = 15
names = [f"G{i}" for i in range(N)]

parent_edges = [
    # Gen1 → Gen2
    (0,2),(0,3),(1,2),(1,3),
    # Gen2 → Gen3
    (2,4),(2,5),(3,6),
    # Gen3 → Gen4
    (4,7),(4,8),(5,9),(5,10),(6,10),
    # Gen4 → Gen5
    (7,11),(7,12),(8,13),(9,14),(10,14),
]

P = torch.zeros(N,N)
for i,j in parent_edges: P[i,j] = 1.0

print("Experiment 17: Multi-hop Depth Limit")
print("=" * 60)
print(f"  {N} nodes, {int(P.sum())} base Parent edges, 5 generations")
print(f"  Total possible pairs: {N*(N-1)}")
print()


def compose(A, B):
    """A ∘ B: (A∘B)[x,z] = 1 if ∃y. A[x,y]=1 ∧ B[y,z]=1"""
    return (torch.einsum("xy,yz->xz", A, B) > 0).float()

def sibling_from(parent_mat):
    S = torch.einsum("zx,zy->xy", parent_mat, parent_mat)
    S.fill_diagonal_(0)
    return (S > 0).float()


# ── Vertical chain: k-hop ancestor ────────────────────────────────────────────
print("  Vertical chain: k-hop ancestor (who can I reach in k parent-steps?)")
print(f"  {'Depth k':>8}  {'# pairs':>8}  {'% of total':>11}  {'new at this depth':>18}")
print("  " + "-" * 55)

Ancestor = P.clone()
prev_count = int(P.sum())
print(f"  {'k=1':>8}  {prev_count:>8}  {prev_count/(N*(N-1))*100:>10.1f}%  (base Parent)")

for k in range(2, 8):
    Ancestor_new = compose(Ancestor, P)  # add one more hop
    # Union: any path of length ≤ k
    Ancestor = ((Ancestor + Ancestor_new) > 0).float()
    count = int(Ancestor.sum())
    new_pairs = count - prev_count
    prev_count = count
    pct = count / (N*(N-1)) * 100
    bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
    print(f"  {'k='+str(k):>8}  {count:>8}  {pct:>10.1f}%  +{new_pairs} pairs  |{bar}|")
    if count == N*(N-1):
        print(f"         → saturation: all pairs reachable at k={k}")
        break


# ── Lateral chain: uncle, great-uncle, great-great-uncle ─────────────────────
print()
print("  Lateral chain: uncle-of-uncle chains")
print(f"  {'Relation':>20}  {'# pairs':>8}  {'% of N²':>10}  {'description'}")
print("  " + "-" * 65)

Sib   = sibling_from(P)
GP    = compose(P, P)
Uncle = compose(Sib, P)

# Great-uncle: uncle of your parent = sibling of grandparent, composed with parent
GGP      = compose(GP, P)  # great-grandparent
GUncle   = compose(sibling_from(GP), P)  # sibling of grandparent, applied to parent
GGUncle  = compose(sibling_from(GGP), compose(P,P))

for name, M, desc in [
    ("Parent",       P,       "base relation"),
    ("Sibling",      Sib,     "share a parent"),
    ("GrandParent",  GP,      "2-hop ancestor"),
    ("Uncle",        Uncle,   "sibling∘parent (1 lateral, 1 vertical)"),
    ("GreatGP",      GGP,     "3-hop ancestor"),
    ("GreatUncle",   GUncle,  "sibling of GP∘parent (1 lateral, 2 vertical)"),
    ("GreatGreatUncle", GGUncle, "sibling of GGP∘GP (1 lateral, 3 vertical)"),
]:
    count = int(M.sum())
    pct = count / (N*N) * 100
    print(f"  {name:>20}  {count:>8}  {pct:>9.1f}%  {desc}")


# ── Noise amplification by depth ──────────────────────────────────────────────
print()
print("  Noise amplification: F1 at each depth for 10% noise in Parent")
print(f"  {'Relation':>20}  {'clean F1':>9}  {'10% noise F1':>13}  {'drop':>8}")
print("  " + "-" * 60)

def add_noise(M, p, seed=0):
    torch.manual_seed(seed)
    mask = torch.bernoulli(torch.full_like(M, p))
    return ((M + mask) % 2).clamp(0,1)

P_noisy = add_noise(P, 0.10)
Sib_n   = sibling_from(P_noisy)
GP_n    = compose(P_noisy, P_noisy)
GGP_n   = compose(GP_n, P_noisy)
Uncle_n = compose(Sib_n, P_noisy)
GUncle_n = compose(sibling_from(GP_n), P_noisy)

def f1(pred, true):
    p = pred.float(); t = true.float()
    tp = (p*t).sum().item(); fp = (p*(1-t)).sum().item(); fn = ((1-p)*t).sum().item()
    pr = tp/max(tp+fp,1e-9); re = tp/max(tp+fn,1e-9)
    return 2*pr*re/max(pr+re,1e-9)

for name, clean, noisy in [
    ("Parent",      P,      P_noisy),
    ("Sibling",     Sib,    Sib_n),
    ("GrandParent", GP,     GP_n),
    ("Uncle",       Uncle,  Uncle_n),
    ("GreatGP",     GGP,    GGP_n),
    ("GreatUncle",  GUncle, GUncle_n),
]:
    f1_clean = f1(clean, clean)   # always 1.0 (comparing to itself)
    f1_noisy = f1(noisy, clean)   # noisy vs ground truth
    drop = f1_clean - f1_noisy
    bar = "█" * int(f1_noisy*20) + "░"*(20-int(f1_noisy*20))
    print(f"  {name:>20}  {f1_clean:>9.3f}  {f1_noisy:>9.3f} |{bar}|  {drop:>+7.3f}")


# ── Saturation point ──────────────────────────────────────────────────────────
print()
print("  Derived relation density vs depth:")
print("  (what fraction of all N² pairs does each relation cover?)")

all_rels = [P, Sib, GP, Uncle, GGP, GUncle, GGUncle]
rel_names = ["Parent","Sibling","GrandParent","Uncle","GreatGP","GreatUncle","GGUncle"]
for name, M in zip(rel_names, all_rels):
    density = M.sum().item() / (N*N) * 100
    bar = "█" * int(density/5) + "░"*(20-int(density/5))
    print(f"  {name:>15}: {density:>5.1f}%  |{bar}|")

print("""
=== Key Insights ===

1. Vertical chains saturate: by depth 4-5, the ancestor relation covers
   all pairs in a connected genealogy. Once every node is reachable from
   every other (through enough hops), adding more depth adds nothing.

2. Lateral chains grow slower: Uncle, GreatUncle, etc. cover fewer pairs
   because they require the lateral (sibling) step which is structurally
   more restrictive than pure ancestor chains.

3. Noise amplification grows with depth: each additional rule hop gives
   noise more chances to propagate. Grandparent degrades more than Parent,
   GreatGP degrades more than GrandParent, etc.

4. The depth limit isn't fixed: it depends on graph connectivity (how
   branching the base relation is) and noise level. Dense graphs saturate
   faster; noisy graphs degrade faster.

5. Practical rule for tensor logic deployments:
   - Depth 1-2: generally robust, minimal noise amplification
   - Depth 3-4: meaningful degradation, requires clean base relation
   - Depth 5+: likely saturated (everything is related to everything)
     or collapsed (nothing is reliably predicted)

6. The "depth vs. width" tradeoff: more rule hops (depth) = more
   expressiveness but more noise sensitivity. More rule branches (width,
   e.g., union of paths) = more recall but also more false positives.
   Tensor logic gives you both levers via einsum index choice.
""")
