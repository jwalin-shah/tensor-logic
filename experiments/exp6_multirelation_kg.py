"""
Experiment 6: Multi-Relation KG — Rule Composition and Transfer
================================================================
Research question: when multiple relation types are defined via tensor-logic
rules over shared embeddings, does learning one relation help with another?

Relations:
  Parent(x,y):    observed, base relation (direct edges)
  Sibling(x,y):   x and y share a parent       ← rule: ∃z. Parent(z,x) ∧ Parent(z,y)
  GrandParent(x,z): x is y's grandparent       ← rule: ∃y. Parent(x,y) ∧ Parent(y,z)
  Uncle(x,z):     x is uncle/aunt of z         ← rule: ∃y. Sibling(x,y) ∧ Parent(y,z)

All computed by chaining einsum rules over a shared embedding space.
Each object gets an embedding; each relation is a bilinear form over embeddings.

Experiment design:
  A. Rules-only (no learned embeddings): just run the rules symbolically.
     Perfect for what the rules cover; nothing for edges without rule support.

  B. Learned embeddings (train_kg.py style): learn per-object vectors so that
     bilinear(e_x, e_z) ≈ 1 for related pairs. Evaluate on held-out pairs.

  C. Transfer test: train only on Parent, then evaluate whether
     GrandParent and Uncle are inferred correctly via rule composition.
     No GrandParent training signal needed — only Parent edges observed.

Key finding to look for: does the tensor-logic rule chain
    Uncle(x,z) :- Sibling(x,y), Parent(y,z) :- Parent(z', x), Parent(z', y), Parent(y,z)
correctly infer Uncle relationships purely from Parent observations?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ── Family KG ─────────────────────────────────────────────────────────────────
# Generation 1: Alice (0), Bob (1)
# Generation 2: Carol (2), Dan (3) — children of Alice and Bob
# Generation 3: Eve (4), Frank (5), Grace (6), Hank (7)
#               Eve, Frank are children of Carol
#               Grace, Hank are children of Dan

N = 8
names = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Hank"]

parent_pairs = [
    (0, 2), (0, 3),  # Alice → Carol, Dan
    (1, 2), (1, 3),  # Bob → Carol, Dan
    (2, 4), (2, 5),  # Carol → Eve, Frank
    (3, 6), (3, 7),  # Dan → Grace, Hank
]

Parent = torch.zeros(N, N)
for i, j in parent_pairs:
    Parent[i, j] = 1.0


# ── Symbolic rule computation ─────────────────────────────────────────────────
def compute_sibling(P):
    """Sibling(x,y) :- Parent(z,x), Parent(z,y), x ≠ y"""
    S = torch.einsum("zx,zy->xy", P, P)
    S.fill_diagonal_(0)  # no self-siblinghood
    return (S > 0).float()

def compute_grandparent(P):
    """GrandParent(x,z) :- Parent(x,y), Parent(y,z)"""
    return (torch.einsum("xy,yz->xz", P, P) > 0).float()

def compute_uncle(P, S):
    """Uncle(x,z) :- Sibling(x,y), Parent(y,z)"""
    return (torch.einsum("xy,yz->xz", S, P) > 0).float()


Sibling = compute_sibling(Parent)
GrandParent = compute_grandparent(Parent)
Uncle = compute_uncle(Parent, Sibling)

print("Experiment 6: Multi-Relation KG — Rule Composition")
print("=" * 65)
print(f"  People: {names}")
print(f"  Parent edges: {[(names[i],names[j]) for i,j in parent_pairs]}")

def show_relation(M, rel_name):
    pairs = [(names[i], names[j]) for i in range(N) for j in range(N) if M[i,j] > 0.5]
    print(f"\n  {rel_name}: {pairs}")

show_relation(Parent, "Parent")
show_relation(Sibling, "Sibling (via rule)")
show_relation(GrandParent, "GrandParent (via rule)")
show_relation(Uncle, "Uncle (via composed rule)")


# ── Count relationship sizes ──────────────────────────────────────────────────
print(f"\n  Relation sizes:")
for M, name in [(Parent, "Parent"), (Sibling, "Sibling"), (GrandParent, "GrandParent"), (Uncle, "Uncle")]:
    n = int(M.sum().item())
    pct = n / (N * N) * 100
    print(f"    {name:>14}: {n:>3} pairs ({pct:.1f}% of {N}×{N} = {N*N})")


# ── Learned embeddings: bilinear scoring ──────────────────────────────────────
print()
print("=== Learning embeddings from Parent only, evaluating rule chains ===")

DIM = 16  # embedding dimension

class BilinearKG(nn.Module):
    """Each object has an embedding; each relation has a bilinear weight matrix."""
    def __init__(self, n, dim, n_relations):
        super().__init__()
        self.emb = nn.Embedding(n, dim)
        self.W = nn.ParameterList([nn.Parameter(torch.randn(dim, dim) * 0.1)
                                   for _ in range(n_relations)])
        nn.init.orthogonal_(self.emb.weight)

    def score(self, rel_idx, src, dst):
        """Bilinear score: e_src^T W_rel e_dst"""
        E = self.emb.weight
        return (E[src] @ self.W[rel_idx] @ E[dst].T).squeeze()

    def score_matrix(self, rel_idx):
        """Full N×N score matrix for a relation."""
        E = self.emb.weight
        return E @ self.W[rel_idx] @ E.T


REL_PARENT = 0
REL_SIBLING = 1
REL_GRANDPARENT = 2
REL_UNCLE = 3
n_relations = 4

def train_kg(supervise_relations, steps=3000, lr=3e-3):
    """
    Train KG model. supervise_relations: list of (rel_idx, target_matrix) pairs.
    """
    model = BilinearKG(N, DIM, n_relations)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    targets = {rel: target for rel, target in supervise_relations}

    for step in range(steps):
        loss = torch.tensor(0.0)
        for rel_idx, target in targets.items():
            scores = model.score_matrix(rel_idx)
            loss = loss + F.binary_cross_entropy_with_logits(scores, target)
        opt.zero_grad(); loss.backward(); opt.step()

    return model

def eval_relation(model, rel_idx, target, threshold=0.0):
    with torch.no_grad():
        scores = model.score_matrix(rel_idx)
        preds = (scores > threshold).float()
        tp = (preds * target).sum().item()
        fp = (preds * (1 - target)).sum().item()
        fn = ((1 - preds) * target).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return prec, rec, f1


# Condition A: Train on Parent only, evaluate all relations
print("\n  Condition A: Train on Parent only")
model_A = train_kg([(REL_PARENT, Parent)])
for rel_idx, rel_name, target in [
    (REL_PARENT, "Parent", Parent),
    (REL_SIBLING, "Sibling", Sibling),
    (REL_GRANDPARENT, "GrandParent", GrandParent),
    (REL_UNCLE, "Uncle", Uncle),
]:
    p, r, f1 = eval_relation(model_A, rel_idx, target)
    print(f"    {rel_name:>14}:  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")

# Condition B: Train on Parent + Sibling + GrandParent, evaluate Uncle (unseen)
print("\n  Condition B: Train on Parent + Sibling + GrandParent, evaluate Uncle (never trained)")
model_B = train_kg([
    (REL_PARENT, Parent),
    (REL_SIBLING, Sibling),
    (REL_GRANDPARENT, GrandParent),
])
for rel_idx, rel_name, target in [
    (REL_PARENT, "Parent", Parent),
    (REL_SIBLING, "Sibling", Sibling),
    (REL_GRANDPARENT, "GrandParent", GrandParent),
    (REL_UNCLE, "Uncle (zero-shot)", Uncle),
]:
    p, r, f1 = eval_relation(model_B, rel_idx, target)
    print(f"    {rel_name:>22}:  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")

# Condition C: Train on all including Uncle
print("\n  Condition C: Train on all relations (upper bound)")
model_C = train_kg([
    (REL_PARENT, Parent),
    (REL_SIBLING, Sibling),
    (REL_GRANDPARENT, GrandParent),
    (REL_UNCLE, Uncle),
])
for rel_idx, rel_name, target in [
    (REL_PARENT, "Parent", Parent),
    (REL_SIBLING, "Sibling", Sibling),
    (REL_GRANDPARENT, "GrandParent", GrandParent),
    (REL_UNCLE, "Uncle", Uncle),
]:
    p, r, f1 = eval_relation(model_C, rel_idx, target)
    print(f"    {rel_name:>14}:  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")


# ── Pure symbolic rule chain ───────────────────────────────────────────────────
print()
print("=== Pure symbolic rule chain: Parent → Sibling → Uncle ===")
# Demonstrate the 3-step rule application symbolically
S_from_P = compute_sibling(Parent)
GP_from_P = compute_grandparent(Parent)
U_from_PS = compute_uncle(Parent, S_from_P)

# Show the uncle matrix
print("\n  Uncle matrix (computed by rule chain, rows=uncle, cols=niece/nephew):")
print("         " + " ".join(f"{n[:4]:>6}" for n in names))
for i in range(N):
    row = f"  {names[i]:>6}: " + " ".join(f"{int(U_from_PS[i,j]):>6}" for j in range(N))
    print(row)

uncle_pairs = [(names[i], names[j]) for i in range(N) for j in range(N) if U_from_PS[i,j] > 0]
print(f"\n  Uncle pairs: {uncle_pairs}")

# Verify: Carol is uncle/aunt of Grace and Hank (Dan's children)
# Dan is uncle/aunt of Eve and Frank (Carol's children)
print()
print("  Verification:")
print(f"    Carol → Grace: {int(U_from_PS[2,6])} (expected 1, Carol is sibling of Dan who is parent of Grace)")
print(f"    Dan → Eve:     {int(U_from_PS[3,4])} (expected 1, Dan is sibling of Carol who is parent of Eve)")
print(f"    Alice → Eve:   {int(U_from_PS[0,4])} (expected 0, Alice is grandparent not uncle)")

print("""
=== Key Insights ===

1. Rule composition works out of the box: Uncle(x,z) computed by chaining
   Sibling (itself computed from Parent) and Parent gives the exact right answer.
   Three tensor contractions, zero hallucination.

2. The embedding model (bilinear) has to DISCOVER this structure from data.
   Training on Parent only, it scores Sibling/GrandParent/Uncle by geometric
   proximity in the embedding space — not by rule application.

3. Transfer gap: condition B (train on 3 relations, evaluate Uncle zero-shot)
   tests whether the embedding space has captured the relational geometry well
   enough that Uncle pairs are geometrically near each other. This is the
   implicit "relational reasoning" of embedding models.

4. The symbolic rule chain is always correct; the embedding model's zero-shot
   generalization depends on how well the bilinear geometry captures the
   compositional structure.

5. Practical upshot: for known rules, use the symbolic chain (perfect, fast).
   For unknown structure (new relations), use learned embeddings to find
   geometric proximity, then use tensor-logic rules to refine.
   This is the hybrid architecture: rules for known structure, embeddings for
   the unknown.
""")
