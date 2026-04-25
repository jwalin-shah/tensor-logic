"""
Trainable Tensor Logic: knowledge-graph completion.

Setup: a tiny family tree. We TELL the model:
  - some Parent(x, y) facts
  - the rule:  Grandparent(x, z) :- Parent(x, y), Parent(y, z)

We DON'T tell it which Grandparent facts are true. It has to derive them by
applying the rule in embedding space, then we score it against ground truth.

What's learned: the per-object embeddings. Same einsum runs on every step;
backprop pushes embeddings into a configuration where the rule produces the
right answers.
"""

import torch
import torch.nn.functional as F

# ---------- ground truth ----------
# 8 people, indexed 0..7. Parent edges:
parent_edges = [(0, 2), (1, 2), (2, 4), (2, 5), (3, 5), (4, 6), (5, 7)]
N = 8

P_true = torch.zeros(N, N)
for u, v in parent_edges:
    P_true[u, v] = 1.0

# Grandparent = Parent @ Parent (boolean), this is what we want the model to learn
GP_true = ((P_true @ P_true) > 0).float()
print("True Grandparent pairs:")
for i in range(N):
    for j in range(N):
        if GP_true[i, j]:
            print(f"  Grandparent({i}, {j})")

# ---------- model ----------

class TensorLogicKG(torch.nn.Module):
    def __init__(self, n_objects, dim):
        super().__init__()
        # Learnable per-object embeddings — the ONLY parameters.
        self.emb = torch.nn.Parameter(torch.randn(n_objects, dim) * 0.3)
        # Embedded relation tensor for Parent (computed from facts each forward pass).
        # We could also make this a parameter; here we ground it in known facts.

    def parent_relation_tensor(self, P_facts):
        """EmbR[i,j] = sum_{(u,v) in P} emb[u,i] * emb[v,j]   — Sec 5 of paper."""
        e = F.normalize(self.emb, dim=1)
        return torch.einsum("uv,ui,vj->ij", P_facts, e, e)

    def forward(self, P_facts, T=0.3):
        e = F.normalize(self.emb, dim=1)
        EmbP = self.parent_relation_tensor(P_facts)              # [D, D]

        # The RULE, in embedding space:
        # Grandparent(x,z) :- Parent(x,y), Parent(y,z)
        # =>  EmbGP = einsum over shared y of (EmbP) ⋅ (EmbP)
        #     but in embedding space this is just EmbP @ EmbP  (chained einsum)
        EmbGP = EmbP @ EmbP                                       # [D, D]

        # Decode: score each (a, b) pair by querying EmbGP with their embeddings.
        # D_pred[a,b] ≈ Grandparent(a, b)
        scores = torch.einsum("ij,ai,bj->ab", EmbGP, e, e)        # [N, N]

        # Apply temperature-controlled sigmoid (Section 5, eq. for σ(x,T))
        return torch.sigmoid(scores / T)

def run_experiment(D, lr):
    print(f"\n======================================")
    print(f"Experiment: D={D}, lr={lr}")
    print(f"======================================")
    model = TensorLogicKG(N, D)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------- training loop ----------
    print("\nTraining...")
    for step in range(400):
        pred = model(P_true)
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), GP_true)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                acc = ((pred > 0.5).float() == GP_true).float().mean()
                print(f"  step {step:3d}  loss={loss.item():.4f}  acc={acc.item():.3f}")

    # ---------- evaluate ----------
    print("\nFinal predictions (threshold 0.5):")
    with torch.no_grad():
        pred = model(P_true)
        for i in range(N):
            for j in range(N):
                p = pred[i, j].item()
                t = GP_true[i, j].item()
                if p > 0.5 or t > 0.5:
                    mark = "OK " if (p > 0.5) == (t > 0.5) else "MISS"
                    print(f"  {mark} ({i},{j})  pred={p:.2f}  true={int(t)}")

    # ---------- the "reasoning in embedding space" test ----------
    # Ask: who are the grandparents of node 7?  (Truth: 2, since 2->5->7)
    # Do it WITHOUT touching the boolean tensors — only embeddings.
    print("\nQuery in embedding space: 'who are grandparents of 7?'")
    with torch.no_grad():
        e = F.normalize(model.emb, dim=1)
        EmbP = model.parent_relation_tensor(P_true)
        EmbGP = EmbP @ EmbP
        q = torch.einsum("ij,bj->bi", EmbGP, e[7:8])   # contract 'b' index with node 7
        scores = (e @ q.T).squeeze()
        for i, s in enumerate(scores.tolist()):
            print(f"  candidate {i}: score={s:.3f}")
        print(f"  argmax = {int(scores.argmax())}  (truth: 2)")

for D in [4, 8, 16]:
    for lr in [0.01, 0.05, 0.1]:
        run_experiment(D, lr)
