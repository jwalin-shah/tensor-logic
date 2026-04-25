"""
Tensor-Logic-Augmented State Space Model.

A State Space Model (SSM) like Mamba uses a linear recurrence:
h[t] = A * h[t-1] + B * x[t]
y[t] = C * h[t]

In a Tensor-Logic-Augmented SSM, the hidden state h[t] represents a structured
knowledge base (e.g. an embedding-space relation tensor), and the recurrence
step applies a tensor-logic rule to forward-chain reasoning over time.

This script demonstrates an SSM that receives a sequence of facts over time
(e.g., Parent facts) and continuously maintains a forward-chained relation
(Grandparent) in its hidden state, doing linear-cost reasoning over a typed state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorSSM(nn.Module):
    def __init__(self, dim):
        """
        dim: Dimension of the entity embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        """
        xs: Sequence of new facts at each time step. Shape [seq_len, dim, dim].
            Each x_t is an embedding-space relation (e.g. EmbParent).

        Returns the sequence of hidden states representing the derived
        Grandparent relation over time.
        """
        seq_len = xs.shape[0]

        # State:
        # P_state: unnormalized accumulation of Parent facts
        # GP_state: unnormalized accumulation of Grandparent paths
        P_state = torch.zeros(self.dim, self.dim)
        GP_state = torch.zeros(self.dim, self.dim)

        out_GP = []

        for t in range(seq_len):
            x_t = xs[t]  # New Parent fact at time t

            # Recurrence step: tensor-logic rule application
            # Rule: Grandparent(x, z) :- Parent(x, y), Parent(y, z)
            # Incremental expansion of (P + x) @ (P + x):
            # GP_t = GP_{t-1} + P_{t-1} @ x_t + x_t @ P_{t-1} + x_t @ x_t
            new_GP_paths = P_state @ x_t + x_t @ P_state + x_t @ x_t

            # Update states
            GP_state = GP_state + new_GP_paths
            P_state = P_state + x_t

            out_GP.append(GP_state.clone())

        return torch.stack(out_GP)

def main():
    # Setup: 4 people (A=0, B=1, C=2, D=3)
    N_PEOPLE = 4
    D = 256  # larger dim to make random embeddings more orthogonal
    T_TEMP = 0.05

    # Assign random orthogonal embeddings to the people
    torch.manual_seed(42)
    # Use exact orthogonal embeddings to remove noise completely
    emb = torch.zeros(N_PEOPLE, D)
    for i in range(N_PEOPLE):
        emb[i, i] = 1.0

    def encode_fact(u, v):
        # Fact: u is parent of v
        # Embed as emb[u] ⊗ emb[v]
        return torch.einsum("i,j->ij", emb[u], emb[v])

    # Temporal sequence of events:
    # t=0: A is parent of B
    # t=1: B is parent of C
    # t=2: C is parent of D
    xs = torch.stack([
        encode_fact(0, 1),
        encode_fact(1, 2),
        encode_fact(2, 3)
    ])

    ssm = TensorSSM(dim=D)

    # Process sequence through the SSM
    print("Feeding sequence of facts into Tensor SSM...")
    out_GP = ssm(xs)

    def check_gp(t, u, v):
        # Decode the belief by querying the SSM's hidden state at time t
        # score = emb[u] @ GP_state[t] @ emb[v]
        score = torch.einsum("i,ij,j->", emb[u], out_GP[t], emb[v])
        # Apply temperature-scaled sigmoid to get truth value
        # we center at 0.5 because dot product is 0 for orthogonal vectors
        # and 1 for positive match
        return torch.sigmoid((score - 0.5) / T_TEMP).item(), score.item()

    print("\nTemporal sequence of facts:")
    print(" t=0: Parent(A, B)")
    print(" t=1: Parent(B, C)")
    print(" t=2: Parent(C, D)")

    print("\nModel's Grandparent beliefs over time (sigmoid output, >0.5 is True):")
    queries = [
        (0, 2, "Grandparent(A, C)"),
        (1, 3, "Grandparent(B, D)"),
        (0, 3, "Grandparent(A, D) [Great-GP, should be false]")
    ]

    for t in range(3):
        print(f"\n After t={t}:")
        for u, v, name in queries:
            s, raw = check_gp(t, u, v)
            is_true = "✓" if s > 0.5 else "✗"
            print(f"   {name:<45}: {s:.3f} (raw {raw:+.3f}) {is_true}")

if __name__ == "__main__":
    main()
