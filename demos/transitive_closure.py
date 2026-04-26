"""
Tensor Logic in 80 lines: transitive closure as einsum + step,
then the same einsum with sigmoid(./T) for analogical reasoning.

Datalog rule:
    Path(x, z) :- Edge(x, z).
    Path(x, z) :- Path(x, y), Edge(y, z).

Tensor Logic form (single equation, fixpointed):
    Path = step( Edge + einsum('xy,yz->xz', Path, Edge) )
"""

import torch

# 5-node DAG-ish graph: 0->1->2->3, plus 1->4
N = 5
edges = [(0, 1), (1, 2), (2, 3), (1, 4)]

Edge = torch.zeros(N, N)
for u, v in edges:
    Edge[u, v] = 1.0


def closure(activation, max_iters=10):
    """Forward-chain Path = activation(Edge + Path @ Edge) to fixpoint."""
    Path = Edge.clone()
    for i in range(max_iters):
        # einsum is the join; sum over shared index y
        joined = torch.einsum("xy,yz->xz", Path, Edge)
        new_Path = activation(Edge + joined)
        if torch.allclose(new_Path, Path):
            print(f"  fixpoint at iter {i}")
            return Path
        Path = new_Path
    return Path


# --- Mode 1: T=0, Heaviside step => pure deduction (boolean transitive closure)
print("=== Deductive (step function, T=0) ===")
deductive = closure(lambda x: (x > 0).float())
print(deductive.int())

# --- Mode 2: T>0, sigmoid => analogical / soft reasoning
#     Same equation, just relaxed nonlinearity. Values become "belief strengths".
print("\n=== Analogical (sigmoid, T=0.5) ===")
T = 0.5
analogical = closure(lambda x: torch.sigmoid(x / T))
print(analogical.round(decimals=2))

# --- Mode 3: embedding-space sanity check
#     Represent each node as a unit vector. Edge becomes a superposition of
#     tensor products. Retrieval = dot product. Show inference still works
#     approximately when we never name a node by index.
print("\n=== Embedding-space retrieval ===")
D = 32  # embedding dim
torch.manual_seed(0)
emb = torch.nn.functional.normalize(torch.randn(N, D), dim=1)  # [N, D]

# Edge_emb[i,j,k] ~ sum over edges (u,v) of emb[u,j] * emb[v,k]
Edge_emb = torch.einsum("ej,ek->jk", emb[[u for u, _ in edges]],
                                       emb[[v for _, v in edges]])
# Query: "who does node 0 point to?" -> contract emb[0] with first axis
query0 = emb[0]                                     # [D]
out_dist = torch.einsum("j,jk->k", query0, Edge_emb)  # [D]
# Retrieve: dot with every node embedding
scores = emb @ out_dist
print("scores for 'edge from node 0 ->':", scores.round(decimals=2).tolist())
print("argmax (should be 1):", int(scores.argmax()))
