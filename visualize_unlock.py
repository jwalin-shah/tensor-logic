import torch
import numpy as np
import matplotlib.pyplot as plt

def solve_transitive_closure(Edge, T, iters=5):
    N = Edge.shape[0]
    Path = Edge.clone()
    for _ in range(iters):
        joined = torch.einsum("xy,yz->xz", Path, Edge)
        if T == 0:
            Path = (Edge + joined > 0).float()
        else:
            Path = torch.sigmoid((Edge + joined - 0.5) / T) # -0.5 threshold for clarity
    return Path.numpy()

# Set up a simple graph: 0->1->2->3, and 1->4
N = 5
edges = [(0, 1), (1, 2), (2, 3), (1, 4)]
Edge = torch.zeros(N, N)
for u, v in edges:
    Edge[u, v] = 1.0

# Generate visualizations
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: The Activation Function
x = np.linspace(-1.5, 1.5, 500)
temperatures = [1.0, 0.5, 0.1, 0.0]
for T in temperatures:
    if T == 0.0:
        y = np.where(x > 0, 1.0, 0.0)
        axs[0].plot(x, y, label="T = 0 (Strict Logic)", linewidth=3, linestyle='--', color='black')
    else:
        y = 1 / (1 + np.exp(-x / T))
        axs[0].plot(x, y, label=f"T = {T} (Neural)", linewidth=2)

axs[0].set_title("The Tensor Logic Primitive", fontweight='bold')
axs[0].set_xlabel("Einsum Output (Pre-activation)")
axs[0].set_ylabel("Truth Value")
axs[0].grid(True, linestyle=':', alpha=0.7)
axs[0].legend()

# Subplot 2: Blurry Neural State (T=0.5)
path_neural = solve_transitive_closure(Edge, T=0.5)
im1 = axs[1].imshow(path_neural, cmap='Blues', vmin=0, vmax=1)
axs[1].set_title("Soft Analogical Reasoning (T=0.5)", fontweight='bold')
axs[1].set_xlabel("Destination Node")
axs[1].set_ylabel("Source Node")
axs[1].set_xticks(range(N))
axs[1].set_yticks(range(N))
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# Subplot 3: Strict Logic State (T=0.0)
path_logic = solve_transitive_closure(Edge, T=0.0)
im2 = axs[2].imshow(path_logic, cmap='Blues', vmin=0, vmax=1)
axs[2].set_title("Strict Deduction (T=0.0)", fontweight='bold')
axs[2].set_xlabel("Destination Node")
axs[2].set_ylabel("Source Node")
axs[2].set_xticks(range(N))
axs[2].set_yticks(range(N))
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.suptitle("The Massive Unlock: Generalized Einsum Bridges Deep Learning and Symbolic Logic", fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("tensor_logic_unlock.png", bbox_inches='tight', dpi=300)
print("Saved tensor_logic_unlock.png")
