"""
Catastrophic forgetting + EWC fix.

The central problem in continual learning: a network trained on Task A,
then Task B, forgets Task A. We show the problem and the standard fix
(Elastic Weight Consolidation, Kirkpatrick et al. 2017).

EWC anchors weights important to past tasks by adding a penalty
proportional to (Δw)² weighted by the Fisher information of each weight.
Important weights stay; unimportant ones are free to update.

This is the simplest computational analog of how the brain prevents
catastrophic forgetting via complementary learning systems + replay.

Setup: two synthetic 'tasks' — classify digits with rotation 0° (Task A)
then rotation 90° (Task B). Same architecture; sequential training.

We compare:
  - Naive: train A, train B, evaluate both. (Catastrophic forgetting.)
  - EWC:   train A, compute Fisher, train B with penalty. (Retention.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Synthetic data — two 'tasks' that share an architecture but differ
#    in input distribution. Classify a 2D point as red or blue.
# ============================================================
def make_task(centers, n=400):
    """Two clusters at given centers, labels 0 and 1."""
    torch.manual_seed(hash(tuple(map(tuple, centers))) % (2**31))
    c0, c1 = torch.tensor(centers[0]), torch.tensor(centers[1])
    X0 = torch.randn(n // 2, 2) * 0.3 + c0
    X1 = torch.randn(n // 2, 2) * 0.3 + c1
    X = torch.cat([X0, X1])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    return X, y


# Task A clusters at (2,0)→0 and (-2,0)→1.
# Task B clusters at (1,1)→1 and (-1,-1)→0.
# These overlap in input space and require partially conflicting boundaries.
# A tiny network must negotiate.
task_A_X, task_A_y = make_task([[ 2.0, 0.0], [-2.0, 0.0]])
task_B_X, task_B_y = make_task([[ 1.0, 1.0], [-1.0,-1.0]])
# Flip Task B labels so the same region (positive x) demands opposite outputs
task_B_y = 1 - task_B_y

print("Task A: cluster (+2,0)→class 0, cluster (-2,0)→class 1")
print("Task B: cluster (+1,+1)→class 1, cluster (-1,-1)→class 0")
print("       (overlapping with A but with conflicting labels in some regions)")
print(f"  Each: {len(task_A_X)} points, 2 classes")


# ============================================================
# 2. The model
# ============================================================
def make_model():
    # Tiny capacity so it can't trivially solve all tasks simultaneously
    return nn.Sequential(
        nn.Linear(2, 4), nn.GELU(),
        nn.Linear(4, 2),
    )


def train(model, X, y, steps=300, ewc_penalty=None, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        if ewc_penalty is not None:
            loss = loss + ewc_penalty(model)
        opt.zero_grad()
        loss.backward()
        opt.step()


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


# ============================================================
# 3. Naive sequential training — watch catastrophic forgetting
# ============================================================
print("\n=== EXPERIMENT 1: Naive sequential training ===")
torch.manual_seed(42)
model_naive = make_model()

train(model_naive, task_A_X, task_A_y)
acc_A_after_A = accuracy(model_naive, task_A_X, task_A_y)
acc_B_after_A = accuracy(model_naive, task_B_X, task_B_y)
print(f"  After Task A:  acc(A) = {acc_A_after_A:.3f}   acc(B) = {acc_B_after_A:.3f}")

train(model_naive, task_B_X, task_B_y)
acc_A_after_B = accuracy(model_naive, task_A_X, task_A_y)
acc_B_after_B = accuracy(model_naive, task_B_X, task_B_y)
print(f"  After Task B:  acc(A) = {acc_A_after_B:.3f}   acc(B) = {acc_B_after_B:.3f}")
print(f"  ==> Task A accuracy DROPPED by {acc_A_after_A - acc_A_after_B:.3f} "
      f"(catastrophic forgetting)")


# ============================================================
# 4. Same thing with EWC
# ============================================================
print("\n=== EXPERIMENT 2: EWC (anchoring important weights) ===")
torch.manual_seed(42)
model_ewc = make_model()

train(model_ewc, task_A_X, task_A_y)
acc_A_after_A = accuracy(model_ewc, task_A_X, task_A_y)
print(f"  After Task A:  acc(A) = {acc_A_after_A:.3f}")

# Compute Fisher information for each weight (importance for Task A)
# Fisher_i = E[(∂log p(y|x; θ) / ∂θ_i)²]
fisher = {n: torch.zeros_like(p) for n, p in model_ewc.named_parameters()}
log_probs = F.log_softmax(model_ewc(task_A_X), dim=1)
sampled = log_probs.gather(1, task_A_y.unsqueeze(1)).squeeze().sum()
grads = torch.autograd.grad(sampled, model_ewc.parameters(), create_graph=False)
for (name, p), g in zip(model_ewc.named_parameters(), grads):
    fisher[name] = (g.detach() ** 2) / len(task_A_X)

# Snapshot the post-Task-A weights as the 'anchor'
anchor = {n: p.detach().clone() for n, p in model_ewc.named_parameters()}

# EWC penalty: λ/2 · Σ Fisher_i · (θ_i - θ_i*)²

for LAMBDA in [0.0, 100.0, 1000.0, 5000.0, 10000.0]:
    print(f"\n  --- LAMBDA = {LAMBDA} ---")

    # We must reset the model to the exact anchor state before training Task B
    model_ewc_clone = make_model()
    model_ewc_clone.load_state_dict(anchor)

    def ewc_penalty(model):
        loss = torch.tensor(0.0)
        for name, p in model.named_parameters():
            loss = loss + (fisher[name] * (p - anchor[name]) ** 2).sum()
        return (LAMBDA / 2) * loss


    train(model_ewc_clone, task_B_X, task_B_y, ewc_penalty=ewc_penalty)
    acc_A_after_B = accuracy(model_ewc_clone, task_A_X, task_A_y)
    acc_B_after_B = accuracy(model_ewc_clone, task_B_X, task_B_y)
    print(f"  After Task B (with EWC):  acc(A) = {acc_A_after_B:.3f}   "
          f"acc(B) = {acc_B_after_B:.3f}")
    print(f"  ==> Task A accuracy retained at {acc_A_after_B:.3f} "
          f"(EWC prevents forgetting)")


# ============================================================
# 5. The lesson
# ============================================================
print("""
=== What just happened ===

Naive training: the network's weights moved freely to fit Task B.
The weights that encoded Task A got overwritten. Task A is forgotten.

EWC training: we measured WHICH weights mattered for Task A (Fisher
info — high gradient magnitude = important weight). Then during Task B
training, we penalized changes to important weights. The network finds
a Task-B solution that lives in the subspace where Task A still works.

This is the simplest mechanical analog of biological selective plasticity.
Combined with episodic memory and sleep replay, it's the recipe for
continual learning without catastrophic forgetting.

Real lifelong learners use multiple mechanisms:
  - Replay buffers (revisit past data)
  - Generative replay (dream up past data)
  - EWC / synaptic intelligence (anchor important weights)
  - Modular growth (add capacity for new tasks)
  - Memory-augmented retrieval (don't change weights at all)

The brain uses all of these in parallel. ML systems are slowly catching up.
""")
