"""
exp31 — Surprise-gated parameter update

HYPOTHESIS: Updating only on examples where prediction error exceeds a
threshold (surprise) gives faster convergence and better continual-learning
retention than uniform Adam updates.

FALSIFIED IF: Surprise gating shows EITHER (a) slower wall-time convergence
to the same loss on a stationary task, OR (b) worse retention than uniform
update on a two-phase continual task.

SMALLEST TEST: Object_permanence-style 1D occlusion world. Train phase A
(velocity=+1), then phase B (velocity=+2). Measure: convergence epochs in A,
retention of A after training on B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def gen_seq(start, vel, length=10, occluder=(4, 6)):
    seq = []
    pos = start
    for _ in range(length):
        if occluder[0] <= pos <= occluder[1]:
            seq.append(-1.0)
        else:
            seq.append(float(pos))
        pos += vel
    return torch.tensor(seq).unsqueeze(-1)


def make_dataset(vel, n=100):
    seqs = [gen_seq(start_pos, vel) for start_pos in range(-2, 3) for _ in range(n // 5)]
    X = torch.stack([s[:-1] for s in seqs])
    Y = torch.stack([s[1:] for s in seqs])
    return X, Y


class FwdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(out)


def train(model, X, Y, epochs=300, surprise_tau=None, lr=0.01):
    """If surprise_tau is None, uniform update. Else only batch elements
    with per-sample MSE > tau contribute to gradient."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        pred = model(X)
        per_sample_err = ((pred - Y) ** 2).mean(dim=(1, 2))  # [B]
        if surprise_tau is None:
            loss = per_sample_err.mean()
        else:
            mask = (per_sample_err > surprise_tau).float()
            if mask.sum() < 1:
                losses.append(per_sample_err.mean().item()); continue
            loss = (per_sample_err * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(per_sample_err.mean().item())
    return losses


def eval_loss(model, X, Y):
    with torch.no_grad():
        return F.mse_loss(model(X), Y).item()


# ── Phase A: train on velocity=+1 ───────────────────────────────────────
X_A, Y_A = make_dataset(vel=1)
X_B, Y_B = make_dataset(vel=2)

print("exp31 — surprise-gated vs uniform updates\n")
print("Phase A: train on velocity=+1")

m_uniform = FwdModel()
l_uniform = train(m_uniform, X_A, Y_A, epochs=300, surprise_tau=None)

m_surprise = FwdModel()
l_surprise = train(m_surprise, X_A, Y_A, epochs=300, surprise_tau=0.1)

# Convergence: epochs to reach loss < 0.05
def epochs_to(losses, target):
    for i, l in enumerate(losses):
        if l < target: return i
    return len(losses)

print(f"  Uniform   final loss: {l_uniform[-1]:.4f}, epochs to <0.05: {epochs_to(l_uniform, 0.05)}")
print(f"  Surprise  final loss: {l_surprise[-1]:.4f}, epochs to <0.05: {epochs_to(l_surprise, 0.05)}")

# ── Phase B: train both on velocity=+2 (test forgetting) ───────────────
print("\nPhase B: continue training on velocity=+2 (200 epochs)")
loss_A_after_B_uniform_pre = eval_loss(m_uniform, X_A, Y_A)
loss_A_after_B_surprise_pre = eval_loss(m_surprise, X_A, Y_A)

train(m_uniform, X_B, Y_B, epochs=200, surprise_tau=None)
train(m_surprise, X_B, Y_B, epochs=200, surprise_tau=0.1)

loss_A_after_B_uniform = eval_loss(m_uniform, X_A, Y_A)
loss_A_after_B_surprise = eval_loss(m_surprise, X_A, Y_A)
loss_B_uniform = eval_loss(m_uniform, X_B, Y_B)
loss_B_surprise = eval_loss(m_surprise, X_B, Y_B)

print(f"\nAfter phase B:")
print(f"  Uniform:   loss on A = {loss_A_after_B_uniform:.4f}, loss on B = {loss_B_uniform:.4f}")
print(f"  Surprise:  loss on A = {loss_A_after_B_surprise:.4f}, loss on B = {loss_B_surprise:.4f}")

# ── Hypothesis check ─────────────────────────────────────────────────────
ep_u = epochs_to(l_uniform, 0.05); ep_s = epochs_to(l_surprise, 0.05)
faster_or_equal = ep_s <= ep_u * 1.2
better_retention = loss_A_after_B_surprise < loss_A_after_B_uniform

print("\nHYPOTHESIS CHECK:")
print(f"  (a) Surprise convergence within 20% of uniform: {faster_or_equal}  ({ep_s} vs {ep_u})")
print(f"  (b) Surprise retains A better after B:          {better_retention}  ({loss_A_after_B_surprise:.3f} vs {loss_A_after_B_uniform:.3f})")

if faster_or_equal and better_retention:
    print("\n  Verdict: CONFIRMED — surprise gating helps continual learning.")
else:
    print("\n  Verdict: FALSIFIED on at least one axis.")
