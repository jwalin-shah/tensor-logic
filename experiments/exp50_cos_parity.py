"""
exp50: Periodic-activation TL on parity — does changing the operator break exp48's ceiling?

exp48 showed sigmoid+iteration cannot express parity at any (α, β, γ_X, δ).
Hypothesis: a periodic activation with period-2 IS a homomorphism (mod-2-like) over
real-valued path counts, so a TL with `act(x) = (1 - cos(π·x))/2` should compute
parity exactly when α=1 (raw integer accumulation).

Activation choice: act(x) = (1 - cos(π·x)) / 2  maps integer x → x mod 2:
  0→0, 1→1, 2→0, 3→1, 4→0, ...   (smooth, differentiable, period 2)

Recurrence under cosine:
    R_{k+1} = act(α · R@A + β · R + γ)
With α=1, β=0, γ=0 and binary A: R@A is integer path-count, act(R@A) is parity.

Tests (matching exp48 setup):
  1. Frozen ideal weights (α=1, β=0, γ=0): should give F1≈1 if the structural claim is right.
  2. Trained from ideal-init: does SGD stay or drift?
  3. Trained from closure-init (α=5, β=5, γ=-2.5): does SGD find the parity basin?
  4. Trained from random init: optimization landscape check.

Compare against exp48's TL_OR4 (F1=0.685, attenuated-OR ceiling) and exp45 TL_OR (F1=0.26).
"""
import math
import statistics
import torch
import torch.nn as nn

import exp45_pathcount_falsify as e45

DEVICE = e45.DEVICE
N_NODES, EDGE_PROB = 16, 0.30
N_TRAIN, N_TEST, N_STEPS, N_SEEDS = 500, 50, 1000, 3


class TL_COS(nn.Module):
    """3-param TL with periodic activation: R ← (1 - cos(π·(α·R@A + β·R + γ)))/2."""
    def __init__(self, K=4, alpha=1.0, beta=0.0, gamma=0.0):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, A):
        R = A.clone()
        for _ in range(self.K):
            comp = torch.einsum("bij,bjk->bik", R, A)
            x = self.alpha * comp + self.beta * R + self.gamma
            R = 0.5 * (1.0 - torch.cos(math.pi * x))
        return R


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  steps={N_STEPS}  seeds={N_SEEDS}")
    print("act(x) = (1 - cos(π·x))/2  →  integer x → x mod 2")

    summary = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        A_tr, R_tr, M_tr, P_tr = e45.gen_data(N_NODES, EDGE_PROB, N_TRAIN, seed * 1000)
        A_te, R_te, M_te, P_te = e45.gen_data(N_NODES, EDGE_PROB, N_TEST, seed * 1000 + 7)

        configs = [
            ("frozen_ideal",    dict(alpha=1.0, beta=0.0, gamma=0.0), False),
            ("train_ideal",     dict(alpha=1.0, beta=0.0, gamma=0.0), True),
            ("train_closure",   dict(alpha=5.0, beta=5.0, gamma=-2.5), True),
        ]
        for name, init, do_train in configs:
            torch.manual_seed(seed)
            m = TL_COS(**init).to(DEVICE)
            if not do_train:
                for p in m.parameters(): p.requires_grad = False
            else:
                e45.train_loop(m, A_tr, P_tr, N_STEPS, loss_kind="bce")
            f1 = e45.eval_closure(m, A_te, P_te)
            a, b, g = m.alpha.item(), m.beta.item(), m.gamma.item()
            summary.setdefault(name, []).append(f1)
            print(f"  seed={seed}  {name:<15} F1={f1:.3f}  (α={a:.2f}, β={b:.2f}, γ={g:.2f})")

        # Random init: 3 fresh seeds per outer seed for landscape view
        for r_idx in range(3):
            torch.manual_seed(seed * 100 + r_idx)
            a0, b0, g0 = [torch.randn(1).item() * 2 for _ in range(3)]
            m = TL_COS(alpha=a0, beta=b0, gamma=g0).to(DEVICE)
            e45.train_loop(m, A_tr, P_tr, N_STEPS, loss_kind="bce")
            f1 = e45.eval_closure(m, A_te, P_te)
            a, b, g = m.alpha.item(), m.beta.item(), m.gamma.item()
            summary.setdefault("train_rand", []).append(f1)
            print(f"  seed={seed} r={r_idx} train_rand     F1={f1:.3f}  init=({a0:+.2f},{b0:+.2f},{g0:+.2f}) → (α={a:.2f},β={b:.2f},γ={g:.2f})")

    print("\n=== summary ===")
    print(f"{'config':<18}{'F1':<22}")
    for name in ["frozen_ideal", "train_ideal", "train_closure", "train_rand"]:
        vs = summary[name]
        m, s = statistics.mean(vs), statistics.stdev(vs) if len(vs) > 1 else 0.0
        print(f"{name:<18}{m:.3f}±{s:.3f}   (n={len(vs)})")

    print("\nReference (from exp45/exp48):")
    print("  TL_OR  (sigmoid, 3-param)  parity F1 ≈ 0.26")
    print("  TL_OR4 (sigmoid + cross-term, 4-param)  parity F1 ≈ 0.69")


if __name__ == "__main__":
    main()
