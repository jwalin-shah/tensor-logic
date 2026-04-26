"""
exp52: MLP capacity sweep on transitive closure.

Pre-empts the obvious objection to exp47 ("MLP h=64 is too small").
At n=128, sweep MLP hidden sizes h ∈ {64, 512, 2048} and compare to TL (3 params).
Also include n=16 sanity check at h=512 — MLP should be near-perfect at small n with
enough capacity, demonstrating both architectures can succeed when the regime fits.

The structural claim: even at h=2048 (~135M params at n=128), MLP cannot recover the
recursive closure rule, while TL with 3 size-independent params hits F1=1.000.
"""

import math
import statistics

import torch
import torch.nn as nn

import exp44_import_closure as e44

N_TRAIN = 500
N_TEST = 30
N_STEPS = 1000
N_SEEDS = 3
DEVICE = e44.DEVICE


class MLP(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.n = n
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(n * n, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, n * n),
        )

    def forward(self, A):
        x = self.net(A.flatten(-2))
        return torch.sigmoid(x.reshape(*A.shape))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print(f"device={DEVICE}  N_TRAIN={N_TRAIN}  N_TEST={N_TEST}  N_STEPS={N_STEPS}  seeds={N_SEEDS}")

    # Baseline: TL trained at n=16 (3 params), zero-shot evaluated at n=16 and n=128.
    print("\n=== TL baseline (3 params, trained once at n=16) ===")
    torch.manual_seed(0)
    train_A, train_R = e44.gen_graphs(16, 2.0 / 16, N_TRAIN, 0)
    tl = e44.TL(K=4).to(DEVICE)
    e44.train(tl, train_A, train_R, N_STEPS)
    print(f"  TL params: alpha={tl.alpha.item():.3f}  beta={tl.beta.item():.3f}  gamma={tl.gamma.item():.3f}")

    for n in [16, 128]:
        K_tl = max(4, math.ceil(math.log2(max(n, 2))))
        tl.K = K_tl
        f1s = []
        for seed in range(N_SEEDS):
            test_A, test_R = e44.gen_graphs(n, 2.0 / n, N_TEST, seed * 1000 + n + 7)
            f1, _ = e44.evaluate(tl, test_A, test_R)
            f1s.append(f1)
        print(f"  TL @ n={n:<5} F1 = {statistics.mean(f1s):.3f} ± {statistics.stdev(f1s):.3f}  (3 params)")

    # MLP capacity sweep
    print(f"\n=== MLP capacity sweep ===")
    print(f"{'n':<6}{'h':<8}{'params':<12}{'F1 (re-trained)':<22}")
    configs = [
        (16, 64),
        (16, 512),
        (128, 64),
        (128, 512),
        (128, 2048),
    ]
    for n, h in configs:
        f1s = []
        params_count = None
        for seed in range(N_SEEDS):
            torch.manual_seed(seed * 17 + n + h)
            train_A, train_R = e44.gen_graphs(n, 2.0 / n, N_TRAIN, seed * 1000 + n + 100)
            mlp = MLP(n, h=h).to(DEVICE)
            if params_count is None:
                params_count = n_params(mlp)
            e44.train(mlp, train_A, train_R, N_STEPS)
            test_A, test_R = e44.gen_graphs(n, 2.0 / n, N_TEST, seed * 1000 + n + 7)
            f1, _ = e44.evaluate(mlp, test_A, test_R)
            f1s.append(f1)
        f1_str = f"{statistics.mean(f1s):.3f}±{statistics.stdev(f1s):.3f}"
        print(f"{n:<6}{h:<8}{params_count:<12}{f1_str:<22}")


if __name__ == "__main__":
    main()
