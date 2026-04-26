"""
exp47: TL closure size generalization.

The 3-param TL recurrence has size-independent parameters: same (alpha, beta, gamma)
work for any n. Train once at n=16, evaluate at n ∈ {32, 64, 128} with K = ceil(log2(n))
iterations. MLP must be re-trained from scratch at each size because its input/output
dim depends on n.

Sparsity p = 2/n keeps avg degree ~2 (Python-import-like) regardless of n, so closure
density doesn't blow up at large n.

Hypothesis: TL F1 ≥ 0.95 across all sizes; MLP F1 degrades as n grows because flat
n²-dim I/O learning gets harder with fixed n_train and h.
"""

import math
import random
import statistics

import torch

import exp44_import_closure as e44

N_SIZES = [16, 32, 64, 128]
N_TRAIN = 500
N_TEST = 50
N_STEPS = 1000
N_SEEDS = 3
DEVICE = e44.DEVICE


def main():
    print(f"device={DEVICE}  N_TRAIN={N_TRAIN}  N_TEST={N_TEST}  N_STEPS={N_STEPS}  seeds={N_SEEDS}")

    # Train TL once at n=16, K=4
    print("\n=== train TL once at n=16 ===")
    torch.manual_seed(0)
    train_A, train_R = e44.gen_graphs(16, 2.0 / 16, N_TRAIN, 0)
    tl = e44.TL(K=4).to(DEVICE)
    e44.train(tl, train_A, train_R, N_STEPS)
    print(f"  TL params after training: alpha={tl.alpha.item():.3f}  beta={tl.beta.item():.3f}  gamma={tl.gamma.item():.3f}")

    # Evaluate zero-shot at all sizes; train fresh MLP at each size.
    print(f"\n{'n':<6}{'p':<8}{'K_tl':<6}{'closure_dens':<14}{'TL F1 (zero-shot)':<22}{'MLP F1 (re-trained)':<22}{'MLP params':<12}")
    for n in N_SIZES:
        p = 2.0 / n
        K_tl = max(4, math.ceil(math.log2(max(n, 2))))
        tl.K = K_tl

        tl_f1s, mlp_f1s = [], []
        closure_density = None
        mlp_params = None

        for seed in range(N_SEEDS):
            test_A, test_R = e44.gen_graphs(n, p, N_TEST, seed * 1000 + n + 7)
            if closure_density is None:
                closure_density = test_R.float().mean().item()

            f1_tl, _ = e44.evaluate(tl, test_A, test_R)
            tl_f1s.append(f1_tl)

            torch.manual_seed(seed * 17 + n)
            train_A_n, train_R_n = e44.gen_graphs(n, p, N_TRAIN, seed * 1000 + n + 100)
            mlp = e44.MLP(n, h=64).to(DEVICE)
            mlp_params = sum(prm.numel() for prm in mlp.parameters())
            e44.train(mlp, train_A_n, train_R_n, N_STEPS)
            f1_mlp, _ = e44.evaluate(mlp, test_A, test_R)
            mlp_f1s.append(f1_mlp)

        tl_str = f"{statistics.mean(tl_f1s):.3f}±{statistics.stdev(tl_f1s):.3f}"
        mlp_str = f"{statistics.mean(mlp_f1s):.3f}±{statistics.stdev(mlp_f1s):.3f}"
        print(f"{n:<6}{p:<8.4f}{K_tl:<6}{closure_density:<14.3f}{tl_str:<22}{mlp_str:<22}{mlp_params:<12}")


if __name__ == "__main__":
    main()
