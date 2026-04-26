"""
exp51: TL closure size generalization to real-codebase scale.

Extends exp47 to n ∈ {128, 256, 512, 1024}. Same protocol: train TL once at n=16,
evaluate zero-shot at every size with K = ceil(log2(n)) iterations.

We skip MLP retraining at n>=256 because:
  (a) exp47 already showed MLP collapses to F1=0 at n=128 with 2.1M params,
  (b) MLP at n=1024 is ~134M params, ~30 min/run, and will collapse anyway.

The point of this experiment: confirm TL's 3 size-independent params hold up at
realistic codebase scales (hundreds to thousands of nodes).
"""

import math
import statistics

import torch

import exp44_import_closure as e44

N_SIZES = [128, 256, 512, 1024]
N_TRAIN = 500
N_TEST = 30
N_STEPS = 1000
N_SEEDS = 3
DEVICE = e44.DEVICE


def main():
    print(f"device={DEVICE}  N_TRAIN={N_TRAIN}  N_TEST={N_TEST}  N_STEPS={N_STEPS}  seeds={N_SEEDS}")

    # Train TL once at n=16 (exact same as exp47).
    print("\n=== train TL once at n=16 ===")
    torch.manual_seed(0)
    train_A, train_R = e44.gen_graphs(16, 2.0 / 16, N_TRAIN, 0)
    tl = e44.TL(K=4).to(DEVICE)
    e44.train(tl, train_A, train_R, N_STEPS)
    print(f"  TL params after training: alpha={tl.alpha.item():.3f}  beta={tl.beta.item():.3f}  gamma={tl.gamma.item():.3f}")

    print(f"\n{'n':<6}{'p':<8}{'K_tl':<6}{'closure_dens':<14}{'TL F1 (zero-shot)':<22}")
    for n in N_SIZES:
        p = 2.0 / n
        K_tl = max(4, math.ceil(math.log2(max(n, 2))))
        tl.K = K_tl

        tl_f1s = []
        closure_density = None

        for seed in range(N_SEEDS):
            test_A, test_R = e44.gen_graphs(n, p, N_TEST, seed * 1000 + n + 7)
            if closure_density is None:
                closure_density = test_R.float().mean().item()

            f1_tl, _ = e44.evaluate(tl, test_A, test_R)
            tl_f1s.append(f1_tl)

        tl_str = f"{statistics.mean(tl_f1s):.3f}±{statistics.stdev(tl_f1s):.3f}"
        print(f"{n:<6}{p:<8.4f}{K_tl:<6}{closure_density:<14.4f}{tl_str:<22}")


if __name__ == "__main__":
    main()
