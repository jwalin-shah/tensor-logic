"""
exp48 verification: does the 4-param TL_OR4 cross-term actually compute XOR?

Mirrors the verification that surfaced after the original exp48 commit
mis-celebrated the parity result. Three conditions on the parity task:

  xor_init_frozen — handcrafted weights (α=4, β=4, γ_X=-8, δ=-2). NO TRAINING.
                    The "XOR operating point" the original exp48 docstring claimed
                    would compute parity. If the architecture truly expresses XOR,
                    these weights should give F1≈1 on parity. They don't.

  xor_init_train  — same init, then SGD-trained for parity.
                    Tells us where SGD ends up when starting from "XOR".

  rand_init       — uniform-random α, β, γ_X, δ ∈ [-2, 2], then SGD-trained.
                    Tells us if exp48's reported F1≈0.685 needs the privileged
                    closure-shaped init or comes for free.

Honest expectation (correcting the original exp48 framing):
  Frozen XOR-init scores well below 1.0 because σ on real numbers cannot
  compose K=4 iterations into mod-2 path-count parity. The "operating point"
  argument was per-cell at iteration-1; it doesn't survive iteration. SGD
  drifts away from XOR-init to a closure-attenuated basin that happens to
  score better on parity than pure closure — but it isn't XOR.
"""

import random
import statistics
import torch

import exp45_pathcount_falsify as e45
from exp48_crossterm_xor import TL_OR4

N_NODES = 16
EDGE_PROB = 0.30
N_TRAIN = 500
N_TEST = 50
N_STEPS = 1000
N_SEEDS = 3
DEVICE = e45.DEVICE


def show_lookup_table(alpha, beta, gamma_x, delta):
    """Print pre-sigmoid and sigmoid for each binary (R@A, R) pair."""
    print(f"  alpha={alpha:.2f} beta={beta:.2f} gamma_X={gamma_x:.2f} delta={delta:.2f}")
    for ra in (0, 1):
        for r in (0, 1):
            pre = alpha * ra + beta * r + gamma_x * ra * r + delta
            post = 1 / (1 + 2.71828 ** (-pre))
            print(f"  (R@A={ra}, R={r})  pre={pre:+.2f}  σ={post:.3f}")


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  steps={N_STEPS}  seeds={N_SEEDS}")

    print("\nXOR operating point lookup (per-cell, single iteration):")
    show_lookup_table(4.0, 4.0, -8.0, -2.0)
    print("→ pre-sigmoid is XOR-shaped (-2, +2, +2, -2). σ values are (0.12, 0.88, 0.88, 0.12) — already not Boolean.")

    rng = random.Random(0)
    summary = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        A_tr, _, _, P_tr = e45.gen_data(N_NODES, EDGE_PROB, N_TRAIN, seed * 1000)
        A_te, _, _, P_te = e45.gen_data(N_NODES, EDGE_PROB, N_TEST, seed * 1000 + 7)

        # 1. Frozen XOR-init
        m_frozen = TL_OR4(alpha=4.0, beta=4.0, gamma_x=-8.0, delta=-2.0).to(DEVICE)
        for p in m_frozen.parameters():
            p.requires_grad_(False)
        f1_frozen = e45.eval_closure(m_frozen, A_te, P_te)

        # 2. XOR init + train
        m_train_xor = TL_OR4(alpha=4.0, beta=4.0, gamma_x=-8.0, delta=-2.0).to(DEVICE)
        e45.train_loop(m_train_xor, A_tr, P_tr, N_STEPS, loss_kind="bce")
        f1_train_xor = e45.eval_closure(m_train_xor, A_te, P_te)

        # 3. Random init + train
        a0 = rng.uniform(-2, 2)
        b0 = rng.uniform(-2, 2)
        g0 = rng.uniform(-2, 2)
        d0 = rng.uniform(-2, 2)
        m_rand = TL_OR4(alpha=a0, beta=b0, gamma_x=g0, delta=d0).to(DEVICE)
        e45.train_loop(m_rand, A_tr, P_tr, N_STEPS, loss_kind="bce")
        f1_rand = e45.eval_closure(m_rand, A_te, P_te)

        summary.setdefault("xor_init_frozen", []).append(f1_frozen)
        summary.setdefault("xor_init_train", []).append(f1_train_xor)
        summary.setdefault("rand_init", []).append(f1_rand)

        a, b, g, d = m_train_xor.alpha.item(), m_train_xor.beta.item(), m_train_xor.gamma_x.item(), m_train_xor.delta.item()
        ar, br, gr, dr = m_rand.alpha.item(), m_rand.beta.item(), m_rand.gamma_x.item(), m_rand.delta.item()
        print(
            f"\nseed={seed}\n"
            f"  xor_init_frozen  parity F1={f1_frozen:.3f}\n"
            f"  xor_init_train   parity F1={f1_train_xor:.3f}  trained=(α={a:.2f}, β={b:.2f}, γ_X={g:.2f}, δ={d:.2f})\n"
            f"  rand_init        parity F1={f1_rand:.3f}  init=(α={a0:.2f}, β={b0:.2f}, γ_X={g0:.2f}, δ={d0:.2f})  trained=(α={ar:.2f}, β={br:.2f}, γ_X={gr:.2f}, δ={dr:.2f})"
        )

    print("\n=== summary (mean ± std over 3 seeds) ===")
    for k in ["xor_init_frozen", "xor_init_train", "rand_init"]:
        vs = summary[k]
        print(f"  {k:<20}  parity F1 = {statistics.mean(vs):.3f} ± {statistics.stdev(vs):.3f}")


if __name__ == "__main__":
    main()
