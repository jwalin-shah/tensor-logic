"""
exp46: TL on non-monotone graph properties — shortest path distance.

Closes the falsification picture started by exp45. Where exp45 showed TL fails
on parity (XOR / GF(2) semiring), exp46 tests another out-of-monoid class:
non-monotone functions of the adjacency matrix.

TL's recurrence
    R ← σ(α·R@A + β·R + γ)
is monotone in A: if A' ≥ A elementwise, then R'(A') ≥ R(A) (since α, R, A
all ≥ 0 and sigmoid is monotone). But shortest-path distance is non-monotone:
adding the direct edge (i, j) flips 1[SD(i,j)==2] from 1 → 0.

Tasks (binary):
  sd_eq_1  — 1[shortest distance i→j == 1]:  exactly A[i,j]. Trivial — TL/MLP both win.
  sd_eq_2  — 1[shortest distance i→j == 2]:  A²[i,j] > 0 AND A[i,j] = 0. NON-MONOTONE.
  sd_geq_2 — 1[reachable in ≥2 hops, not 1]: closure(A) AND NOT A. NON-MONOTONE,
              moderate density. Cleanest non-monotone test (~19% density at p=0.30).
  sd_geq_3 — 1[shortest distance >= 3 and reachable]:
              closure(A)[i,j] = 1 AND A[i,j] = 0 AND (A²)[i,j] = 0. NON-MONOTONE,
              very sparse (~3% density) — even MLP struggles.

Three models: TL_OR, TL_LIN, MLP. Same n=16, p=0.30, n_train=500, 3 seeds.
Hypothesis: TL_OR fails sd_eq_2 and sd_geq_3 (non-monotone); MLP wins both.
"""

import random
import statistics
import torch
import torch.nn.functional as F

import exp44_import_closure as e44
import exp45_pathcount_falsify as e45

N_NODES = 16
EDGE_PROB = 0.30
N_TRAIN = 500
N_TEST = 50
N_STEPS = 1000
N_SEEDS = 3
DEVICE = e45.DEVICE


def shortest_distances(A):
    """Boolean shortest-hop distance. Returns D where D[i,j] = min hops i→j;
    D[i,i] = 0; D[i,j] = -1 if j unreachable from i."""
    n = A.shape[-1]
    D = torch.full((n, n), -1.0)
    D.fill_diagonal_(0.0)
    A_pow = torch.eye(n)
    for k in range(1, n + 1):
        A_pow = ((A_pow @ A) > 0).float()
        if A_pow.sum() == 0:
            break
        newly = (A_pow > 0) & (D < 0)
        D[newly] = float(k)
    return D


def gen_data_sd(n_nodes, p, n_graphs, seed):
    rng = random.Random(seed)
    As = []
    SD1s, SD2s, SDg2s, SDg3s = [], [], [], []
    for _ in range(n_graphs):
        A = e44.random_dag(n_nodes, p, rng)
        D = shortest_distances(A)
        SD1s.append((D == 1).float())
        SD2s.append((D == 2).float())
        SDg2s.append((D >= 2).float())
        SDg3s.append((D >= 3).float())
        As.append(A)
    return torch.stack(As), torch.stack(SD1s), torch.stack(SD2s), torch.stack(SDg2s), torch.stack(SDg3s)


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  n_train={N_TRAIN}  steps={N_STEPS}  seeds={N_SEEDS}")

    summary = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        A_tr, SD1_tr, SD2_tr, SDg2_tr, SDg3_tr = gen_data_sd(N_NODES, EDGE_PROB, N_TRAIN, seed * 1000)
        A_te, SD1_te, SD2_te, SDg2_te, SDg3_te = gen_data_sd(N_NODES, EDGE_PROB, N_TEST, seed * 1000 + 7)

        if seed == 0:
            print(
                f"  densities: sd_eq_1={SD1_tr.mean().item():.3f}  "
                f"sd_eq_2={SD2_tr.mean().item():.3f}  sd_geq_2={SDg2_tr.mean().item():.3f}  "
                f"sd_geq_3={SDg3_tr.mean().item():.3f}"
            )

        for task, target_tr, target_te in [
            ("sd_eq_1", SD1_tr, SD1_te),
            ("sd_eq_2", SD2_tr, SD2_te),
            ("sd_geq_2", SDg2_tr, SDg2_te),
            ("sd_geq_3", SDg3_tr, SDg3_te),
        ]:
            for model_name in ["TL_OR", "TL_LIN", "MLP"]:
                if model_name == "TL_OR":
                    model = e45.TL_OR().to(DEVICE)
                elif model_name == "TL_LIN":
                    model = e45.TL_LIN().to(DEVICE)
                else:
                    model = e45.MLP(N_NODES, sigmoid_out=True).to(DEVICE)

                if model_name == "TL_LIN":
                    e45.train_loop_lin(model, A_tr, target_tr, N_STEPS)
                    f1 = e45.eval_closure_lin(model, A_te, target_te)
                else:
                    e45.train_loop(model, A_tr, target_tr, N_STEPS, loss_kind="bce")
                    f1 = e45.eval_closure(model, A_te, target_te)
                summary.setdefault((task, model_name), []).append(f1)

                if model_name in ("TL_OR", "TL_LIN"):
                    a, b, g = model.alpha.item(), model.beta.item(), model.gamma.item()
                    extra = f"  (α={a:.2f}, β={b:.2f}, γ={g:.2f})"
                else:
                    extra = ""
                print(f"  seed={seed}  {task:<10} {model_name:<8} F1={f1:.3f}{extra}")

    print("\n=== summary (mean ± std over 3 seeds) ===")
    print(f"{'task':<12}{'model':<10}{'F1':<22}")
    for task in ["sd_eq_1", "sd_eq_2", "sd_geq_2", "sd_geq_3"]:
        for model_name in ["TL_OR", "TL_LIN", "MLP"]:
            vs = summary[(task, model_name)]
            print(f"{task:<12}{model_name:<10}{statistics.mean(vs):.3f}±{statistics.stdev(vs):.3f}")


if __name__ == "__main__":
    main()
