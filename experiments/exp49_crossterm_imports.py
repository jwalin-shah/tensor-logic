"""
exp49: Does the Hadamard cross-term help on real-code transitive closure?

Test TL_OR4 (α, β, γ_X, δ) vs TL_OR3 (α, β, γ) on exp44's import-closure tasks
(random DAGs at p=0.12, plus the real tensor-project import graph zero-shot).

Hypothesis: code-closure is monotone/OR-like, so γ_X should stay near 0 and
TL_OR4 ≈ TL_OR3. If γ_X moves significantly, code has non-monotone substructure
worth investigating.
"""
import statistics
import torch
import torch.nn as nn

import exp44_import_closure as e44

DEVICE = e44.DEVICE
N_NODES = e44.N_NODES
EDGE_PROB = e44.EDGE_PROB
N_TEST = e44.N_TEST
N_STEPS = e44.N_STEPS
N_SEEDS = e44.N_SEEDS


class TL_OR4(nn.Module):
    def __init__(self, K=e44.K_TL):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma_x = nn.Parameter(torch.tensor(0.0))
        self.delta = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A):
        R = A.clone()
        for _ in range(self.K):
            comp = torch.einsum("bij,bjk->bik", R, A)
            cross = comp * R
            R = torch.sigmoid(self.alpha * comp + self.beta * R + self.gamma_x * cross + self.delta)
        return R


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  steps={N_STEPS}  seeds={N_SEEDS}")
    real_A, real_R = e44.real_graph_padded(N_NODES)
    print(f"real graph closure density={real_R.sum().item()/(N_NODES*N_NODES):.3f}")

    n_train_list = [20, 100, 500]
    summary = {}
    for n_train in n_train_list:
        for model_name in ["TL3", "TL4"]:
            run = {"in_dist": [], "real": [], "ood_sparse": [], "ood_dense": [], "gamma_x": []}
            for seed in range(N_SEEDS):
                torch.manual_seed(seed * 13 + n_train)
                tA, tR = e44.gen_graphs(N_NODES, EDGE_PROB, n_train, seed*1000+n_train)
                teA, teR = e44.gen_graphs(N_NODES, EDGE_PROB, N_TEST, seed*1000+n_train+7)
                osA, osR = e44.gen_graphs(N_NODES, 0.05, N_TEST, seed*1000+n_train+11)
                odA, odR = e44.gen_graphs(N_NODES, 0.25, N_TEST, seed*1000+n_train+17)

                model = (e44.TL() if model_name == "TL3" else TL_OR4()).to(DEVICE)
                e44.train(model, tA, tR, N_STEPS)
                f1_id, _ = e44.evaluate(model, teA, teR)
                f1_re, _ = e44.evaluate(model, real_A, real_R)
                f1_os, _ = e44.evaluate(model, osA, osR)
                f1_od, _ = e44.evaluate(model, odA, odR)
                run["in_dist"].append(f1_id); run["real"].append(f1_re)
                run["ood_sparse"].append(f1_os); run["ood_dense"].append(f1_od)

                if model_name == "TL4":
                    a, b, gx, d = model.alpha.item(), model.beta.item(), model.gamma_x.item(), model.delta.item()
                    run["gamma_x"].append(gx)
                    extra = f"  α={a:.2f} β={b:.2f} γ_X={gx:.2f} δ={d:.2f}"
                else:
                    a, b, g = model.alpha.item(), model.beta.item(), model.gamma.item()
                    extra = f"  α={a:.2f} β={b:.2f} γ={g:.2f}"
                print(f"  n_train={n_train:<4} {model_name} seed={seed}  ID={f1_id:.3f} real={f1_re:.3f} ood_sp={f1_os:.3f} ood_de={f1_od:.3f}{extra}")

            summary[(n_train, model_name)] = run

    print("\n=== summary (mean ± std over 3 seeds) ===")
    print(f"{'n_train':<8}{'model':<6}{'in_dist':<14}{'real':<14}{'ood_sparse':<14}{'ood_dense':<14}{'γ_X':<10}")
    for n_train in n_train_list:
        for model_name in ["TL3", "TL4"]:
            run = summary[(n_train, model_name)]
            row = f"{n_train:<8}{model_name:<6}"
            for k in ["in_dist", "real", "ood_sparse", "ood_dense"]:
                m = statistics.mean(run[k]); s = statistics.stdev(run[k])
                row += f"{m:.3f}±{s:.3f}  "
            if run["gamma_x"]:
                gx_m = statistics.mean(run["gamma_x"]); gx_s = statistics.stdev(run["gamma_x"])
                row += f"{gx_m:+.2f}±{gx_s:.2f}"
            print(row)


if __name__ == "__main__":
    main()
