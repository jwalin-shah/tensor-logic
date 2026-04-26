"""
exp48: 4-param TL with multiplicative cross-term — does it crack parity?

Recurrence (4 parameters):
    R ← σ(α·(R @ A) + β·R + γ·((R @ A) ⊙ R) + δ)

The cross-term γ·(R@A)·R lets TL place each binary input pair (R@A, R) ∈ {0,1}²
at an independent pre-sigmoid value. This is sufficient for XOR:
    (0,0) → δ
    (1,0) → α + δ
    (0,1) → β + δ
    (1,1) → α + β + γ + δ
With α=4, β=4, γ=-8, δ=-2: pre-sigmoid takes (-2, +2, +2, -2) — XOR shape.
3-param TL cannot do this — only 3 free coefficients for 4 distinct input pairs.

Init at closure operating point (γ=0 → TL_OR4 ≡ TL_OR_3param exactly), so SGD
must swing γ very negative for parity. Test: does it find the XOR operating
point from a closure-init starting point?

Tasks (same as exp45): closure, multipath2, parity. Same n=16, p=0.30 DAGs.
"""

import statistics
import torch
import torch.nn as nn

import exp44_import_closure as e44
import exp45_pathcount_falsify as e45

N_NODES = 16
EDGE_PROB = 0.30
N_TRAIN = 500
N_TEST = 50
N_STEPS = 1000
N_SEEDS = 3
DEVICE = e45.DEVICE


class TL_OR4(nn.Module):
    """4-param TL with multiplicative cross-term."""
    def __init__(self, K=4, alpha=5.0, beta=5.0, gamma_x=0.0, delta=-2.5):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma_x = nn.Parameter(torch.tensor(gamma_x))
        self.delta = nn.Parameter(torch.tensor(delta))

    def forward(self, A):
        R = A.clone()
        for _ in range(self.K):
            comp = torch.einsum("bij,bjk->bik", R, A)
            cross = comp * R
            R = torch.sigmoid(
                self.alpha * comp + self.beta * R + self.gamma_x * cross + self.delta
            )
        return R


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  steps={N_STEPS}  seeds={N_SEEDS}")
    print("Init for TL_OR4: closure operating point (alpha=5, beta=5, gamma_x=0, delta=-2.5)")

    summary = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        A_tr, R_tr, M_tr, P_tr = e45.gen_data(N_NODES, EDGE_PROB, N_TRAIN, seed * 1000)
        A_te, R_te, M_te, P_te = e45.gen_data(N_NODES, EDGE_PROB, N_TEST, seed * 1000 + 7)

        for task, target_tr, target_te in [
            ("closure", R_tr, R_te),
            ("multipath2", M_tr, M_te),
            ("parity", P_tr, P_te),
        ]:
            for model_name in ["TL_OR", "TL_OR4", "MLP"]:
                if model_name == "TL_OR":
                    model = e45.TL_OR().to(DEVICE)
                elif model_name == "TL_OR4":
                    model = TL_OR4().to(DEVICE)
                else:
                    model = e45.MLP(N_NODES, sigmoid_out=True).to(DEVICE)

                e45.train_loop(model, A_tr, target_tr, N_STEPS, loss_kind="bce")
                f1 = e45.eval_closure(model, A_te, target_te)
                summary.setdefault((task, model_name), []).append(f1)

                if model_name == "TL_OR":
                    a, b, g = model.alpha.item(), model.beta.item(), model.gamma.item()
                    extra = f"  (α={a:.2f}, β={b:.2f}, γ={g:.2f})"
                elif model_name == "TL_OR4":
                    a = model.alpha.item()
                    b = model.beta.item()
                    gx = model.gamma_x.item()
                    d = model.delta.item()
                    extra = f"  (α={a:.2f}, β={b:.2f}, γ_X={gx:.2f}, δ={d:.2f})"
                else:
                    extra = ""
                print(f"  seed={seed}  {task:<11} {model_name:<8} F1={f1:.3f}{extra}")

    print("\n=== summary (mean ± std over 3 seeds) ===")
    print(f"{'task':<14}{'model':<10}{'F1':<22}")
    for task in ["closure", "multipath2", "parity"]:
        for model_name in ["TL_OR", "TL_OR4", "MLP"]:
            vs = summary[(task, model_name)]
            print(f"{task:<14}{model_name:<10}{statistics.mean(vs):.3f}±{statistics.stdev(vs):.3f}")


if __name__ == "__main__":
    main()
