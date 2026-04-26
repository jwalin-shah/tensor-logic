"""
exp45: Falsification test for the exp44 inductive-bias claim.

If TL wins exp44 because its einsum recurrence matches transitive-closure
structure (Boolean OR over matrix powers), then on a task whose structure
DOES NOT match — counting distinct paths (sum-over-paths semiring) — TL
should LOSE. Validates that the exp44 win is structure-matching, not
"TL universally beats MLP".

v1 used regression on log(1+C) at p=0.12 — TL still won by a small margin
because the dominant signal in that target is "is there any path?" (closure).
v2 isolates counting from closure with two tasks at denser graphs (p=0.30):

  closure    — R[i,j] = 1 iff there exists ≥1 i→j path.    (TL's home turf.)
  multipath2 — M[i,j] = 1 iff there exist ≥2 distinct paths from i to j.
               Closure-equivalent cells (C=1) are NEGATIVE in this task.
  parity     — P[i,j] = (# distinct i→j paths) mod 2.
               XOR over compositions, GF(2) matrix arithmetic. Sigmoid-thresholded
               sum cannot represent this — saturation kills the parity bit.

Three models:
  TL_OR  — sigmoid output: R ← σ(α·R@A + β·R + γ).  3 params. (exp44 design.)
  TL_LIN — linear output : R ← α·R@A + β·R + γ.    3 params. (No sigmoid.)
  MLP    — 2-hidden h=64, ~37k params.

BCE loss on each binary task. F1 metric.

Why TL_OR should fail multipath2: after K=1 iteration, R@A at cell (i,j) IS
the number of distinct 2-hop routes — the count signal exists. But sigmoid
with sharp parameters saturates at 1 for any count ≥1, destroying the
distinction between "1 path" and "≥2 paths". Across K iterations the count
info is fully erased.
"""

import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import exp44_import_closure as e44

N_NODES = 16
EDGE_PROB = 0.30
N_TRAIN = 500
N_TEST = 50
N_STEPS = 1000
BATCH = 32
LR = 1e-2
K_TL = 4
H_MLP = 64
N_SEEDS = 3
DEVICE = e44.DEVICE


def path_count(A):
    """C[i,j] = # distinct i→j paths in DAG. C = I + A + A^2 + ... ."""
    n = A.shape[-1]
    I = torch.eye(n)
    C = I.clone()
    Ak = I.clone()
    for _ in range(n):
        Ak = Ak @ A
        if Ak.abs().sum() < 1e-9:
            break
        C = C + Ak
    return C


def gen_data(n_nodes, p, n_graphs, seed):
    rng = random.Random(seed)
    As, Rs, Ms, Ps = [], [], [], []  # R=closure, M=multipath2, P=parity
    for _ in range(n_graphs):
        A = e44.random_dag(n_nodes, p, rng)
        R = e44.transitive_closure(A)
        C = path_count(A)
        M = (C >= 2).float()
        P = (C.long() % 2).float()
        As.append(A)
        Rs.append(R)
        Ms.append(M)
        Ps.append(P)
    return torch.stack(As), torch.stack(Rs), torch.stack(Ms), torch.stack(Ps)


class TL_OR(nn.Module):
    def __init__(self, K=K_TL):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A):
        R = A.clone()
        for _ in range(self.K):
            comp = torch.einsum("bij,bjk->bik", R, A)
            R = torch.sigmoid(self.alpha * comp + self.beta * R + self.gamma)
        return R


class TL_LIN(nn.Module):
    """Same recurrence shape, no sigmoid. Target-matched init for path count.

    For path count C = I + A + A^2 + ..., the ideal recurrence is
        C_{k+1} = I + C_k @ A,   C_0 = A
    so a "natural" init is alpha=1 (compose), beta=1 (preserve), gamma=0.
    But our recurrence is `R ← alpha*R@A + beta*R + gamma`, which doesn't
    have a per-cell I term. The linear TL can approximate via beta carrying
    the I forward, but the shape is fundamentally one-off. That's the point.
    """
    def __init__(self, K=K_TL):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, A):
        R = A.clone()
        for _ in range(self.K):
            comp = torch.einsum("bij,bjk->bik", R, A)
            R = self.alpha * comp + self.beta * R + self.gamma
        return R


class MLP(nn.Module):
    def __init__(self, n, h=H_MLP, sigmoid_out=False):
        super().__init__()
        self.n = n
        self.sigmoid_out = sigmoid_out
        self.net = nn.Sequential(
            nn.Linear(n * n, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, n * n),
        )

    def forward(self, A):
        x = self.net(A.flatten(-2)).reshape(*A.shape)
        return torch.sigmoid(x) if self.sigmoid_out else x


def train_loop(model, A_train, target_train, n_steps, loss_kind, lr=LR, batch=BATCH):
    opt = optim.Adam(model.parameters(), lr=lr)
    n_g = A_train.shape[0]
    for _ in range(n_steps):
        idx = torch.randint(0, n_g, (min(batch, n_g),))
        A = A_train[idx].to(DEVICE)
        T = target_train[idx].to(DEVICE)
        pred = model(A)
        if loss_kind == "bce":
            loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), T)
        else:
            loss = F.mse_loss(pred, T)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def eval_closure(model, A, R, threshold=0.5):
    model.eval()
    A = A.to(DEVICE)
    R = R.to(DEVICE)
    pred = model(A)
    pred_bin = (pred > threshold).float()
    tp = ((pred_bin == 1) & (R == 1)).sum().item()
    fp = ((pred_bin == 1) & (R == 0)).sum().item()
    fn = ((pred_bin == 0) & (R == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    model.train()
    return f1


def train_loop_lin(model, A_train, target_train, n_steps, lr=LR, batch=BATCH):
    """For TL_LIN: wraps the model's unbounded output in sigmoid before BCE."""
    opt = optim.Adam(model.parameters(), lr=lr)
    n_g = A_train.shape[0]
    for _ in range(n_steps):
        idx = torch.randint(0, n_g, (min(batch, n_g),))
        A = A_train[idx].to(DEVICE)
        T = target_train[idx].to(DEVICE)
        pred = torch.sigmoid(model(A))
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), T)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def eval_closure_lin(model, A, target, threshold=0.5):
    model.eval()
    A = A.to(DEVICE)
    T = target.to(DEVICE)
    pred = torch.sigmoid(model(A))
    pred_bin = (pred > threshold).float()
    tp = ((pred_bin == 1) & (T == 1)).sum().item()
    fp = ((pred_bin == 1) & (T == 0)).sum().item()
    fn = ((pred_bin == 0) & (T == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    model.train()
    return f1


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  n_train={N_TRAIN}  steps={N_STEPS}  seeds={N_SEEDS}")

    summary = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        A_tr, R_tr, M_tr, P_tr = gen_data(N_NODES, EDGE_PROB, N_TRAIN, seed * 1000)
        A_te, R_te, M_te, P_te = gen_data(N_NODES, EDGE_PROB, N_TEST, seed * 1000 + 7)

        if seed == 0:
            print(
                f"  closure density={(R_tr.float().mean().item()):.3f}  "
                f"multipath2 density={(M_tr.float().mean().item()):.3f}  "
                f"parity density={(P_tr.float().mean().item()):.3f}"
            )

        for task, target_tr, target_te in [
            ("closure", R_tr, R_te),
            ("multipath2", M_tr, M_te),
            ("parity", P_tr, P_te),
        ]:
            for model_name in ["TL_OR", "TL_LIN", "MLP"]:
                if model_name == "TL_OR":
                    model = TL_OR().to(DEVICE)
                elif model_name == "TL_LIN":
                    model = TL_LIN().to(DEVICE)
                else:
                    model = MLP(N_NODES, sigmoid_out=True).to(DEVICE)

                # Wrap TL_LIN's output in sigmoid for BCE training stability
                # by appending a final sigmoid in the loss path.
                if model_name == "TL_LIN":
                    train_loop_lin(model, A_tr, target_tr, N_STEPS)
                    f1 = eval_closure_lin(model, A_te, target_te)
                else:
                    train_loop(model, A_tr, target_tr, N_STEPS, loss_kind="bce")
                    f1 = eval_closure(model, A_te, target_te)
                summary.setdefault((task, model_name), []).append(f1)

                if model_name in ("TL_OR", "TL_LIN"):
                    a, b, g = model.alpha.item(), model.beta.item(), model.gamma.item()
                    extra = f"  (α={a:.2f}, β={b:.2f}, γ={g:.2f})"
                else:
                    extra = ""
                print(f"  seed={seed}  {task:<11} {model_name:<8} F1={f1:.3f}{extra}")

    print("\n=== summary (mean ± std over 3 seeds) ===")
    print(f"{'task':<14}{'model':<10}{'F1':<22}")
    for task in ["closure", "multipath2", "parity"]:
        for model_name in ["TL_OR", "TL_LIN", "MLP"]:
            vs = summary[(task, model_name)]
            print(f"{task:<14}{model_name:<10}{statistics.mean(vs):.3f}±{statistics.stdev(vs):.3f}")


if __name__ == "__main__":
    main()
