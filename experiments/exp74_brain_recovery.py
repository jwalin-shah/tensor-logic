"""
exp74: better adjacency recovery from voxel timeseries (follow-on to exp73).

exp73 reported "TL on ground-truth A_s F1=1.000, on recovered A_hat
F1=0.347" at brain edge_prob=0.08. This experiment digs in, and finds
that exp73's framing was partly metric-saturated: at p=0.08 the
ground-truth closure density is **0.984**, so predict-all-ones already
scores F1≈0.992. The interesting regime is sparser brains where F1
actually discriminates between recovery methods.

Run at p=0.025 (closure density ~0.12, chance F1 ~0.22). Three
directional recovery methods, top-k thresholded per row:
  lag1     — exp73's baseline. score[i,j] = sum_{stim, t} V[t,i] * V[t+1,j]
  lagK     — sum of lag-1..lag-L cross-correlations
  granger  — linear regression V[t+1] = V[t] @ W with ridge; |W[i,j]| as score

Plus a corruption sweep: take A_s and randomly flip a fraction f of cells,
report Jaccard + TL F1 vs f. Gives the "how good does recovery need to
be?" curve independent of the recovery method.

The TL recurrence is identical to exp44/55 (3 scalars), trained on K=16
random DAGs. K_brain=64. Stub generative model is the same as exp73.

Tier: T1, laptop, ~3 minutes.
"""

import math
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

K_TRAIN = 16
K_BRAIN = 64
EDGE_PROB_TRAIN = 0.12
# Sparser than exp73 (which used 0.08): at 0.08 the closure density is
# 0.984 and predict-all-ones already gets F1≈0.99 — the metric is
# uninformative. At 0.025 closure density ≈0.12 and chance F1 ≈0.22,
# so F1 actually discriminates between recovery methods.
EDGE_PROB_BRAIN = 0.025
SUBJECT_FLIP_FRAC = 0.10  # density-preserving: replace 10% of A_star's edges per subject
N_STIMULI = 200
T_STEPS = 12
LAG_K = 4
GRANGER_RIDGE = 1e-3
N_TRAIN_GRAPHS = 500
N_TEST = 50
N_SUBJECTS = 8
N_STEPS = 1500
BATCH = 32
LR = 1e-2
K_TL = 4
N_SEEDS = 3
CORRUPTION_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def random_dag(n, p, rng):
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                A[i, j] = 1.0
    return A


def random_directed(n, p, rng):
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                A[i, j] = 1.0
    return A


def transitive_closure(A):
    R = A.clone()
    n = A.shape[-1]
    for _ in range(int(math.ceil(math.log2(max(n, 2)))) + 4):
        R_new = ((R @ R + R) > 0).float()
        if torch.equal(R_new, R):
            break
        R = R_new
    return R


def gen_dags(n_nodes, p, n_graphs, seed):
    rng = random.Random(seed)
    As, Rs = [], []
    for _ in range(n_graphs):
        A = random_dag(n_nodes, p, rng)
        Rs.append(transitive_closure(A))
        As.append(A)
    return torch.stack(As), torch.stack(Rs)


def latent_brain(K, p, seed):
    rng = random.Random(seed)
    return random_directed(K, p, rng)


def perturb_brain(A_star, frac_edges_changed, seed):
    """Density-preserving per-subject perturbation (see exp75 docstring)."""
    rng = random.Random(seed)
    A = A_star.clone()
    K = A.shape[0]
    edges = [(i, j) for i in range(K) for j in range(K) if i != j and A[i, j] == 1.0]
    non_edges = [(i, j) for i in range(K) for j in range(K) if i != j and A[i, j] == 0.0]
    n_change = max(1, int(round(frac_edges_changed * len(edges))))
    rng.shuffle(edges)
    rng.shuffle(non_edges)
    for i, j in edges[:n_change]:
        A[i, j] = 0.0
    for i, j in non_edges[:n_change]:
        A[i, j] = 1.0
    return A


def tribe_stub_voxels(A, n_stimuli, t_steps, noise_std, seed):
    K = A.shape[0]
    rng = torch.Generator().manual_seed(seed)
    stim = torch.bernoulli(torch.full((n_stimuli, K), 0.10), generator=rng)
    V = torch.zeros(n_stimuli, t_steps, K)
    v = stim.clone()
    for t in range(t_steps):
        drive = stim if t < 3 else torch.zeros_like(stim)
        v = torch.tanh(0.6 * (v @ A) + 0.4 * v + drive)
        if noise_std > 0:
            v = v + noise_std * torch.randn(v.shape, generator=rng)
        V[:, t, :] = v
    return V


def topk_to_adj(score, top_k):
    K = score.shape[0]
    A = torch.zeros(K, K)
    score = score.clone()
    score.fill_diagonal_(float("-inf"))
    _, topi = torch.topk(score, k=min(top_k, K - 1), dim=1)
    for i in range(K):
        for j in topi[i].tolist():
            A[i, j] = 1.0
    return A


def recover_lag1(V, top_k):
    Vi = V[:, :-1, :]
    Vj = V[:, 1:, :]
    score = torch.einsum("sti,stj->ij", Vi, Vj)
    return topk_to_adj(score, top_k)


def recover_lagK(V, lag_max, top_k):
    n_stim, T, K = V.shape
    score = torch.zeros(K, K)
    for lag in range(1, lag_max + 1):
        if lag >= T:
            break
        Vi = V[:, :-lag, :]
        Vj = V[:, lag:, :]
        score = score + torch.einsum("sti,stj->ij", Vi, Vj)
    return topk_to_adj(score, top_k)


def recover_granger(V, top_k, ridge=GRANGER_RIDGE):
    """Linear regression V[t+1] = V[t] @ W; |W[i,j]| ranks i->j evidence."""
    n_stim, T, K = V.shape
    Vt = V[:, :-1, :].reshape(-1, K)
    Vtp1 = V[:, 1:, :].reshape(-1, K)
    XtX = Vt.t() @ Vt + ridge * torch.eye(K)
    XtY = Vt.t() @ Vtp1
    W = torch.linalg.solve(XtX, XtY)
    return topk_to_adj(W.abs(), top_k)


def corrupt_brain(A, fraction, seed):
    """Flip a `fraction` of off-diagonal cells uniformly at random."""
    rng = random.Random(seed)
    K = A.shape[0]
    pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
    n_flip = int(round(fraction * len(pairs)))
    rng.shuffle(pairs)
    A_c = A.clone()
    for i, j in pairs[:n_flip]:
        A_c[i, j] = 1.0 - A_c[i, j]
    return A_c


class TL(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A, K_iters=K_TL):
        R = A.clone()
        for _ in range(K_iters):
            comp = torch.einsum("bij,bjk->bik", R, A)
            R = torch.sigmoid(self.alpha * comp + self.beta * R + self.gamma)
        return R


def train_tl(model, train_A, train_R, n_steps=N_STEPS, lr=LR, batch=BATCH):
    opt = optim.Adam(model.parameters(), lr=lr)
    n_g = train_A.shape[0]
    for _ in range(n_steps):
        idx = torch.randint(0, n_g, (min(batch, n_g),))
        A = train_A[idx].to(DEVICE)
        R = train_R[idx].to(DEVICE)
        R_pred = model(A)
        loss = F.binary_cross_entropy(R_pred.clamp(1e-6, 1 - 1e-6), R)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def f1_closure(model, A, R_true, K_iters):
    model.eval()
    R_pred = model(A.to(DEVICE), K_iters=K_iters)
    pred = (R_pred > 0.5).float().cpu()
    tp = ((pred == 1) & (R_true == 1)).sum().item()
    fp = ((pred == 1) & (R_true == 0)).sum().item()
    fn = ((pred == 0) & (R_true == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    model.train()
    return 2 * prec * rec / (prec + rec + 1e-9)


def jaccard(A_hat, A_true):
    inter = ((A_hat == 1) & (A_true == 1)).sum().item()
    union = ((A_hat == 1) | (A_true == 1)).sum().item()
    return inter / (union + 1e-9)


def main():
    print(f"device={DEVICE}  K_train={K_TRAIN}  K_brain={K_BRAIN}  N_subjects={N_SUBJECTS}  seeds={N_SEEDS}")

    A_star = latent_brain(K_BRAIN, EDGE_PROB_BRAIN, seed=2026)
    R_star = transitive_closure(A_star)
    closure_density_star = R_star.sum().item() / (K_BRAIN * K_BRAIN)
    K_iters_brain = int(math.ceil(math.log2(K_BRAIN))) + 4
    top_k_per_row = int(K_BRAIN * EDGE_PROB_BRAIN)
    print(f"latent brain: K={K_BRAIN}, edges={int(A_star.sum().item())}, edge_density={A_star.mean().item():.3f}, closure_density={closure_density_star:.3f}, top_k={top_k_per_row}, K_iters_brain={K_iters_brain}")

    # Build per-subject (A_s, V_clean, V_noisy, R_s) once.
    subjects = []
    for s in range(N_SUBJECTS):
        A_s = perturb_brain(A_star, SUBJECT_FLIP_FRAC, seed=1000 + s)
        R_s = transitive_closure(A_s)
        V_clean = tribe_stub_voxels(A_s, N_STIMULI, T_STEPS, noise_std=0.0, seed=2000 + s)
        V_noisy = tribe_stub_voxels(A_s, N_STIMULI, T_STEPS, noise_std=0.20, seed=3000 + s)
        subjects.append((A_s, R_s, V_clean, V_noisy))

    methods = {
        "lag1":    lambda V: recover_lag1(V, top_k_per_row),
        "lagK":    lambda V: recover_lagK(V, LAG_K, top_k_per_row),
        "granger": lambda V: recover_granger(V, top_k_per_row),
    }

    # Pre-compute recovered adjacencies (don't depend on TL seed).
    rec_clean = {m: [] for m in methods}
    rec_noisy = {m: [] for m in methods}
    jac_clean = {m: [] for m in methods}
    jac_noisy = {m: [] for m in methods}
    for A_s, _, V_clean, V_noisy in subjects:
        for name, fn in methods.items():
            A_hat_clean = fn(V_clean)
            A_hat_noisy = fn(V_noisy)
            rec_clean[name].append(A_hat_clean)
            rec_noisy[name].append(A_hat_noisy)
            jac_clean[name].append(jaccard(A_hat_clean, A_s))
            jac_noisy[name].append(jaccard(A_hat_noisy, A_s))

    print("\n=== adjacency recovery: Jaccard vs A_s, closure density of A_hat ===")
    print(f"{'method':<10}{'jac_clean':<12}{'jac_noisy':<12}{'rho_clean':<12}{'rho_noisy':<12}")
    for name in methods:
        jc = statistics.mean(jac_clean[name])
        jn = statistics.mean(jac_noisy[name])
        rho_c = statistics.mean(transitive_closure(A_hat).mean().item() for A_hat in rec_clean[name])
        rho_n = statistics.mean(transitive_closure(A_hat).mean().item() for A_hat in rec_noisy[name])
        print(f"{name:<10}{jc:<12.3f}{jn:<12.3f}{rho_c:<12.3f}{rho_n:<12.3f}")
    print(f"(closure density of true brain A_s ≈ {closure_density_star:.3f}; chance F1 ≈ {2*closure_density_star/(1+closure_density_star):.3f})")

    A_gt = torch.stack([s[0] for s in subjects])
    R_gt = torch.stack([s[1] for s in subjects])

    # ------ corruption sweep: A_s with f random flips ------
    corrupted = {}
    for f in CORRUPTION_FRACTIONS:
        A_corr_list = []
        jac_list = []
        for s_idx, (A_s, _, _, _) in enumerate(subjects):
            A_c = corrupt_brain(A_s, f, seed=4000 + s_idx + int(f * 1000))
            A_corr_list.append(A_c)
            jac_list.append(jaccard(A_c, A_s))
        corrupted[f] = (torch.stack(A_corr_list), statistics.mean(jac_list))

    # ------ train TL across seeds, evaluate everything ------
    summary_methods_clean = {m: [] for m in methods}
    summary_methods_noisy = {m: [] for m in methods}
    summary_gt = []
    summary_corrupted = {f: [] for f in CORRUPTION_FRACTIONS}
    summary_in_dist = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 13 + 56)
        train_A, train_R = gen_dags(K_TRAIN, EDGE_PROB_TRAIN, N_TRAIN_GRAPHS, seed * 1000)
        test_A, test_R = gen_dags(K_TRAIN, EDGE_PROB_TRAIN, N_TEST, seed * 1000 + 7)

        tl = TL().to(DEVICE)
        train_tl(tl, train_A, train_R)

        f1_id = f1_closure(tl, test_A, test_R, K_iters=K_TL)
        f1_gt = f1_closure(tl, A_gt, R_gt, K_iters=K_iters_brain)
        summary_in_dist.append(f1_id)
        summary_gt.append(f1_gt)

        for name in methods:
            A_hat_c = torch.stack(rec_clean[name])
            A_hat_n = torch.stack(rec_noisy[name])
            summary_methods_clean[name].append(f1_closure(tl, A_hat_c, R_gt, K_iters=K_iters_brain))
            summary_methods_noisy[name].append(f1_closure(tl, A_hat_n, R_gt, K_iters=K_iters_brain))

        for f in CORRUPTION_FRACTIONS:
            A_corr, _ = corrupted[f]
            summary_corrupted[f].append(f1_closure(tl, A_corr, R_gt, K_iters=K_iters_brain))

        a, b, g = tl.alpha.item(), tl.beta.item(), tl.gamma.item()
        print(f"  seed={seed}  TL(α={a:.2f} β={b:.2f} γ={g:.2f})  in_dist={f1_id:.3f}  tribe_gt={f1_gt:.3f}")

    def fmt(vs):
        m = statistics.mean(vs)
        s = statistics.stdev(vs) if len(vs) > 1 else 0.0
        return f"{m:.3f} ± {s:.3f}"

    print(f"\n=== TL closure F1 by recovery method (mean over {N_SEEDS} seeds × {N_SUBJECTS} subjects) ===")
    print(f"{'condition':<22}{'F1':<18}{'Jaccard':<10}")
    print(f"{'in_dist (K=16)':<22}{fmt(summary_in_dist):<18}{'—':<10}")
    print(f"{'tribe_gt':<22}{fmt(summary_gt):<18}{'1.000':<10}")
    for name in methods:
        jc = statistics.mean(jac_clean[name])
        jn = statistics.mean(jac_noisy[name])
        print(f"{f'  {name}_clean':<22}{fmt(summary_methods_clean[name]):<18}{jc:<10.3f}")
        print(f"{f'  {name}_noisy':<22}{fmt(summary_methods_noisy[name]):<18}{jn:<10.3f}")

    print(f"\n=== corruption sweep: A_s with f random flips → TL F1 ===")
    print(f"{'flip_frac':<12}{'Jaccard':<12}{'TL F1':<18}")
    for f in CORRUPTION_FRACTIONS:
        _, jc = corrupted[f]
        print(f"{f:<12.2f}{jc:<12.3f}{fmt(summary_corrupted[f]):<18}")


if __name__ == "__main__":
    main()
