"""
exp57: end-to-end joint optimization of A_hat through the TL closure
recurrence (follow-on to exp55/exp56).

Two findings:

(1) Methodology bug in exp55/exp56's `perturb_brain`. The earlier
    function applied a 5% per-cell flip to A_star, which on a sparse
    brain (p=0.025, 86 edges out of 4032 cells) flips ~200 random
    cells — adding ~196 new edges to the 86 existing ones. Subject
    brains ended up ~3x denser than A_star, with closure density
    near 1. This inflated all the "F1 on A_s" numbers in exp55/56.
    Fix: density-preserving perturbation — replace `frac` of A_star's
    edges with a same-count random draw from non-edges. Numbers in
    those experiments need re-running with this fix.

(2) End-to-end joint optimization works. Treat A_hat as a continuous
    learnable parameter in [0,1]^{K×K}, init from soft Granger, and
    optimize (A_hat, α, β, γ) against closure supervision via BCE
    through the K-step TL recurrence. With full R_s supervision,
    F1 = 0.976 (vs init Granger top-k disc F1 = 0.494; chance = 0.217).
    With only 25% of R_s cells visible in the loss, generalization to
    held-out 75% of cells is F1 = 0.692 — well above chance, well
    below full-supervision, so the closure target does carry redundant
    structure but not infinitely.

Three configurations per subject:
  init_disc      — Granger top-k discrete A_hat, no opt (= exp56 setup)
  init_soft      — Granger soft (max-normalized), no opt
                   [degenerate: max-normalization shrinks most values
                    too small to activate β; pred_density collapses to 0]
  joint_full     — joint-optimize (A_hat, α, β, γ) against full R_s
  joint_partial  — same, but only 25% of R_s cells visible in the loss
                   (held-out 75% used purely for evaluation — tests
                   whether closure structure is recoverable from
                   sparse supervision)

This is a test of the MECHANISM. The question of "what real-world
supervision approximates R_s in practice?" is downstream — if the
mechanism didn't work even with idealized supervision, that path
would be dead. It does, so the path is open.

Tier: T1, laptop, ~3-4 minutes.
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
EDGE_PROB_BRAIN = 0.025
SUBJECT_FLIP_FRAC = 0.10  # fraction of A_star edges replaced per subject
N_STIMULI = 200
T_STEPS = 12
GRANGER_RIDGE = 1e-3
N_TRAIN_GRAPHS = 500
N_TEST = 50
N_SUBJECTS = 8
N_TL_PRETRAIN_STEPS = 1500
N_JOINT_STEPS = 800
JOINT_LR = 5e-2
BATCH = 32
LR = 1e-2
K_TL = 4
K_ITERS_BRAIN = int(math.ceil(math.log2(K_BRAIN))) + 4
SUPERVISION_FRAC = 0.25
N_SEEDS = 3
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
    """Density-preserving perturbation: replace `frac_edges_changed` of A_star's
    edges with new edges drawn from non-edge cells. Earlier versions used a
    per-cell flip prob, which inflates density when A_star is sparse (a
    5% per-cell flip on a 2.5%-density brain ~triples the edge count and
    saturates the closure)."""
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


def recover_granger(V, top_k, ridge=GRANGER_RIDGE):
    n_stim, T, K = V.shape
    Vt = V[:, :-1, :].reshape(-1, K)
    Vtp1 = V[:, 1:, :].reshape(-1, K)
    XtX = Vt.t() @ Vt + ridge * torch.eye(K)
    XtY = Vt.t() @ Vtp1
    W = torch.linalg.solve(XtX, XtY)
    return topk_to_adj(W.abs(), top_k), W


class TL(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A, K_iters=K_TL):
        R = A.clone()
        for _ in range(K_iters):
            comp = torch.einsum("bij,bjk->bik", R, A) if R.dim() == 3 else (R @ A)
            R = torch.sigmoid(self.alpha * comp + self.beta * R + self.gamma)
        return R


def pretrain_tl(seed):
    """Train (alpha, beta, gamma) on synthetic K=16 DAGs (same as exp44/55/56)."""
    torch.manual_seed(seed * 13 + 57)
    train_A, train_R = gen_dags(K_TRAIN, EDGE_PROB_TRAIN, N_TRAIN_GRAPHS, seed * 1000)
    tl = TL().to(DEVICE)
    opt = optim.Adam(tl.parameters(), lr=LR)
    n_g = train_A.shape[0]
    for _ in range(N_TL_PRETRAIN_STEPS):
        idx = torch.randint(0, n_g, (min(BATCH, n_g),))
        A = train_A[idx].to(DEVICE)
        R = train_R[idx].to(DEVICE)
        R_pred = tl(A)
        loss = F.binary_cross_entropy(R_pred.clamp(1e-6, 1 - 1e-6), R)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return tl


def joint_optimize(A_init, R_supervise, mask, tl, n_steps=N_JOINT_STEPS, lr=JOINT_LR, freeze_tl=False):
    """Per-subject joint optimization. Returns final A_hat (continuous) + final tl."""
    K = A_init.shape[0]
    A_init_clamped = A_init.clamp(0.02, 0.98)
    theta = torch.log(A_init_clamped / (1 - A_init_clamped)).clone().detach().requires_grad_(True)
    params = [theta] + ([] if freeze_tl else list(tl.parameters()))
    opt = optim.Adam(params, lr=lr)
    eye = torch.eye(K)
    for step in range(n_steps):
        A_hat = torch.sigmoid(theta) * (1 - eye)
        R_pred = tl(A_hat, K_iters=K_ITERS_BRAIN)
        R_clamped = R_pred.clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(R_clamped[mask], R_supervise[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(theta.detach()) * (1 - eye), tl


def f1(pred, target):
    p = (pred > 0.5).float()
    t = (target > 0.5).float()
    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def jaccard(A_hat, A_true):
    p = (A_hat > 0.5).float()
    t = (A_true > 0.5).float()
    inter = ((p == 1) & (t == 1)).sum().item()
    union = ((p == 1) | (t == 1)).sum().item()
    return inter / (union + 1e-9)


def main():
    print(f"device={DEVICE}  K_train={K_TRAIN}  K_brain={K_BRAIN}  N_subjects={N_SUBJECTS}  seeds={N_SEEDS}")

    A_star = latent_brain(K_BRAIN, EDGE_PROB_BRAIN, seed=2026)
    R_star = transitive_closure(A_star)
    closure_density = R_star.mean().item()
    chance_f1 = 2 * closure_density / (1 + closure_density)
    print(f"latent brain: K={K_BRAIN}, edges={int(A_star.sum().item())}, edge_density={A_star.mean().item():.3f}, closure_density={closure_density:.3f}, chance_F1={chance_f1:.3f}")

    top_k_per_row = max(2, int(K_BRAIN * EDGE_PROB_BRAIN))

    subjects = []
    for s in range(N_SUBJECTS):
        A_s = perturb_brain(A_star, SUBJECT_FLIP_FRAC, seed=1000 + s)
        R_s = transitive_closure(A_s)
        V = tribe_stub_voxels(A_s, N_STIMULI, T_STEPS, noise_std=0.0, seed=2000 + s)
        A_init_disc, W_granger = recover_granger(V, top_k_per_row)
        # use the soft Granger weights as init (richer than the discrete top-k)
        W = W_granger.abs()
        W = W / (W.max() + 1e-9)
        A_init = W * (1 - torch.eye(K_BRAIN))
        subjects.append((A_s, R_s, A_init, A_init_disc))

    init_f1_disc = []
    init_f1_soft = []
    init_jac = []
    init_disc_density = []
    init_soft_mean = []
    init_soft_pred_density = []
    joint_full_f1, joint_full_jac, joint_full_pred_density = [], [], []
    joint_part_train_f1, joint_part_held_f1, joint_part_jac = [], [], []
    rng_supervision = random.Random(57)

    for seed in range(N_SEEDS):
        tl = pretrain_tl(seed)
        with torch.no_grad():
            for s_idx, (A_s, R_s, A_init, A_init_disc) in enumerate(subjects):
                R_pred_disc = tl(A_init_disc.unsqueeze(0).to(DEVICE), K_iters=K_ITERS_BRAIN).squeeze(0).cpu()
                R_pred_soft = tl(A_init.unsqueeze(0).to(DEVICE), K_iters=K_ITERS_BRAIN).squeeze(0).cpu()
                init_f1_disc.append(f1(R_pred_disc, R_s))
                init_f1_soft.append(f1(R_pred_soft, R_s))
                init_jac.append(jaccard(A_init_disc, A_s))
                init_disc_density.append(A_init_disc.mean().item())
                init_soft_mean.append(A_init.mean().item())
                init_soft_pred_density.append((R_pred_soft > 0.5).float().mean().item())

        for s_idx, (A_s, R_s, A_init, _) in enumerate(subjects):
            R_s_dev = R_s.to(DEVICE)
            mask_full = torch.ones_like(R_s_dev, dtype=torch.bool)
            tl_clone = TL().to(DEVICE)
            tl_clone.load_state_dict(tl.state_dict())
            A_hat_final, tl_clone = joint_optimize(A_init.to(DEVICE), R_s_dev, mask_full, tl_clone, freeze_tl=False)
            with torch.no_grad():
                R_pred = tl_clone(A_hat_final.unsqueeze(0), K_iters=K_ITERS_BRAIN).squeeze(0).cpu()
            joint_full_f1.append(f1(R_pred, R_s))
            joint_full_jac.append(jaccard(A_hat_final.cpu(), A_s))
            joint_full_pred_density.append((R_pred > 0.5).float().mean().item())

            mask_partial = torch.bernoulli(torch.full(R_s.shape, SUPERVISION_FRAC), generator=torch.Generator().manual_seed(7000 + seed * 100 + s_idx)).bool().to(DEVICE)
            mask_held = ~mask_partial
            tl_clone2 = TL().to(DEVICE)
            tl_clone2.load_state_dict(tl.state_dict())
            A_hat_p, tl_clone2 = joint_optimize(A_init.to(DEVICE), R_s_dev, mask_partial, tl_clone2, freeze_tl=False)
            with torch.no_grad():
                R_pred_p = tl_clone2(A_hat_p.unsqueeze(0), K_iters=K_ITERS_BRAIN).squeeze(0).cpu()
            R_pred_p_dev = R_pred_p.to(DEVICE)
            train_pred = (R_pred_p_dev[mask_partial] > 0.5).float()
            train_tgt = (R_s_dev[mask_partial] > 0.5).float()
            held_pred = (R_pred_p_dev[mask_held] > 0.5).float()
            held_tgt = (R_s_dev[mask_held] > 0.5).float()

            def cell_f1(p, t):
                tp = ((p == 1) & (t == 1)).sum().item()
                fp = ((p == 1) & (t == 0)).sum().item()
                fn = ((p == 0) & (t == 1)).sum().item()
                prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
                return 2 * prec * rec / (prec + rec + 1e-9)

            joint_part_train_f1.append(cell_f1(train_pred, train_tgt))
            joint_part_held_f1.append(cell_f1(held_pred, held_tgt))
            joint_part_jac.append(jaccard(A_hat_p.cpu(), A_s))

        a, b, g = tl.alpha.item(), tl.beta.item(), tl.gamma.item()
        print(f"  seed={seed}  TL pretrained: α={a:.2f} β={b:.2f} γ={g:.2f}")

    def fmt(vs):
        m = statistics.mean(vs)
        s = statistics.stdev(vs) if len(vs) > 1 else 0.0
        return f"{m:.3f} ± {s:.3f}"

    print(f"\n=== summary (mean over {N_SEEDS} seeds × {N_SUBJECTS} subjects) ===")
    print(f"target closure density={closure_density:.3f}")
    print(f"{'condition':<28}{'F1':<18}{'pred_dens':<11}{'A_hat Jac':<14}")
    print(f"{'chance (predict-all-ones)':<28}{chance_f1:<18.3f}{'1.000':<11}{'—':<14}")
    print(f"{'init: Granger top-k disc':<28}{fmt(init_f1_disc):<18}{statistics.mean(init_disc_density):<11.3f}{fmt(init_jac):<14}")
    print(f"{'init: Granger soft (norm)':<28}{fmt(init_f1_soft):<18}{statistics.mean(init_soft_pred_density):<11.3f}{'—':<14}")
    print(f"{'joint, full R_s sup':<28}{fmt(joint_full_f1):<18}{statistics.mean(joint_full_pred_density):<11.3f}{fmt(joint_full_jac):<14}")
    print(f"{'joint, 25% R_s sup (train)':<28}{fmt(joint_part_train_f1):<18}{'—':<11}{'—':<14}")
    print(f"{'joint, 25% R_s sup (held)':<28}{fmt(joint_part_held_f1):<18}{'—':<11}{fmt(joint_part_jac):<14}")
    print(f"\n(joint opt: {N_JOINT_STEPS} steps, lr={JOINT_LR}, A_hat init = soft normalized Granger,")
    print(f" K_iters={K_ITERS_BRAIN}, partial supervision uses {SUPERVISION_FRAC*100:.0f}% of R_s cells.)")


if __name__ == "__main__":
    main()
