"""
exp73: TL transitive-closure recurrence applied to TRIBE-style voxel data.

Context. Meta released TRIBE v2 (March 2026), a tri-modal foundation model
that predicts fMRI voxel responses (~70k voxels) from video/audio/text
stimuli. This experiment asks the natural follow-on question for this
repo: can the 3-scalar TL closure recurrence
    R_{k+1} = sigmoid(alpha * R_k @ A + beta * R_k + gamma)
that worked zero-shot on real Python import graphs (exp44, exp53, exp54)
also operate over a brain connectivity graph derived from TRIBE-style
voxel timeseries?

We don't have TRIBE weights wired up here. Instead we use a faithful
*stub* generative model:
  - latent connectivity A* (K x K, sparse directed, with optional cycles)
  - per-subject perturbation A_s = A* with random edge flips
  - "voxel timeseries" V[time, K] = run several steps of a
    saturating linear dynamical system seeded by random stimuli on A_s
  - estimated adjacency A_hat from co-activation across stimuli (lagged
    correlation, top-k thresholded per row)

The TL recurrence is trained ONLY on synthetic random DAGs at K=16 (the
same training distribution as exp44), then evaluated on:
  in_dist        — held-out random DAGs at K=16
  tribe_gt       — A_s itself (the ground-truth brain connectivity)
  tribe_hat      — A_hat recovered from voxel timeseries (the realistic case)
  tribe_hat_noisy — A_hat with measurement noise added to V

The headline question: does TL closure on A_hat recover the reachability
structure of A_s? If it does at K=64, the same recipe should work when
TRIBE weights are dropped in (just replace `tribe_stub_voxels` with a real
TRIBE call). If it fails on the stub, it will fail harder on real fMRI.

Compare against an MLP (~37k params, trained at K=16) cropped to K=16 for
the K=16 eval and reported as "no native answer" at K=64 (same caveat as
exp53/exp54 for variable-size graphs).

Tier: T1, laptop, ~2 minutes. Pure CPU, no external downloads.
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
EDGE_PROB_BRAIN = 0.08
SUBJECT_FLIP_FRAC = 0.10  # density-preserving: replace 10% of A_star's edges per subject
N_STIMULI = 200
T_STEPS = 12
N_TRAIN_GRAPHS = 500
N_TEST = 50
N_SUBJECTS = 8
N_STEPS = 1500
BATCH = 32
LR = 1e-2
K_TL = 4
H_MLP = 64
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
        R = transitive_closure(A)
        As.append(A)
        Rs.append(R)
    return torch.stack(As), torch.stack(Rs)


def latent_brain(K, p, seed):
    """Latent ground-truth connectivity. Directed, with cycles allowed."""
    rng = random.Random(seed)
    return random_directed(K, p, rng)


def perturb_brain(A_star, frac_edges_changed, seed):
    """Density-preserving per-subject perturbation: replace a fraction of
    A_star's edges with same-count random draw from non-edges. (Earlier
    versions used per-cell flip prob, which inflates density on sparse
    brains; see exp75 docstring.)"""
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
    """Pretend-TRIBE: simulate voxel timeseries from a connectivity matrix.

    Real TRIBE v2 maps (video, audio, text) -> voxels[~70k]. Here we
    abstract that to: stimulus -> voxels[K], where the dynamical system
    on the latent connectivity is the source of structure. Replacing
    this function with a real TRIBE call is a one-line swap.

    Returns V of shape (n_stimuli, t_steps, K).
    """
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


def recover_adjacency(V, top_k_per_row=4):
    """Estimate A_hat from voxel timeseries via lag-1 cross-correlation.

    For each ordered pair (i, j), score = sum_stim sum_t V[s, t, i] * V[s, t+1, j].
    Keep the top-k outgoing edges per source node, binarize.
    This is a deliberately crude estimator — exactly the kind of thing
    we'd plug a fancier method into when wiring real TRIBE outputs.
    """
    n_stim, T, K = V.shape
    Vi = V[:, :-1, :]
    Vj = V[:, 1:, :]
    score = torch.einsum("sti,stj->ij", Vi, Vj)
    score.fill_diagonal_(float("-inf"))
    A_hat = torch.zeros(K, K)
    topv, topi = torch.topk(score, k=min(top_k_per_row, K - 1), dim=1)
    for i in range(K):
        for j in topi[i].tolist():
            A_hat[i, j] = 1.0
    return A_hat


class TL(nn.Module):
    """3-scalar closure recurrence (identical to exp44)."""
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A, K_iters=None):
        if K_iters is None:
            K_iters = K_TL
        R = A.clone()
        for _ in range(K_iters):
            comp = torch.einsum("bij,bjk->bik", R, A)
            R = torch.sigmoid(self.alpha * comp + self.beta * R + self.gamma)
        return R


class MLP(nn.Module):
    def __init__(self, n, h=H_MLP):
        super().__init__()
        self.n = n
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


def train(model, train_A, train_R, n_steps, lr=LR, batch=BATCH):
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
    return loss.item()


@torch.no_grad()
def evaluate(model, A, R, threshold=0.5, K_iters=None):
    model.eval()
    A = A.to(DEVICE)
    R = R.to(DEVICE)
    R_pred = model(A, K_iters=K_iters) if isinstance(model, TL) else model(A)
    pred = (R_pred > threshold).float()
    tp = ((pred == 1) & (R == 1)).sum().item()
    fp = ((pred == 1) & (R == 0)).sum().item()
    fn = ((pred == 0) & (R == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    model.train()
    return f1


def adjacency_jaccard(A_hat, A_true):
    """Cell-level Jaccard between recovered and true adjacency."""
    inter = ((A_hat == 1) & (A_true == 1)).sum().item()
    union = ((A_hat == 1) | (A_true == 1)).sum().item()
    return inter / (union + 1e-9)


def main():
    print(f"device={DEVICE}  K_train={K_TRAIN}  K_brain={K_BRAIN}  steps={N_STEPS}  seeds={N_SEEDS}")

    # Build the TRIBE-stub eval suite (shared across seeds since the brain is fixed).
    A_star = latent_brain(K_BRAIN, EDGE_PROB_BRAIN, seed=2026)
    print(f"latent brain: K={K_BRAIN}, edge_prob={EDGE_PROB_BRAIN}, edges={int(A_star.sum().item())}")

    subjects_gt, subjects_hat, subjects_hat_noisy = [], [], []
    closures_gt = []
    jaccards_clean, jaccards_noisy = [], []
    for s in range(N_SUBJECTS):
        A_s = perturb_brain(A_star, SUBJECT_FLIP_FRAC, seed=1000 + s)
        R_s = transitive_closure(A_s)

        V_clean = tribe_stub_voxels(A_s, N_STIMULI, T_STEPS, noise_std=0.0, seed=2000 + s)
        V_noisy = tribe_stub_voxels(A_s, N_STIMULI, T_STEPS, noise_std=0.20, seed=3000 + s)
        A_hat = recover_adjacency(V_clean, top_k_per_row=int(K_BRAIN * EDGE_PROB_BRAIN))
        A_hat_noisy = recover_adjacency(V_noisy, top_k_per_row=int(K_BRAIN * EDGE_PROB_BRAIN))

        subjects_gt.append(A_s)
        subjects_hat.append(A_hat)
        subjects_hat_noisy.append(A_hat_noisy)
        closures_gt.append(R_s)
        jaccards_clean.append(adjacency_jaccard(A_hat, A_s))
        jaccards_noisy.append(adjacency_jaccard(A_hat_noisy, A_s))

    A_gt = torch.stack(subjects_gt)
    A_hat = torch.stack(subjects_hat)
    A_hat_noisy = torch.stack(subjects_hat_noisy)
    R_gt = torch.stack(closures_gt)

    print(f"adjacency recovery (clean):  Jaccard mean={statistics.mean(jaccards_clean):.3f}")
    print(f"adjacency recovery (noisy):  Jaccard mean={statistics.mean(jaccards_noisy):.3f}")

    K_iters_brain = int(math.ceil(math.log2(K_BRAIN))) + 4

    summary = {"in_dist": [], "tribe_gt": [], "tribe_hat": [], "tribe_hat_noisy": []}
    mlp_summary = {"in_dist": []}
    n_params_tl = 0
    n_params_mlp = 0

    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 13 + 55)
        train_A, train_R = gen_dags(K_TRAIN, EDGE_PROB_TRAIN, N_TRAIN_GRAPHS, seed * 1000)
        test_A, test_R = gen_dags(K_TRAIN, EDGE_PROB_TRAIN, N_TEST, seed * 1000 + 7)

        tl = TL().to(DEVICE)
        n_params_tl = sum(p.numel() for p in tl.parameters())
        train(tl, train_A, train_R, N_STEPS)

        mlp = MLP(K_TRAIN).to(DEVICE)
        n_params_mlp = sum(p.numel() for p in mlp.parameters())
        train(mlp, train_A, train_R, N_STEPS)

        f1_id = evaluate(tl, test_A, test_R)
        f1_mlp = evaluate(mlp, test_A, test_R)
        f1_gt = evaluate(tl, A_gt, R_gt, K_iters=K_iters_brain)
        f1_hat = evaluate(tl, A_hat, R_gt, K_iters=K_iters_brain)
        f1_hat_noisy = evaluate(tl, A_hat_noisy, R_gt, K_iters=K_iters_brain)

        summary["in_dist"].append(f1_id)
        summary["tribe_gt"].append(f1_gt)
        summary["tribe_hat"].append(f1_hat)
        summary["tribe_hat_noisy"].append(f1_hat_noisy)
        mlp_summary["in_dist"].append(f1_mlp)

        a, b, g = tl.alpha.item(), tl.beta.item(), tl.gamma.item()
        print(
            f"  seed={seed}  TL(α={a:.2f} β={b:.2f} γ={g:.2f})  "
            f"in_dist={f1_id:.3f}  tribe_gt={f1_gt:.3f}  "
            f"tribe_hat={f1_hat:.3f}  tribe_hat_noisy={f1_hat_noisy:.3f}  "
            f"MLP_in_dist={f1_mlp:.3f}"
        )

    print("\n=== summary (mean F1 over {} seeds) ===".format(N_SEEDS))
    print(f"TL  ({n_params_tl} params)")
    for cond in ["in_dist", "tribe_gt", "tribe_hat", "tribe_hat_noisy"]:
        vs = summary[cond]
        m = statistics.mean(vs)
        s = statistics.stdev(vs) if len(vs) > 1 else 0.0
        print(f"  {cond:<20} {m:.3f} ± {s:.3f}")
    print(f"MLP ({n_params_mlp} params, K=16 only — no native answer at K={K_BRAIN})")
    vs = mlp_summary["in_dist"]
    m = statistics.mean(vs)
    s = statistics.stdev(vs) if len(vs) > 1 else 0.0
    print(f"  in_dist              {m:.3f} ± {s:.3f}")

    print("\nNotes:")
    print("  tribe_gt   = TL closure on the latent A_s (ground-truth brain).")
    print("  tribe_hat  = TL closure on A_hat recovered from clean voxel timeseries.")
    print("  tribe_hat_noisy = same with measurement noise on voxels (std=0.20).")
    print("  Replace tribe_stub_voxels(...) with a real TRIBE v2 forward pass to")
    print("  re-run this end-to-end on actual fMRI predictions.")


if __name__ == "__main__":
    main()
