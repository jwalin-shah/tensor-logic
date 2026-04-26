"""
exp44: TL as substrate for transitive-closure learning on Python-import-style DAGs.

Hypothesis: TL's einsum recurrence
    R_{k+1} = sigmoid(alpha * (R_k @ A) + beta * R_k + gamma)
parameterized with 3 scalars matches transitive closure exactly. On sparse
DAGs (p≈0.12, Python-import-like), 3-param TL should beat a ~40k-param MLP
in-distribution AND generalize zero-shot to the real tensor-project import
graph and to OOD sparsity. Structural prior pays off most at low data.

Eval conditions:
  in_dist     — held-out random DAGs, same distribution as train
  real        — real Python imports from this project, zero-shot (padded to N)
  ood_sparse  — random DAGs at p=0.05
  ood_dense   — random DAGs at p=0.25

Sweep: N_TRAIN in {20, 100, 500} x 3 seeds x {TL, MLP}.
"""

import os
import ast
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

N_NODES = 16
EDGE_PROB = 0.12
N_TEST = 50
N_TRAIN_LIST = [20, 100, 500]
N_STEPS = 1000
BATCH = 32
LR = 1e-2
K_TL = 4
H_MLP = 64
N_SEEDS = 3
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TENSOR_PROJECT_DIR = "/Users/jwalinshah/projects/tensor"


def random_dag(n, p, rng):
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                A[i, j] = 1.0
    return A


def transitive_closure(A):
    R = A.clone()
    n = A.shape[-1]
    for _ in range(n + 1):
        R_new = ((R @ R + R) > 0).float()
        if torch.equal(R_new, R):
            break
        R = R_new
    return R


def gen_graphs(n_nodes, p, n_graphs, seed):
    rng = random.Random(seed)
    As, Rs = [], []
    for _ in range(n_graphs):
        A = random_dag(n_nodes, p, rng)
        R = transitive_closure(A)
        As.append(A)
        Rs.append(R)
    return torch.stack(As), torch.stack(Rs)


def extract_imports(root_dir):
    """Walk root_dir, parse each .py with ast, return adjacency over internal modules."""
    py_files = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.endswith(".py"):
                py_files.append(os.path.join(dp, fn))
    file_to_name = {}
    for f in py_files:
        rel = os.path.relpath(f, root_dir)
        name = rel.replace(os.sep, ".")[:-3]
        if name.endswith(".__init__"):
            name = name[:-9]
        if name and not name.startswith("."):
            file_to_name[f] = name
    name_set = set(file_to_name.values())
    names = sorted(name_set)
    name_to_idx = {n: i for i, n in enumerate(names)}

    edges = set()
    for f, src in file_to_name.items():
        try:
            tree = ast.parse(open(f).read(), filename=f)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in name_set:
                        edges.add((src, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module in name_set:
                    edges.add((src, node.module))
                for alias in node.names:
                    sub = f"{node.module}.{alias.name}"
                    if sub in name_set:
                        edges.add((src, sub))
    n = len(names)
    A = torch.zeros(n, n)
    for a, b in edges:
        A[name_to_idx[a], name_to_idx[b]] = 1.0
    return names, A


def real_graph_padded(n_target):
    """Return adjacency + closure for the tensor project, cropped/padded to n_target.
    Strategy: pick the n_target nodes with highest in+out degree (most connected)."""
    names, A = extract_imports(TENSOR_PROJECT_DIR)
    n = A.shape[0]
    if n == 0:
        # fall back: a tiny synthetic graph
        return torch.zeros(1, n_target, n_target), torch.zeros(1, n_target, n_target)
    deg = (A.sum(0) + A.sum(1))
    keep = torch.argsort(deg, descending=True)[:n_target].tolist()
    keep_sorted = sorted(keep)
    sub_A = A[keep_sorted][:, keep_sorted]
    if sub_A.shape[0] < n_target:
        pad = torch.zeros(n_target, n_target)
        pad[:sub_A.shape[0], :sub_A.shape[0]] = sub_A
        sub_A = pad
    sub_R = transitive_closure(sub_A)
    return sub_A.unsqueeze(0), sub_R.unsqueeze(0)


class TL(nn.Module):
    """3-param recurrent einsum that EXACTLY matches transitive closure shape.

    Init at a known-good closure operating point (alpha=5, beta=5, gamma=-2.5)
    so absent cells stay near 0 and present cells saturate near 1; optimizer
    only has to refine sharpness. With gamma=0 init, BCE on 10%-dense targets
    drives gamma very negative and TL collapses to predicting all zeros.
    """
    def __init__(self, K=K_TL):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A):  # A: (B, n, n)
        R = A.clone()
        for _ in range(self.K):
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
    losses = []
    for _ in range(n_steps):
        idx = torch.randint(0, n_g, (min(batch, n_g),))
        A = train_A[idx].to(DEVICE)
        R = train_R[idx].to(DEVICE)
        R_pred = model(A)
        loss = F.binary_cross_entropy(R_pred.clamp(1e-6, 1 - 1e-6), R)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate(model, A, R, threshold=0.5):
    model.eval()
    A = A.to(DEVICE)
    R = R.to(DEVICE)
    R_pred = model(A)
    pred = (R_pred > threshold).float()
    tp = ((pred == 1) & (R == 1)).sum().item()
    fp = ((pred == 1) & (R == 0)).sum().item()
    fn = ((pred == 0) & (R == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (pred == R).float().mean().item()
    model.train()
    return f1, acc


def main():
    print(f"device={DEVICE}  N={N_NODES}  p={EDGE_PROB}  steps={N_STEPS}  seeds={N_SEEDS}")

    # Build the eval suites once
    real_A, real_R = real_graph_padded(N_NODES)
    real_density = real_R.sum().item() / (N_NODES * N_NODES)
    print(f"real graph: padded/cropped to {N_NODES}, closure density={real_density:.3f}")

    rng_master = random.Random(0)

    summary = {}
    for n_train in N_TRAIN_LIST:
        for model_name in ["TL", "MLP"]:
            run_results = {"in_dist": [], "real": [], "ood_sparse": [], "ood_dense": []}
            n_params_last = 0
            for seed in range(N_SEEDS):
                torch.manual_seed(seed * 13 + n_train)
                train_A, train_R = gen_graphs(N_NODES, EDGE_PROB, n_train, seed * 1000 + n_train)
                test_A, test_R = gen_graphs(N_NODES, EDGE_PROB, N_TEST, seed * 1000 + n_train + 7)
                ood_sparse_A, ood_sparse_R = gen_graphs(N_NODES, 0.05, N_TEST, seed * 1000 + n_train + 11)
                ood_dense_A, ood_dense_R = gen_graphs(N_NODES, 0.25, N_TEST, seed * 1000 + n_train + 17)

                if model_name == "TL":
                    model = TL().to(DEVICE)
                else:
                    model = MLP(N_NODES).to(DEVICE)
                n_params_last = sum(p.numel() for p in model.parameters())

                losses = train(model, train_A, train_R, N_STEPS)
                f1_id, _ = evaluate(model, test_A, test_R)
                f1_re, _ = evaluate(model, real_A, real_R)
                f1_os, _ = evaluate(model, ood_sparse_A, ood_sparse_R)
                f1_od, _ = evaluate(model, ood_dense_A, ood_dense_R)

                run_results["in_dist"].append(f1_id)
                run_results["real"].append(f1_re)
                run_results["ood_sparse"].append(f1_os)
                run_results["ood_dense"].append(f1_od)

                if model_name == "TL":
                    a, b, g = model.alpha.item(), model.beta.item(), model.gamma.item()
                    extra = f"  TL(α={a:.2f}, β={b:.2f}, γ={g:.2f})"
                else:
                    extra = ""
                print(
                    f"  n_train={n_train:<4} {model_name:<3} seed={seed} "
                    f"loss={losses[-1]:.4f}  ID={f1_id:.3f}  real={f1_re:.3f}  "
                    f"ood_sp={f1_os:.3f}  ood_de={f1_od:.3f}{extra}"
                )

            summary[(n_train, model_name)] = (run_results, n_params_last)

    print("\n=== summary (mean F1 over 3 seeds) ===")
    print(f"{'n_train':<10}{'model':<6}{'params':<10}{'in_dist':<14}{'real':<14}{'ood_sparse':<14}{'ood_dense':<14}")
    for n_train in N_TRAIN_LIST:
        for model_name in ["TL", "MLP"]:
            run_results, n_params = summary[(n_train, model_name)]
            row = f"{n_train:<10}{model_name:<6}{n_params:<10}"
            for cond in ["in_dist", "real", "ood_sparse", "ood_dense"]:
                vs = run_results[cond]
                m = statistics.mean(vs)
                s = statistics.stdev(vs) if len(vs) > 1 else 0.0
                row += f"{m:.3f}±{s:.3f}  "
            print(row)


if __name__ == "__main__":
    main()
