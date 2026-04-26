"""
exp53: zero-shot transitive closure on real OSS Python codebases.

Hypothesis (strong form of exp44): a 3-scalar TL recurrence trained on
synthetic random DAGs predicts the EXACT transitive closure of import
graphs for real OSS Python packages, zero-shot. A standard MLP trained on
the same synthetic data does not generalize across graph sizes at all.

Pipeline:
  1. pip-download source tarballs for a fixed list of well-known Python
     packages (no install, no deps).
  2. Extract each, parse internal imports via ast (skip stdlib + external).
  3. Train TL (3 scalars: alpha, beta, gamma) on synthetic 16-node DAGs at
     p=0.12 — same training distribution as exp44.
  4. Evaluate TL on each real package at NATIVE size (variable n).
  5. Evaluate the exp44 MLP cropped to its trained N=16 (top-16 most-
     connected modules), as a sanity-check baseline.
  6. Report F1 + parameter counts per package.

This is the version of exp44 that supports the "generalizes to real
codebases" framing. exp44 evaluated only on the tensor project's own
imports; this evaluates on N independent OSS packages.
"""

import ast
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Small/mid Python packages with mostly self-contained internal structure.
PACKAGES = ["click", "rich", "httpx", "flask", "requests", "jinja2", "markdown", "tqdm"]

N_TRAIN_NODES = 16
EDGE_PROB = 0.12
N_TRAIN_GRAPHS = 500
N_STEPS = 1500
BATCH = 32
LR = 1e-2
K_TL = 4
H_MLP = 64
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


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
    import random
    rng = random.Random(seed)
    As, Rs = [], []
    for _ in range(n_graphs):
        A = random_dag(n_nodes, p, rng)
        R = transitive_closure(A)
        As.append(A)
        Rs.append(R)
    return torch.stack(As), torch.stack(Rs)


def find_package_root(extract_dir: Path, package: str) -> Path:
    """Find the directory containing the package's actual source."""
    candidates = []
    for sub in extract_dir.iterdir():
        if sub.is_dir():
            inner = sub / package
            if inner.is_dir() and (inner / "__init__.py").exists():
                return inner
            src_inner = sub / "src" / package
            if src_inner.is_dir() and (src_inner / "__init__.py").exists():
                return src_inner
            candidates.append(sub)
    for cand in candidates:
        for inner in cand.rglob("__init__.py"):
            if inner.parent.name == package:
                return inner.parent
    raise FileNotFoundError(f"Could not find package source for {package} in {extract_dir}")


def download_and_extract(package: str, dest: Path) -> Path:
    """pip download source, extract, return path to package source dir."""
    pkg_dir = dest / package
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)
    print(f"  downloading {package}...", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:",
         "-d", str(pkg_dir), package],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"pip download failed for {package}:\n{r.stderr}")
    archive = next((p for p in pkg_dir.iterdir() if p.suffix in (".gz", ".zip", ".tar") or p.name.endswith(".tar.gz")), None)
    if archive is None:
        raise RuntimeError(f"No archive found in {pkg_dir}")
    extract_to = pkg_dir / "extracted"
    extract_to.mkdir()
    if archive.name.endswith(".tar.gz") or archive.suffix == ".gz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_to)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_to)
    else:
        raise RuntimeError(f"Unknown archive type: {archive}")
    return find_package_root(extract_to, package)


def extract_imports(root_dir: Path):
    """Parse all .py files in root_dir; return adjacency over modules internal to this package."""
    py_files = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.endswith(".py"):
                py_files.append(Path(dp) / fn)
    file_to_name = {}
    pkg_name = root_dir.name
    for f in py_files:
        rel = f.relative_to(root_dir.parent)
        name = str(rel).replace(os.sep, ".")[:-3]
        if name.endswith(".__init__"):
            name = name[:-9]
        file_to_name[f] = name
    name_set = set(file_to_name.values())
    names = sorted(name_set)
    name_to_idx = {n: i for i, n in enumerate(names)}

    edges = set()
    for f, src in file_to_name.items():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"), filename=str(f))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in name_set:
                        edges.add((src, alias.name))
            elif isinstance(node, ast.ImportFrom):
                # Resolve relative imports to absolute internal names.
                if node.level and node.level > 0:
                    parts = src.split(".")
                    base = parts[: max(0, len(parts) - node.level)]
                    if node.module:
                        base = base + node.module.split(".")
                    target = ".".join(base)
                    if target in name_set:
                        edges.add((src, target))
                    for alias in node.names:
                        sub = f"{target}.{alias.name}" if target else alias.name
                        if sub in name_set:
                            edges.add((src, sub))
                elif node.module:
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


class TL(nn.Module):
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


def train(model, train_A, train_R):
    opt = optim.Adam(model.parameters(), lr=LR)
    n_g = train_A.shape[0]
    for _ in range(N_STEPS):
        idx = torch.randint(0, n_g, (min(BATCH, n_g),))
        A = train_A[idx].to(DEVICE)
        R = train_R[idx].to(DEVICE)
        R_pred = model(A)
        loss = F.binary_cross_entropy(R_pred.clamp(1e-6, 1 - 1e-6), R)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


@torch.no_grad()
def eval_f1(model, A, R, threshold=0.5):
    model.eval()
    A = A.to(DEVICE)
    R = R.to(DEVICE)
    pred = (model(A) > threshold).float()
    tp = ((pred == 1) & (R == 1)).sum().item()
    fp = ((pred == 1) & (R == 0)).sum().item()
    fn = ((pred == 0) & (R == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    model.train()
    return f1


def crop_top_n(A, n_target):
    n = A.shape[0]
    if n <= n_target:
        pad = torch.zeros(n_target, n_target)
        pad[:n, :n] = A
        return pad
    deg = A.sum(0) + A.sum(1)
    keep = sorted(torch.argsort(deg, descending=True)[:n_target].tolist())
    return A[keep][:, keep]


def main():
    print(f"device={DEVICE}  train: {N_TRAIN_GRAPHS} random DAGs at n={N_TRAIN_NODES}, p={EDGE_PROB}")

    torch.manual_seed(0)
    train_A, train_R = gen_graphs(N_TRAIN_NODES, EDGE_PROB, N_TRAIN_GRAPHS, seed=42)

    tl = TL().to(DEVICE)
    mlp = MLP(N_TRAIN_NODES).to(DEVICE)
    print(f"TL params:  {sum(p.numel() for p in tl.parameters())}")
    print(f"MLP params: {sum(p.numel() for p in mlp.parameters())}")
    print("training TL on random DAGs...")
    tl_loss = train(tl, train_A, train_R)
    print(f"  TL final loss = {tl_loss:.5f}  (alpha={tl.alpha.item():.2f} beta={tl.beta.item():.2f} gamma={tl.gamma.item():.2f})")
    print("training MLP on random DAGs...")
    mlp_loss = train(mlp, train_A, train_R)
    print(f"  MLP final loss = {mlp_loss:.5f}")

    # Evaluate on synthetic in-distribution as sanity check.
    test_A, test_R = gen_graphs(N_TRAIN_NODES, EDGE_PROB, 100, seed=1234)
    tl_id = eval_f1(tl, test_A, test_R)
    mlp_id = eval_f1(mlp, test_A, test_R)
    print(f"\nin-dist (random DAGs, n={N_TRAIN_NODES}):  TL F1={tl_id:.3f}  MLP F1={mlp_id:.3f}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = []
        for pkg in PACKAGES:
            try:
                src_dir = download_and_extract(pkg, tmp_path)
                names, A = extract_imports(src_dir)
                n = A.shape[0]
                if n < 4:
                    print(f"  {pkg}: only {n} internal modules, skipping")
                    continue
                R = transitive_closure(A)

                # TL: native size, single graph
                tl_f1 = eval_f1(tl, A.unsqueeze(0), R.unsqueeze(0))

                # MLP: forced to its trained N=16 — crop top-16 by degree, eval on cropped subgraph.
                A_crop = crop_top_n(A, N_TRAIN_NODES)
                R_crop = transitive_closure(A_crop)
                mlp_f1 = eval_f1(mlp, A_crop.unsqueeze(0), R_crop.unsqueeze(0))

                density = R.sum().item() / max(1, n * n)
                rows.append((pkg, n, density, tl_f1, mlp_f1))
                print(f"  {pkg:<10} n={n:<4} density={density:.3f}  TL F1={tl_f1:.3f}   MLP(crop16) F1={mlp_f1:.3f}")
            except Exception as e:
                print(f"  {pkg}: ERROR — {e}")

    if not rows:
        print("\nNo packages evaluated successfully.")
        return

    print("\n=== summary ===")
    print(f"{'package':<12}{'n':<6}{'density':<10}{'TL F1':<10}{'MLP F1 (crop-16)':<18}")
    for pkg, n, density, tl_f1, mlp_f1 in rows:
        print(f"{pkg:<12}{n:<6}{density:<10.3f}{tl_f1:<10.3f}{mlp_f1:<18.3f}")

    tl_mean = statistics.mean(r[3] for r in rows)
    mlp_mean = statistics.mean(r[4] for r in rows)
    print(f"\nmean TL F1 across {len(rows)} packages:  {tl_mean:.3f}")
    print(f"mean MLP F1 (cropped) across {len(rows)} packages:  {mlp_mean:.3f}")
    print(f"\nNote: TL evaluated at NATIVE n per package; MLP can only be evaluated at its")
    print(f"trained n={N_TRAIN_NODES}, so its score reflects the cropped 16-node subgraph,")
    print(f"not the full package. The MLP has no zero-shot answer for variable-n graphs.")


if __name__ == "__main__":
    main()
