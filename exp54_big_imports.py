"""
exp54: zero-shot transitive closure on LARGE real OSS Python codebases.

exp53 evaluated on 8 small/mid packages (n=17 to n=100); rich (n=100) was
the only case where TL F1 fell below 0.98, and that was from K=4
iterations being too short for a 100-node graph. This experiment pushes
to substantially larger packages (n in the hundreds to low thousands)
with K scaled to graph size, to probe where TL actually breaks (if at
all) and to put the "generalizes to real codebases" claim under real
load.

Setup is identical to exp53 except:
  - bigger packages (sqlalchemy, fastapi, sympy, networkx, scikit-learn, django)
  - K_TL is adaptive: K = ceil(log2(max(n, 2))) + 4. Closure converges in
    log2(diameter) iterations of R = R^2 | R, so this is comfortable.
  - reports n, density, TL F1, and MLP F1 cropped to n=16 (same caveat as
    exp53: the MLP has no native-n answer for big graphs).
"""

import ast
import math
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

PACKAGES = ["sqlalchemy", "fastapi", "sympy", "networkx", "scikit-learn", "django"]

N_TRAIN_NODES = 16
EDGE_PROB = 0.12
N_TRAIN_GRAPHS = 500
N_STEPS = 1500
BATCH = 32
LR = 1e-2
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


# scikit-learn's pip-package name and import-package name differ.
PKG_TO_IMPORT_NAME = {"scikit-learn": "sklearn"}


def find_package_root(extract_dir: Path, package: str) -> Path:
    target = PKG_TO_IMPORT_NAME.get(package, package).replace("-", "_")
    for sub in extract_dir.iterdir():
        if sub.is_dir():
            for candidate in (sub / target, sub / "src" / target):
                if candidate.is_dir() and (candidate / "__init__.py").exists():
                    return candidate
    for inner in extract_dir.rglob("__init__.py"):
        if inner.parent.name == target:
            return inner.parent
    raise FileNotFoundError(f"Could not find package source for {package} (looking for {target!r}) in {extract_dir}")


def download_and_extract(package: str, dest: Path) -> Path:
    pkg_dir = dest / package.replace("/", "_")
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
        raise RuntimeError(f"pip download failed for {package}:\n{r.stderr[:1000]}")
    archive = next((p for p in pkg_dir.iterdir()
                    if p.name.endswith((".tar.gz", ".tgz", ".zip", ".tar"))), None)
    if archive is None:
        raise RuntimeError(f"No archive found in {pkg_dir}")
    extract_to = pkg_dir / "extracted"
    extract_to.mkdir()
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_to)
    elif archive.name.endswith(".tar"):
        with tarfile.open(archive, "r:") as tf:
            tf.extractall(extract_to)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_to)
    else:
        raise RuntimeError(f"Unknown archive type: {archive}")
    return find_package_root(extract_to, package)


def extract_imports(root_dir: Path):
    py_files = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.endswith(".py"):
                py_files.append(Path(dp) / fn)
    file_to_name = {}
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
    def __init__(self, K=4):
        super().__init__()
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.gamma = nn.Parameter(torch.tensor(-2.5))

    def forward(self, A, K=None):
        K = K if K is not None else self.K
        R = A.clone()
        for _ in range(K):
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
def eval_f1(pred, R, threshold=0.5):
    pred = (pred > threshold).float()
    tp = ((pred == 1) & (R == 1)).sum().item()
    fp = ((pred == 1) & (R == 0)).sum().item()
    fn = ((pred == 0) & (R == 1)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


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
    print("training TL...")
    tl_loss = train(tl, train_A, train_R)
    print(f"  TL loss={tl_loss:.5f}  alpha={tl.alpha.item():.2f} beta={tl.beta.item():.2f} gamma={tl.gamma.item():.2f}")
    print("training MLP...")
    mlp_loss = train(mlp, train_A, train_R)
    print(f"  MLP loss={mlp_loss:.5f}\n")

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
                density = R.sum().item() / max(1, n * n)

                K = max(4, int(math.ceil(math.log2(max(n, 2)))) + 4)
                A_dev = A.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = tl(A_dev, K=K).cpu()
                tl_f1 = eval_f1(pred, R.unsqueeze(0))

                A_crop = crop_top_n(A, N_TRAIN_NODES)
                R_crop = transitive_closure(A_crop)
                with torch.no_grad():
                    mlp_pred = mlp(A_crop.unsqueeze(0).to(DEVICE)).cpu()
                mlp_f1 = eval_f1(mlp_pred, R_crop.unsqueeze(0))

                rows.append((pkg, n, density, K, tl_f1, mlp_f1))
                print(f"  {pkg:<14} n={n:<5} density={density:.3f}  K={K:<3}  TL F1={tl_f1:.3f}   MLP(crop16) F1={mlp_f1:.3f}")
            except Exception as e:
                print(f"  {pkg}: ERROR — {str(e)[:300]}")

    if not rows:
        print("\nNo packages evaluated successfully.")
        return

    print("\n=== summary ===")
    print(f"{'package':<14}{'n':<7}{'density':<10}{'K':<5}{'TL F1':<10}{'MLP F1 (crop-16)':<18}")
    for pkg, n, density, K, tl_f1, mlp_f1 in rows:
        print(f"{pkg:<14}{n:<7}{density:<10.3f}{K:<5}{tl_f1:<10.3f}{mlp_f1:<18.3f}")

    tl_mean = statistics.mean(r[4] for r in rows)
    mlp_mean = statistics.mean(r[5] for r in rows)
    print(f"\nmean TL F1 across {len(rows)} large packages:  {tl_mean:.3f}")
    print(f"mean MLP F1 (cropped to n=16):                 {mlp_mean:.3f}")


if __name__ == "__main__":
    main()
