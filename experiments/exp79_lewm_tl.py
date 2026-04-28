"""
Experiment 79: Minimal JEPA World Model + TL Relational Layer
=============================================================
Pipeline: synthetic pixel scenes → JEPA encoder → linear probe →
TL fixpoint → counterfactual retraction + complexity scaling.

Claim: a holistic pixel latent encodes enough relational structure for
a linear probe to decode binary object relations, and TL derives
higher-order facts with predictable (not catastrophic) depth-2 degradation.

Falsification: probe accuracy <70% → need object-centric encoder (Slot Attention).

Run: python experiments/exp79_lewm_tl.py [--skip-train]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensor_logic import Domain, Relation

torch.manual_seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
CANVAS = 64
OBJ_SIZE = 10
N_OBJ = 3
OBJ_NAMES = ["R", "G", "B"]
COLORS = [(220, 50, 50), (50, 200, 50), (50, 50, 220)]
RELATIONS = ["above", "left_of", "touching", "occluded"]
PAIRS = [(a, b) for a in OBJ_NAMES for b in OBJ_NAMES if a != b]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}
OBJ_IDX = {n: i for i, n in enumerate(OBJ_NAMES)}
LATENT_DIM = 64
N_PAIRS = 6
DATA_DIR = os.path.join(os.path.dirname(__file__), "exp79_data")

# ── 1. Data Generation ────────────────────────────────────────────────────

def generate_sequence(n_frames=8, rng=None):
    """Returns frames (T,3,H,W) float32 in [0,1], positions (T,N,2) float64."""
    if rng is None:
        rng = np.random.default_rng()
    half = OBJ_SIZE // 2
    pos = rng.integers(half + 4, CANVAS - half - 4, size=(N_OBJ, 2)).astype(float)
    vel = rng.uniform(-3.0, 3.0, size=(N_OBJ, 2))

    frames_np, pos_hist = [], []
    for _ in range(n_frames):
        pos += vel
        for i in range(N_OBJ):
            for d in range(2):
                if pos[i, d] < half + 1:
                    pos[i, d] = half + 1.0
                    vel[i, d] = abs(vel[i, d])
                elif pos[i, d] > CANVAS - half - 1:
                    pos[i, d] = float(CANVAS - half - 1)
                    vel[i, d] = -abs(vel[i, d])
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        for i, (color, p) in enumerate(zip(COLORS, pos)):
            x, y = int(round(p[0])), int(round(p[1]))
            canvas[max(0, y - half):min(CANVAS, y + half),
                   max(0, x - half):min(CANVAS, x + half)] = color
        frames_np.append(canvas.copy())
        pos_hist.append(pos.copy())

    frames = torch.from_numpy(np.stack(frames_np)).permute(0, 3, 1, 2).float() / 255.0
    return frames, np.array(pos_hist)


def compute_relations(pos):
    """pos: (N_OBJ, 2) float array. Returns {(rel, a, b): bool} for all pairs."""
    rels = {}
    half = OBJ_SIZE // 2
    for i, a in enumerate(OBJ_NAMES):
        for j, b in enumerate(OBJ_NAMES):
            if i == j:
                continue
            pi, pj = pos[i], pos[j]
            rels[("above", a, b)] = bool(pi[1] < pj[1])
            rels[("left_of", a, b)] = bool(pi[0] < pj[0])
            gap_x = abs(pi[0] - pj[0]) - OBJ_SIZE
            gap_y = abs(pi[1] - pj[1]) - OBJ_SIZE
            rels[("touching", a, b)] = bool(max(gap_x, gap_y) < 2)
            bbox = abs(pi[0] - pj[0]) < OBJ_SIZE and abs(pi[1] - pj[1]) < OBJ_SIZE
            rels[("occluded", a, b)] = bool(bbox and i > j)
    return rels


def generate_labeled_frames(n=500, seed=99):
    """Returns list of (frame (3,H,W), pos (N,2)) tuples."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        frames, pos_hist = generate_sequence(n_frames=1, rng=rng)
        out.append((frames[0], pos_hist[0]))
    return out


# ── 2. JEPA Model ─────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),    # (B, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # (B, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (B, 128, 8, 8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)  # unit sphere: stable scale for stop-grad JEPA


class Predictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


def jepa_loss(encoder, predictor, frames_batch, device):
    """frames_batch: (B, T, 3, H, W). Returns scalar loss."""
    B, T, C, H, W = frames_batch.shape
    flat = frames_batch.view(B * T, C, H, W).to(device)
    z_all = encoder(flat).view(B, T, -1)           # (B, T, D)

    z_t = z_all[:, :-1].reshape(-1, z_all.shape[-1])           # (B*(T-1), D)
    z_tp1 = z_all[:, 1:].reshape(-1, z_all.shape[-1]).detach() # stop-gradient
    l_pred = F.mse_loss(predictor(z_t), z_tp1)

    # encoder outputs are L2-normalized (unit sphere) → no collapse risk, no reg needed
    return l_pred


# ── 3. JEPA Training ──────────────────────────────────────────────────────

def train_jepa(device, n_seq=9000, n_epochs=20, batch_size=128, lr=1e-3):
    """Train JEPA, save encoder.pt. Returns trained Encoder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("  Generating JEPA sequences...")
    rng = np.random.default_rng(42)
    seqs = [generate_sequence(n_frames=8, rng=rng)[0] for _ in range(n_seq)]
    data = torch.stack(seqs)  # (N, T, 3, H, W)

    encoder = Encoder().to(device)
    predictor = Predictor().to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr
    )

    n_batches = (n_seq + batch_size - 1) // batch_size
    for epoch in range(n_epochs):
        idx = torch.randperm(n_seq)
        total = 0.0
        for b in range(n_batches):
            batch = data[idx[b * batch_size:(b + 1) * batch_size]].to(device)
            opt.zero_grad()
            loss = jepa_loss(encoder, predictor, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{n_epochs}  loss={total / n_batches:.4f}")

    torch.save(encoder.state_dict(), os.path.join(DATA_DIR, "encoder.pt"))
    print(f"  Saved encoder to {DATA_DIR}/encoder.pt")
    return encoder


# ── 4. Linear Probe ───────────────────────────────────────────────────────

class RelationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({
            rel: nn.Linear(LATENT_DIM + N_PAIRS, 1) for rel in RELATIONS
        })

    def forward(self, z, pair_idx):
        """z: (B, D), pair_idx: (B,) long. Returns {rel: (B,) logit}."""
        pair_oh = F.one_hot(pair_idx, num_classes=N_PAIRS).float()
        x = torch.cat([z, pair_oh], dim=-1)
        return {rel: self.heads[rel](x).squeeze(-1) for rel in RELATIONS}


def make_probe_dataset(labeled_frames, encoder, device):
    """Returns (z, pair_idx, labels) tensors. labels: (N*6, 4) float."""
    encoder.eval()
    zs, pair_idxs, all_labels = [], [], []
    with torch.no_grad():
        for frame, pos in labeled_frames:
            z = encoder(frame.unsqueeze(0).to(device)).squeeze(0).cpu()
            rels = compute_relations(pos)
            for (a, b) in PAIRS:
                zs.append(z)
                pair_idxs.append(PAIR_INDEX[(a, b)])
                all_labels.append([float(rels[(r, a, b)]) for r in RELATIONS])
    return (
        torch.stack(zs),
        torch.tensor(pair_idxs, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.float32),
    )


def eval_probe(probe, labeled_frames, encoder, device):
    """Returns per-relation val accuracy dict without training."""
    z, pi, lab = make_probe_dataset(labeled_frames, encoder, device)
    probe.eval()
    with torch.no_grad():
        logits = probe(z.to(device), pi.to(device))
    acc = {}
    for i, rel in enumerate(RELATIONS):
        preds = (logits[rel].cpu() > 0).float()
        acc[rel] = (preds == lab[:, i]).float().mean().item()
    return acc


def train_probe(encoder, labeled_frames, device, n_epochs=50, lr=1e-2):
    """Train probe on first 400 frames, eval on last 100. Returns (probe, val_acc)."""
    train_frames, val_frames = labeled_frames[:400], labeled_frames[400:]
    z_tr, pi_tr, lab_tr = make_probe_dataset(train_frames, encoder, device)
    z_val, pi_val, lab_val = make_probe_dataset(val_frames, encoder, device)

    probe = RelationProbe().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(n_epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(z_tr.to(device), pi_tr.to(device))
        loss = sum(
            F.binary_cross_entropy_with_logits(logits[rel], lab_tr[:, i].to(device))
            for i, rel in enumerate(RELATIONS)
        )
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(z_val.to(device), pi_val.to(device))
    val_acc = {}
    for i, rel in enumerate(RELATIONS):
        preds = (logits[rel].cpu() > 0).float()
        val_acc[rel] = (preds == lab_val[:, i]).float().mean().item()

    torch.save(probe.state_dict(), os.path.join(DATA_DIR, "probe.pt"))
    return probe, val_acc


# ── 5. TL Wiring + Fixpoint ───────────────────────────────────────────────

_DOMAIN = Domain(OBJ_NAMES)


def build_tl_state(fact_dict):
    """fact_dict: {(rel_name, a, b): bool}. Returns dict of derived tensors."""
    rel_objs = {r: Relation(r, _DOMAIN, _DOMAIN) for r in RELATIONS}
    for (rel_name, a, b), val in fact_dict.items():
        if val and rel_name in rel_objs:
            rel_objs[rel_name][a, b] = 1.0

    above_t   = rel_objs["above"].eval()
    left_of_t = rel_objs["left_of"].eval()
    touch_t   = rel_objs["touching"].eval()
    occl_t    = rel_objs["occluded"].eval()

    # blocked_path[X,Z] = ∃Y: touching(X,Y) ∧ above(Y,Z)
    blocked_path = (touch_t @ above_t).clamp(0, 1)
    # same_side[X,Z] = ∃Y: left_of(X,Y) ∧ left_of(Y,Z)
    same_side = (left_of_t @ left_of_t).clamp(0, 1)
    # clear_above[X] = ¬∃Y: above(Y,X)  (1-ary)
    clear_above_rel = Relation("clear_above", _DOMAIN)
    for xi, xname in enumerate(OBJ_NAMES):
        clear_above_rel[xname] = float(1.0 - above_t[:, xi].max().item())

    return {
        "above": above_t,
        "left_of": left_of_t,
        "touching": touch_t,
        "occluded": occl_t,
        "blocked_path": blocked_path,
        "same_side": same_side,
        "clear_above": clear_above_rel.eval(),
    }


def remove_object(fact_dict, obj):
    """Return copy of fact_dict with all facts where obj appears removed."""
    return {k: v for k, v in fact_dict.items() if k[1] != obj and k[2] != obj}


def compute_gt_derived(pos, exclude_obj=None):
    """Ground-truth derived relations from raw geometry. Returns tensors."""
    n = len(OBJ_NAMES)
    blocked = torch.zeros(n, n)
    same = torch.zeros(n, n)
    clear = torch.ones(n)

    active = [i for i, nm in enumerate(OBJ_NAMES) if nm != exclude_obj]

    def touching(pi, pj):
        return max(abs(pi[0] - pj[0]) - OBJ_SIZE, abs(pi[1] - pj[1]) - OBJ_SIZE) < 2

    def above(pi, pj):
        return pi[1] < pj[1]

    def left_of(pi, pj):
        return pi[0] < pj[0]

    for xi in active:
        for zi in active:
            if xi == zi:
                continue
            for yi in active:
                if yi == xi or yi == zi:
                    continue
                if touching(pos[xi], pos[yi]) and above(pos[yi], pos[zi]):
                    blocked[xi, zi] = 1.0
                if left_of(pos[xi], pos[yi]) and left_of(pos[yi], pos[zi]):
                    same[xi, zi] = 1.0
        for yi in active:
            if yi == xi:
                continue
            if above(pos[yi], pos[xi]):
                clear[xi] = 0.0

    return {"blocked_path": blocked, "same_side": same, "clear_above": clear}


# ── 6. Evaluation ─────────────────────────────────────────────────────────

def _probe_facts_for_frame(frame, pos, probe, encoder, device):
    """Run probe on a single frame, return fact_dict."""
    encoder.eval()
    probe.eval()
    with torch.no_grad():
        z = encoder(frame.unsqueeze(0).to(device)).squeeze(0).cpu()
        logits = probe(
            z.unsqueeze(0).expand(N_PAIRS, -1),
            torch.arange(N_PAIRS, dtype=torch.long),
        )
    fact_dict = {}
    for pair_i, (a, b) in enumerate(PAIRS):
        for rel in RELATIONS:
            fact_dict[(rel, a, b)] = bool(logits[rel][pair_i].item() > 0)
    return fact_dict


def _b_is_active(state):
    bi = OBJ_IDX["B"]
    return bool(
        state["blocked_path"][bi, :].any()
        or state["blocked_path"][:, bi].any()
        or state["same_side"][bi, :].any()
        or state["same_side"][:, bi].any()
    )


def run_retraction_tl_only(labeled_frames):
    """TL-only retraction with ground-truth facts. Returns (n_active, n_correct)."""
    rng = np.random.default_rng(123)
    order = rng.permutation(len(labeled_frames))
    n_active = n_correct = 0
    for idx in order:
        if n_active >= 50:
            break
        frame, pos = labeled_frames[int(idx)]
        gt_facts = compute_relations(pos)
        state_full = build_tl_state(gt_facts)
        if not _b_is_active(state_full):
            continue
        n_active += 1
        facts_no_b = remove_object(gt_facts, "B")
        state_no_b = build_tl_state(facts_no_b)
        gt_no_b = compute_gt_derived(pos, exclude_obj="B")
        if (torch.equal(state_no_b["blocked_path"].round(), gt_no_b["blocked_path"])
                and torch.equal(state_no_b["same_side"].round(), gt_no_b["same_side"])
                and torch.equal(state_no_b["clear_above"].round(), gt_no_b["clear_above"])):
            n_correct += 1
    return n_active, n_correct


def run_retraction_e2e(labeled_frames, probe, encoder, device):
    """End-to-end retraction with probe outputs. Returns (n_active, n_correct)."""
    rng = np.random.default_rng(456)
    order = rng.permutation(len(labeled_frames))
    n_active = n_correct = 0
    for idx in order:
        if n_active >= 50:
            break
        frame, pos = labeled_frames[int(idx)]
        probe_facts = _probe_facts_for_frame(frame, pos, probe, encoder, device)
        state_full = build_tl_state(probe_facts)
        if not _b_is_active(state_full):
            continue
        n_active += 1
        facts_no_b = remove_object(probe_facts, "B")
        state_no_b = build_tl_state(facts_no_b)
        gt_no_b = compute_gt_derived(pos, exclude_obj="B")
        if (torch.equal(state_no_b["blocked_path"].round(), gt_no_b["blocked_path"])
                and torch.equal(state_no_b["same_side"].round(), gt_no_b["same_side"])
                and torch.equal(state_no_b["clear_above"].round(), gt_no_b["clear_above"])):
            n_correct += 1
    return n_active, n_correct


def compute_complexity_scaling(val_acc, val_frames, probe, encoder, device):
    """Compute depth-2 accuracy, save plot. Returns (depth1_acc, depth2_acc) dicts."""
    bp_correct = ss_correct = total = 0
    for frame, pos in val_frames[:100]:
        probe_facts = _probe_facts_for_frame(frame, pos, probe, encoder, device)
        state = build_tl_state(probe_facts)
        gt = compute_gt_derived(pos)
        for xi in range(len(OBJ_NAMES)):
            for zi in range(len(OBJ_NAMES)):
                if xi == zi:
                    continue
                bp_correct += int(state["blocked_path"][xi, zi].round().item()
                                  == gt["blocked_path"][xi, zi].item())
                ss_correct += int(state["same_side"][xi, zi].round().item()
                                  == gt["same_side"][xi, zi].item())
                total += 1

    bp_acc = bp_correct / total if total else 0.0
    ss_acc = ss_correct / total if total else 0.0
    depth1 = {r: val_acc[r] for r in RELATIONS}
    depth2 = {"blocked_path": bp_acc, "same_side": ss_acc}

    avg_d1 = (val_acc["above"] + val_acc["left_of"]) / 2
    compound = avg_d1 ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["depth-1\n(above)", "depth-1\n(left_of)", "depth-2\nblocked_path", "depth-2\nsame_side"],
        [val_acc["above"], val_acc["left_of"], bp_acc, ss_acc],
        color=["steelblue", "steelblue", "darkorange", "darkorange"],
    )
    ax.axhline(compound, color="red", linestyle="--",
               label=f"compound model avg_d1²={compound:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("exp79: Complexity Scaling (depth-1 vs depth-2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "complexity_curve.png"))
    plt.close()
    print(f"  Saved complexity_curve.png")

    return depth1, depth2


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="exp79: LeWM + TL relational layer")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load saved encoder.pt and probe.pt instead of retraining")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n── 1. Generating labeled frames ──────────────────────────────")
    labeled_frames = generate_labeled_frames(n=500, seed=99)
    print(f"  {len(labeled_frames)} labeled frames ready")

    print("\n── 2. JEPA encoder ───────────────────────────────────────────")
    encoder = Encoder().to(device)
    enc_path = os.path.join(DATA_DIR, "encoder.pt")
    if args.skip_train and os.path.exists(enc_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        print("  Loaded saved encoder")
    else:
        encoder = train_jepa(device)

    print("\n── 3. Linear probe ───────────────────────────────────────────")
    probe = RelationProbe().to(device)
    probe_path = os.path.join(DATA_DIR, "probe.pt")
    if args.skip_train and os.path.exists(probe_path):
        probe.load_state_dict(torch.load(probe_path, map_location=device))
        val_acc = eval_probe(probe, labeled_frames[400:], encoder, device)
        print("  Loaded saved probe")
    else:
        probe, val_acc = train_probe(encoder, labeled_frames, device)

    print("  Val accuracy:")
    for rel, acc in val_acc.items():
        gate = ("✓" if acc >= 0.9 else "✗") if rel in ("above", "left_of") else "~"
        print(f"    {gate} {rel}: {acc:.3f}")

    gate_probe = val_acc["above"] >= 0.9 and val_acc["left_of"] >= 0.9
    if not gate_probe:
        print("\n  FAIL: probe gate not met. Holistic latent insufficient → consider Slot Attention.")

    print("\n── 4. TL retraction tests ────────────────────────────────────")
    n_active_tl, n_correct_tl = run_retraction_tl_only(labeled_frames)
    gate_tl = n_correct_tl == n_active_tl
    print(f"  TL-only:    {n_correct_tl}/{n_active_tl}  {'PASS' if gate_tl else 'FAIL — rules wrong'}")

    n_active_e2e, n_correct_e2e = run_retraction_e2e(labeled_frames, probe, encoder, device)
    gate_e2e = n_correct_e2e >= 40
    print(f"  End-to-end: {n_correct_e2e}/{n_active_e2e}  {'PASS' if gate_e2e else 'needs ≥40'}")

    print("\n── 5. Complexity scaling ─────────────────────────────────────")
    depth1, depth2 = compute_complexity_scaling(val_acc, labeled_frames[400:], probe, encoder, device)
    print(f"  Depth-1: above={depth1['above']:.3f}  left_of={depth1['left_of']:.3f}")
    print(f"  Depth-2: blocked_path={depth2['blocked_path']:.3f}  same_side={depth2['same_side']:.3f}")

    results = {
        "probe_val_acc": val_acc,
        "tl_only_retraction": {"active": n_active_tl, "correct": n_correct_tl},
        "e2e_retraction": {"active": n_active_e2e, "correct": n_correct_e2e},
        "depth1_acc": depth1,
        "depth2_acc": depth2,
        "gates": {"probe": gate_probe, "tl_only": gate_tl, "e2e": gate_e2e},
    }
    with open(os.path.join(DATA_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {DATA_DIR}/results.json")

    all_pass = gate_probe and gate_tl and gate_e2e
    print(f"\n{'ALL PASS' if all_pass else 'PARTIAL'}: probe={gate_probe} tl_only={gate_tl} e2e={gate_e2e}")


if __name__ == "__main__":
    main()
