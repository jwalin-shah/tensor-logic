"""
Experiment 83: Slot Attention + TL Relational Layer
=============================================================
Pipeline: synthetic pixel scenes -> visual encoder -> Slot Attention ->
relation probe -> TL fixpoint -> counterfactual retraction + complexity scaling.

Claim: object-centric slots should expose binary relations more cleanly than
the holistic JEPA latent used in exp79_lewm_tl.py.

Falsification: above/left_of probe accuracy <90%, or end-to-end retraction
gets fewer than 40/50 active cases correct.

Run: python experiments/exp83_slot_attention.py
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
from tensor_logic.research.synthetic_scene import (
    CANVAS,
    COLORS,
    N_OBJ,
    N_PAIRS,
    OBJ_IDX,
    OBJ_NAMES,
    OBJ_SIZE,
    PAIR_INDEX,
    PAIRS,
    RELATIONS,
    build_spatial_tl_state as build_tl_state,
    compute_ground_truth_spatial_derivations as compute_gt_derived,
    compute_pairwise_spatial_relations as compute_relations,
    generate_colored_object_sequence as generate_sequence,
    generate_labeled_scene_frames as generate_labeled_frames,
    remove_object_facts as remove_object,
)
from experiments.runtime_paths import experiment_output_dir

torch.manual_seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────
LATENT_DIM = 64
FIXTURE_DATA_DIR = os.path.join(os.path.dirname(__file__), "exp83_slot_data")
DATA_DIR = str(experiment_output_dir("exp83_slot_attention"))


from tensor_logic.research.slot_attention import SlotAttention, VisualEncoder
from scipy.optimize import linear_sum_assignment

# ── Constants ──────────────────────────────────────────────────────────────
N_SLOTS = 4

# ── 2. Slot Models ────────────────────────────────────────────────────────

class ColorProbe(nn.Module):
    def __init__(self, slot_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Linear(slot_dim, 3) # RGB prediction

    def forward(self, slots):
        return self.net(slots)


class RelationProbe(nn.Module):
    def __init__(self, slot_dim=LATENT_DIM):
        super().__init__()
        # Predicts 4 relations for a pair of slots
        self.net = nn.Sequential(
            nn.Linear(slot_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, len(RELATIONS))
        )

    def forward(self, z1, z2):
        """z1, z2: (B, D). Returns (B, 4) logits."""
        return self.net(torch.cat([z1, z2], dim=-1))


def match_slots_to_objs(slot_colors, gt_colors):
    """
    slot_colors: (K, 3), gt_colors: (3, 3)
    Returns indices mapping gt_objs to slots.
    """
    # Simple MSE distance between predicted slot color and GT colors
    dist = torch.cdist(slot_colors.unsqueeze(0), gt_colors.unsqueeze(0)).squeeze(0) # (K, 3)
    # Hungarian matching
    slot_idx, obj_idx = linear_sum_assignment(dist.detach().cpu().numpy())
    # We want a map from obj_idx to slot_idx
    mapping = {o: s for s, o in zip(slot_idx, obj_idx)}
    return [mapping[i] for i in range(len(gt_colors))]


# ── 3. Training ──────────────────────────────────────────────────────────

def train_slot_pipeline(device, n_frames=1000, n_epochs=50, batch_size=32, lr=1e-3):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("  Generating data...")
    labeled_frames = generate_labeled_frames(n=n_frames, seed=99)
    train_data, val_data = labeled_frames[:800], labeled_frames[800:]

    encoder = VisualEncoder(slot_dim=LATENT_DIM).to(device)
    slot_attn = SlotAttention(num_slots=N_SLOTS, input_dim=64, slot_dim=LATENT_DIM).to(device)
    color_probe = ColorProbe(slot_dim=LATENT_DIM).to(device)
    rel_probe = RelationProbe(slot_dim=LATENT_DIM).to(device)

    params = list(encoder.parameters()) + list(slot_attn.parameters()) + \
             list(color_probe.parameters()) + list(rel_probe.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    gt_colors_t = torch.tensor(COLORS, dtype=torch.float32).to(device) / 255.0

    for epoch in range(n_epochs):
        perm = torch.randperm(len(train_data))
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            idx = perm[i:i + batch_size]
            batch = [train_data[j] for j in idx]
            frames = torch.stack([f for f, _ in batch]).to(device)
            
            # 1. Get slots
            feats = encoder(frames)
            slots = slot_attn(feats) # (B, K, D)
            
            # 2. Match slots to objects
            slot_colors = color_probe(slots) # (B, K, 3)
            
            loss = 0
            for b in range(len(batch)):
                # Match for this sample
                indices = match_slots_to_objs(slot_colors[b], gt_colors_t)
                # Color loss for matched slots
                loss += F.mse_loss(slot_colors[b][indices], gt_colors_t)
                
                # Relation loss
                pos = batch[b][1]
                rels_gt = compute_relations(pos)
                
                # We have 6 pairs of objects
                obj_slots = slots[b][indices] # (3, D) -> R, G, B
                for pi, (a, b_obj) in enumerate(PAIRS):
                    ai, bi = OBJ_IDX[a], OBJ_IDX[b_obj]
                    logits = rel_probe(obj_slots[ai].unsqueeze(0), obj_slots[bi].unsqueeze(0))
                    target = torch.tensor([float(rels_gt[(r, a, b_obj)]) for r in RELATIONS], 
                                          dtype=torch.float32).to(device).unsqueeze(0)
                    loss += F.binary_cross_entropy_with_logits(logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={total_loss/len(train_data):.4f}")

    return encoder, slot_attn, color_probe, rel_probe, val_data


# ── 4. Evaluation ─────────────────────────────────────────────────────────

def eval_slot_pipeline(encoder, slot_attn, color_probe, rel_probe, val_data, device):
    encoder.eval()
    slot_attn.eval()
    color_probe.eval()
    rel_probe.eval()
    
    gt_colors_t = torch.tensor(COLORS, dtype=torch.float32).to(device) / 255.0
    metrics = {f"{rel}_acc": [] for rel in RELATIONS}
    
    with torch.no_grad():
        for frame, pos in val_data:
            feats = encoder(frame.unsqueeze(0).to(device))
            slots = slot_attn(feats)
            slot_colors = color_probe(slots)
            indices = match_slots_to_objs(slot_colors[0], gt_colors_t)
            obj_slots = slots[0][indices]
            
            rels_gt = compute_relations(pos)
            for a, b_obj in PAIRS:
                ai, bi = OBJ_IDX[a], OBJ_IDX[b_obj]
                logits = rel_probe(obj_slots[ai].unsqueeze(0), obj_slots[bi].unsqueeze(0))
                preds = (logits > 0).float().squeeze(0)
                for ri, rel in enumerate(RELATIONS):
                    metrics[f"{rel}_acc"].append(float(preds[ri] == float(rels_gt[(rel, a, b_obj)])))
                    
    return {k: sum(v)/len(v) for k, v in metrics.items()}


# ── 6. Evaluation ─────────────────────────────────────────────────────────

def _probe_facts_for_frame(frame, pos, encoder, slot_attn, color_probe, rel_probe, device):
    """Run probe on a single frame, return fact_dict."""
    encoder.eval()
    slot_attn.eval()
    color_probe.eval()
    rel_probe.eval()
    gt_colors_t = torch.tensor(COLORS, dtype=torch.float32).to(device) / 255.0
    with torch.no_grad():
        feats = encoder(frame.unsqueeze(0).to(device))
        slots = slot_attn(feats)
        slot_colors = color_probe(slots)
        indices = match_slots_to_objs(slot_colors[0], gt_colors_t)
        obj_slots = slots[0][indices]
        
    fact_dict = {}
    for a, b_obj in PAIRS:
        ai, bi = OBJ_IDX[a], OBJ_IDX[b_obj]
        logits = rel_probe(obj_slots[ai].unsqueeze(0), obj_slots[bi].unsqueeze(0))
        preds = (logits > 0).float().squeeze(0)
        for ri, rel in enumerate(RELATIONS):
            fact_dict[(rel, a, b_obj)] = bool(preds[ri].item() > 0)
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


def run_retraction_e2e(labeled_frames, encoder, slot_attn, color_probe, rel_probe, device):
    """End-to-end retraction with probe outputs. Returns (n_active, n_correct)."""
    rng = np.random.default_rng(456)
    order = rng.permutation(len(labeled_frames))
    n_active = n_correct = 0
    for idx in order:
        if n_active >= 50:
            break
        frame, pos = labeled_frames[int(idx)]
        probe_facts = _probe_facts_for_frame(frame, pos, encoder, slot_attn, color_probe, rel_probe, device)
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


def compute_complexity_scaling(val_metrics, val_frames, encoder, slot_attn, color_probe, rel_probe, device):
    """Compute depth-2 accuracy, save plot. Returns (depth1_acc, depth2_acc) dicts."""
    bp_correct = ss_correct = total = 0
    for frame, pos in val_frames[:100]:
        probe_facts = _probe_facts_for_frame(frame, pos, encoder, slot_attn, color_probe, rel_probe, device)
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
    depth1 = {r: val_metrics[f"{r}_acc"] for r in RELATIONS}
    depth2 = {"blocked_path": bp_acc, "same_side": ss_acc}

    avg_d1 = (val_metrics["above_acc"] + val_metrics["left_of_acc"]) / 2
    compound = avg_d1 ** 2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        ["depth-1\n(above)", "depth-1\n(left_of)", "depth-2\nblocked_path", "depth-2\nsame_side"],
        [val_metrics["above_acc"], val_metrics["left_of_acc"], bp_acc, ss_acc],
        color=["steelblue", "steelblue", "darkorange", "darkorange"],
    )
    ax.axhline(compound, color="red", linestyle="--",
               label=f"compound model avg_d1²={compound:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("exp83 (Slot): Complexity Scaling (depth-1 vs depth-2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "complexity_curve.png"))
    plt.close()
    print(f"  Saved complexity_curve.png")

    return depth1, depth2


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="exp83: Slot Attention + TL relational layer")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (not supported in this PoC yet)")
    parser.add_argument("--output-dir", default=DATA_DIR,
                        help=f"Directory for runtime outputs; use {FIXTURE_DATA_DIR} only when intentionally refreshing fixtures.")
    args = parser.parse_args()
    DATA_DIR = args.output_dir

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n── 1. Training Slot Pipeline ─────────────────────────────────")
    encoder, slot_attn, color_probe, rel_probe, val_data = train_slot_pipeline(device, n_epochs=100)

    print("\n── 2. Evaluation ─────────────────────────────────────────────")
    val_metrics = eval_slot_pipeline(encoder, slot_attn, color_probe, rel_probe, val_data, device)
    print("  Val accuracy:")
    for rel in RELATIONS:
        acc = val_metrics[f"{rel}_acc"]
        gate = ("✓" if acc >= 0.9 else "✗") if rel in ("above", "left_of") else "~"
        print(f"    {gate} {rel}: acc={acc:.3f}")

    gate_probe = val_metrics["above_acc"] >= 0.9 and val_metrics["left_of_acc"] >= 0.9

    print("\n── 3. TL retraction tests ────────────────────────────────────")
    n_active_tl, n_correct_tl = run_retraction_tl_only(val_data)
    gate_tl = n_correct_tl == n_active_tl
    print(f"  TL-only:    {n_correct_tl}/{n_active_tl}  {'PASS' if gate_tl else 'FAIL'}")

    n_active_e2e, n_correct_e2e = run_retraction_e2e(val_data, encoder, slot_attn, color_probe, rel_probe, device)
    gate_e2e = n_correct_e2e >= 40
    print(f"  End-to-end: {n_correct_e2e}/{n_active_e2e}  {'PASS' if gate_e2e else 'needs ≥40'}")

    print("\n── 4. Complexity scaling ─────────────────────────────────────")
    depth1, depth2 = compute_complexity_scaling(val_metrics, val_data, encoder, slot_attn, color_probe, rel_probe, device)
    print(f"  Depth-1: above={depth1['above']:.3f}  left_of={depth1['left_of']:.3f}")
    print(f"  Depth-2: blocked_path={depth2['blocked_path']:.3f}  same_side={depth2['same_side']:.3f}")

    results = {
        "probe_val_metrics": val_metrics,
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
