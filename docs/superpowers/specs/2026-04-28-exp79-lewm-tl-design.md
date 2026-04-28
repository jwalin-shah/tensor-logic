# exp79: LeWM + TL Relational Layer

**Date:** 2026-04-28  
**File:** `experiments/exp79_lewm_tl.py`  
**Cost:** Mac MPS, ~1 hour total  
**Status:** Design approved, pending implementation

---

## Goal

First end-to-end perception→reasoning pipeline in the repo. Train a minimal JEPA world model on synthetic pixel scenes, extract discrete relational facts via a linear probe, wire those facts into TL fixpoint reasoning, and verify counterfactual retraction.

Claim being tested: **a holistic pixel latent encodes enough relational structure for a linear probe to decode binary object relations, and TL fixpoint can derive higher-order facts from those relations with predictable (not catastrophic) degradation at depth-2.**

Informative failure mode: if the linear probe fails (accuracy <70%), the latent doesn't encode discrete relations cleanly → need an object-centric encoder (Slot Attention) instead of a holistic one.

---

## Section 1: Data Generation

- 10k sequences × 8 frames each, pure numpy
- Frame: 64×64 RGB, 3 colored squares (red/green/blue, ~10px side)
- Physics: constant velocity + bounce at walls, random initial positions/velocities
- Ground-truth relations computed geometrically per frame:
  - `above(A,B)` — center_y(A) < center_y(B)
  - `left_of(A,B)` — center_x(A) < center_x(B)
  - `touching(A,B)` — bounding boxes gap < 2px
  - `occluded(A,B)` — A's bbox overlaps B's and A has higher draw-order index
- Train split: 9k sequences. Probe label split: 500 labeled frames held out (400 train / 100 val).
- Saved to: `experiments/exp79_data/`

---

## Section 2: JEPA Encoder + Predictor

**Architecture (~633K params):**
- Encoder: 3-layer CNN (3→32→64→128 channels, stride-2 convs) → flatten → linear → z ∈ ℝ⁶⁴
- Predictor: 2-layer MLP (64→128→64), takes z_t, predicts ẑ_{t+1}

**Losses:**
- `L_pred = MSE(ẑ_{t+1}, sg(z_{t+1}))` — stop-gradient on target prevents collapse
- `L_var = mean(max(0, 1 − std(z_t, dim=batch)))` — VICReg-style variance term, prevents zero-collapse
- `L_reg = 0.01 × ||z_t||²` — keeps latent bounded
- `L = L_pred + L_var + L_reg`

**Training:** Adam lr=1e-3, batch=128, 20 epochs, MPS. ~20 min. Random seed 42. Device: `mps` if available, else `cpu`.

**Saved to:** `experiments/exp79_data/encoder.pt`

**CLI:** `--skip-train` reuses saved `encoder.pt` and `probe.pt` (if both exist), skipping sections 2 and 3.

---

## Section 3: Linear Probe

- Freeze encoder after JEPA training
- 3 objects → 6 ordered pairs × 4 relations = 24 binary classifiers
- Single shared linear head: input = [z (64-dim) ‖ pair_one_hot (6-dim)] = 70-dim → sigmoid
- BCELoss, Adam lr=1e-2, 50 epochs, no regularization
- Val split: 100 labeled frames

**Falsification gate:** `above` and `left_of` val accuracy ≥90%  
**Logged but not gated:** `touching`, `occluded` (expected lower — holistic latent limitation)

**Saved to:** `experiments/exp79_data/probe.pt`

---

## Section 4: TL Wiring + Fixpoint

Threshold probe outputs at 0.5 → binary facts. Wire into `tensor_logic`:
- `Domain(["R", "G", "B"])`
- Relations: `above`, `left_of`, `touching`, `occluded` (2-ary)
- Derived rules (raw tensor ops, no tag protocol):
  - `blocked_path[X,Z] = (touching[X,:] @ above[:,Z]).clamp(0,1)` — 2-ary; X touches something above Z
  - `same_side[X,Z] = (left_of[X,:] @ left_of[:,Z]).clamp(0,1)` — 2-ary; transitive left
  - `clear_above[X] = 1 - above[:,X].max(dim=0).values` — **1-ary** (`Relation("clear_above", domain)`); nothing is above X

Run `fixpoint()` → derived relation tensors.

---

## Section 5: Evaluation

### TL-Only Retraction Test (pure logic correctness)
- 50 test frames, **only frames where B participates in ≥1 derived fact pre-removal** (skip vacuous frames)
- Feed **ground-truth labels** as TL facts
- Remove all facts where B appears as either argument (both `rel(B,X)` and `rel(X,B)`), re-run fixpoint
- Check retracted derived facts match ground-truth geometry (recomputed without B)
- **Gate:** 100% of active frames — if this fails, the rules are wrong, not the probe

### End-to-End Retraction Test
- Same 50 frames, feed **probe outputs** as TL facts
- **Gate:** ≥40/50 retraction matches

### Complexity Scaling (depth 1 vs 2)
- Depth-1 accuracy: direct probe val accuracy per relation (`above`, `left_of`, `touching`, `occluded`)
- Depth-2 accuracy: `blocked_path` and `same_side` accuracy vs ground-truth geometry (computed analytically from object positions)
- Plot both with expected compound curve (probe_acc²) overlaid
- **Expected result:** depth-2 accuracy follows compound model — degradation is predictable, not catastrophic (contrast with "Illusion of Thinking" LRM collapse)
- Saved to: `experiments/exp79_data/complexity_curve.png`

### Results
All metrics saved to `experiments/exp79_data/results.json`.

---

## Connection to Prior Work

- exp27: TL as world model (blocks world) — exp79 closes the perception bridge exp27 left open
- exp55/58/59: Hanoi complexity — same complexity-scaling thread, now with perception
- exp66: Datalog negation — `clear_above` uses the same `(1 - tensor)` complement pattern
- "Illusion of Thinking" (Shojaee et al., Apple 2026): complexity scaling evaluation design
- LeWorldModel (arXiv 2603.19312): JEPA architecture inspiration; we implement a minimal variant (~633K params vs paper's 15M)
