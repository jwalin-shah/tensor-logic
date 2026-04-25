"""
Phase 5: Generalization sweep — does TL's per-cell prior give OOD transfer?

Tests whether TL's einsum factorization buys out-of-distribution
generalization that a generic MLP doesn't get. Train on ONE canonical
world; evaluate on transferred worlds where the model has never seen
the test condition.

  Train condition:
    N=2 objects, occluder=(3,5,3,5) [center], NO collision, T=20

  Transfer conditions:
    1. In-distribution      (N=2, center occluder)         [sanity]
    2. Object-count: N=1    (single object, center occluder)
    3. Object-count: N=3    (more objects than training)
    4. Object-count: N=4    (even more objects)
    5. Occluder shifted TL  (occluder at top-left)
    6. Occluder shifted BR  (occluder at bottom-right)

NO collision so TL's per-cell factorization assumption actually holds
(phase 3d already showed TL's structural ceiling under collision).
This isolates the question: does TL's prior give generalization?

Decision rule:
  - TL's accuracy degradation from in-dist → transfer should be SMALLER
    than MLP's. If TL holds within 2pp on count transfer while MLP
    drops >5pp, the structural prior buys us OOD generalization.
  - If TL transfers no better than MLP, TL's einsum has no inductive-
    bias advantage on this family of tasks.

Models: TL, MLP h=64/128/256. Seeds: 3.
"""

import time
import statistics
import torch
import torch.nn.functional as F

from world import GridWorld, N_ACTIONS
from tl_model import TLForwardModel
from mlp_model import MLPForwardModel
from train_phase3b import (
    collect_episodes,
    rollout,
    precision_at_k,
    in_occluder_recall,
)


TRAIN_OCCLUDER = (3, 5, 3, 5)
TRAJ_LEN = 20
N_EPOCHS = 30
BATCH_SIZE = 64
LR = 5e-3
SEEDS = [0, 1, 2]
MLP_HIDDENS = [64, 128, 256]

# (label, n_objects, occluder)
TRANSFER_CONDITIONS = [
    ('in-dist',       2, TRAIN_OCCLUDER),
    ('count N=1',     1, TRAIN_OCCLUDER),
    ('count N=3',     3, TRAIN_OCCLUDER),
    ('count N=4',     4, TRAIN_OCCLUDER),
    ('occ shift TL',  2, (1, 3, 1, 3)),
    ('occ shift BR',  2, (5, 7, 5, 7)),
]


def evaluate(model, condition, seed_offset):
    label, n_objects, occluder = condition
    world = GridWorld(size=8, n_objects=n_objects, seed=seed_offset + 1000,
                      collision=False, single_channel=True,
                      occluder_zone=occluder)
    obs, true, actions = collect_episodes(world, n_traj=400, length=TRAJ_LEN)

    with torch.no_grad():
        beliefs, _ = rollout(model, obs, actions)
        preds = beliefs[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        trues = true[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        p = precision_at_k(preds, trues, n_objects)
        perm_res = in_occluder_recall(preds, trues, occluder, n_objects)
        perm = perm_res[0] if perm_res[0] is not None else float('nan')
    return p, perm


def train_one(model, train_obs, train_true, train_a):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(N_EPOCHS):
        idx = torch.randperm(len(train_obs))
        for start in range(0, len(train_obs), BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            beliefs, _ = rollout(model, train_obs[b], train_a[b])
            loss = F.mse_loss(beliefs[:, 1:], train_true[b, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()


def fmt_pct(mean, std):
    if mean != mean:  # NaN
        return "  n/a       "
    return f"{mean*100:5.1f}±{std*100:4.1f}"


def print_table(title, results):
    print(f"\n=== {title} (mean ± std across {len(SEEDS)} seeds) ===")
    cond_labels = [c[0] for c in TRANSFER_CONDITIONS]
    header = f"  {'model':<12s} | " + " | ".join(f"{cl:^11s}" for cl in cond_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in results:
        label = 'TL' if key[0] == 'TL' else f"MLP h={key[1]}"
        cells = []
        for cl in cond_labels:
            vals = [v for v in results[key][cl] if v == v]  # filter NaN
            if not vals:
                cells.append("    n/a    ")
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(fmt_pct(mean, std))
        print(f"  {label:<12s} | " + " | ".join(cells))


def main():
    print("=== Phase 5: TL vs MLP transfer to new world conditions ===")
    print(f"  TRAIN: N=2, occluder={TRAIN_OCCLUDER}, NO collision, T={TRAJ_LEN}")
    print(f"  TRANSFER: {[c[0] for c in TRANSFER_CONDITIONS]}")
    print(f"  Models: TL + MLP h={MLP_HIDDENS}, seeds={SEEDS}\n")

    p_results = {}
    perm_results = {}
    t_start = time.time()

    for seed in SEEDS:
        print(f"--- seed={seed} ---")
        world_train = GridWorld(size=8, n_objects=2, seed=seed * 100,
                                collision=False, single_channel=True,
                                occluder_zone=TRAIN_OCCLUDER)
        train_obs, train_true, train_a = collect_episodes(
            world_train, n_traj=2000, length=TRAJ_LEN)

        # TL
        torch.manual_seed(seed)
        tl_model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
        t0 = time.time()
        train_one(tl_model, train_obs, train_true, train_a)
        print(f"  TL trained ({time.time()-t0:.1f}s)")
        for cond in TRANSFER_CONDITIONS:
            p, perm = evaluate(tl_model, cond, seed * 100)
            key = ('TL', None)
            p_results.setdefault(key, {}).setdefault(cond[0], []).append(p)
            perm_results.setdefault(key, {}).setdefault(cond[0], []).append(perm)
            print(f"    {cond[0]:<14s}  P@{cond[1]}={p:.2%}  in-occ={perm:.2%}")

        # MLPs
        for h in MLP_HIDDENS:
            torch.manual_seed(seed)
            mlp_model = MLPForwardModel(grid=8, n_actions=N_ACTIONS, hidden=h)
            t0 = time.time()
            train_one(mlp_model, train_obs, train_true, train_a)
            print(f"  MLP h={h} trained ({time.time()-t0:.1f}s)")
            for cond in TRANSFER_CONDITIONS:
                p, perm = evaluate(mlp_model, cond, seed * 100)
                key = ('MLP', h)
                p_results.setdefault(key, {}).setdefault(cond[0], []).append(p)
                perm_results.setdefault(key, {}).setdefault(cond[0], []).append(perm)
                print(f"    {cond[0]:<14s}  P@{cond[1]}={p:.2%}  in-occ={perm:.2%}")
        print()

    print(f"Total wall time: {time.time()-t_start:.1f}s")

    print_table("P@N (top-N where N = true object count)", p_results)
    print_table("In-occluder recall", perm_results)

    print("\n=== Transfer degradation: P@N drop from in-dist (mean across seeds) ===")
    cond_labels = [c[0] for c in TRANSFER_CONDITIONS]
    for key in p_results:
        label = 'TL' if key[0] == 'TL' else f"MLP h={key[1]}"
        in_dist = statistics.mean(p_results[key]['in-dist'])
        parts = [f"in-dist={in_dist*100:.1f}%"]
        for cl in cond_labels[1:]:
            transfer = statistics.mean(p_results[key][cl])
            delta = (transfer - in_dist) * 100
            parts.append(f"{cl}: {delta:+.1f}pp")
        print(f"  {label:<12s} | " + " | ".join(parts))

    print("\n=== Decision rule check (TL must drop LESS than MLP under transfer) ===")
    transfer_axes = ['count N=1', 'count N=3', 'count N=4', 'occ shift TL', 'occ shift BR']
    tl_in_dist = statistics.mean(p_results[('TL', None)]['in-dist'])
    for axis in transfer_axes:
        tl_transfer = statistics.mean(p_results[('TL', None)][axis])
        tl_drop = (tl_in_dist - tl_transfer) * 100
        verdict_parts = [f"TL drop={tl_drop:+.1f}pp"]
        all_better = True
        for h in MLP_HIDDENS:
            mlp_in_dist = statistics.mean(p_results[('MLP', h)]['in-dist'])
            mlp_transfer = statistics.mean(p_results[('MLP', h)][axis])
            mlp_drop = (mlp_in_dist - mlp_transfer) * 100
            if tl_drop > mlp_drop:  # TL drops MORE → MLP transfers better
                all_better = False
            verdict_parts.append(f"MLP h={h} drop={mlp_drop:+.1f}pp")
        verdict = "TL transfers BETTER" if all_better else "TL does NOT transfer better"
        print(f"  {axis:<14s}: {' | '.join(verdict_parts)}  → {verdict}")


if __name__ == "__main__":
    main()
