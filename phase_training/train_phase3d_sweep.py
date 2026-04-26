"""
Phase 3d: multi-seed sweep — TL vs MLP world models with belief tensor.

Runs 3 seeds × 4 configs (TL + MLP hidden=64/128/256) = 12 runs total.
Re-collects trajectories per seed (so trajectory distribution varies too,
not just model init). Reports mean ± std on P@2 and in-occluder recall
to test whether the single-seed phase 3c result (TL > MLP by 3pp/9pp) is
robust to seed and MLP hyperparameter choice.

Decision rule:
  - TL > MLP by >1pp on in-occluder across all settings → real result
  - Gap shrinks to noise on better MLP HPs → memory does most of the work
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


OCCLUDER = (3, 5, 3, 5)
N_OBJECTS = 2
TRAJ_LEN = 20
N_EPOCHS = 30
BATCH_SIZE = 64
LR = 5e-3
SEEDS = [0, 1, 2]
MLP_HIDDENS = [64, 128, 256]


def train_one(model, train_obs, train_true, train_a, val_obs, val_true, val_a):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, N_EPOCHS + 1):
        idx = torch.randperm(len(train_obs))
        for start in range(0, len(train_obs), BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            beliefs, _ = rollout(model, train_obs[b], train_a[b])
            loss = F.mse_loss(beliefs[:, 1:], train_true[b, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        val_beliefs, _ = rollout(model, val_obs, val_a)
        preds = val_beliefs[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        trues = val_true[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        p = precision_at_k(preds, trues, N_OBJECTS)
        perm_res = in_occluder_recall(preds, trues, OCCLUDER, N_OBJECTS)
        perm = perm_res[0] if perm_res[0] is not None else float('nan')
    return p, perm


def n_params_mlp(h):
    return 68 * h + h + h * 64 + 64


def main():
    print("=== Phase 3d: multi-seed sweep — TL vs MLP world models ===")
    print(f"  Seeds: {SEEDS}, MLP hiddens: {MLP_HIDDENS}")
    print(f"  Epochs: {N_EPOCHS}, batch: {BATCH_SIZE}, lr: {LR}\n")

    results = {}  # (model_name, hidden_or_None) -> list of (p, perm) per seed
    t_start = time.time()

    for seed in SEEDS:
        print(f"--- seed={seed} ---")

        world_train = GridWorld(size=8, n_objects=N_OBJECTS, seed=seed * 100 + 0,
                                collision=True, single_channel=True, occluder_zone=OCCLUDER)
        world_val = GridWorld(size=8, n_objects=N_OBJECTS, seed=seed * 100 + 1,
                              collision=True, single_channel=True, occluder_zone=OCCLUDER)
        train_obs, train_true, train_a = collect_episodes(world_train, n_traj=2000, length=TRAJ_LEN)
        val_obs, val_true, val_a = collect_episodes(world_val, n_traj=400, length=TRAJ_LEN)

        torch.manual_seed(seed)
        tl_model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
        t0 = time.time()
        p, perm = train_one(tl_model, train_obs, train_true, train_a, val_obs, val_true, val_a)
        results.setdefault(('TL', None), []).append((p, perm))
        print(f"  TL              P@2={p:.2%}  in-occ={perm:.2%}  ({time.time()-t0:.1f}s)")

        for h in MLP_HIDDENS:
            torch.manual_seed(seed)
            mlp_model = MLPForwardModel(grid=8, n_actions=N_ACTIONS, hidden=h)
            t0 = time.time()
            p, perm = train_one(mlp_model, train_obs, train_true, train_a, val_obs, val_true, val_a)
            results.setdefault(('MLP', h), []).append((p, perm))
            print(f"  MLP h={h:<3d}      P@2={p:.2%}  in-occ={perm:.2%}  ({time.time()-t0:.1f}s)")
        print()

    print(f"Total wall time: {time.time()-t_start:.1f}s\n")

    print("=== Aggregated results (mean ± std across seeds) ===")
    print(f"  {'model':<12s}  {'params':>8s}  {'P@2':>18s}  {'in-occ recall':>18s}")
    for key, vals in results.items():
        ps = [v[0] for v in vals]
        ms = [v[1] for v in vals]
        if key[0] == 'TL':
            n_p = 4 * 8 * 8 * 8 * 8
            label = 'TL'
        else:
            n_p = n_params_mlp(key[1])
            label = f"MLP h={key[1]}"
        p_mean, p_std = statistics.mean(ps), (statistics.stdev(ps) if len(ps) > 1 else 0.0)
        m_mean, m_std = statistics.mean(ms), (statistics.stdev(ms) if len(ms) > 1 else 0.0)
        print(f"  {label:<12s}  {n_p:>8,}  "
              f"{p_mean*100:6.2f}% ± {p_std*100:4.2f}    "
              f"{m_mean*100:6.2f}% ± {m_std*100:4.2f}")

    print("\n=== Per-seed details ===")
    for key, vals in results.items():
        label = 'TL' if key[0] == 'TL' else f"MLP h={key[1]}"
        seed_strs = [f"s{s}: P@2={v[0]:.2%} in-occ={v[1]:.2%}" for s, v in zip(SEEDS, vals)]
        print(f"  {label:<12s} | " + " | ".join(seed_strs))

    # Decision rule output
    print("\n=== Decision rule check ===")
    tl_in_occ = [v[1] for v in results[('TL', None)]]
    tl_mean = statistics.mean(tl_in_occ)
    print(f"  TL mean in-occ recall:  {tl_mean*100:.2f}%")
    for h in MLP_HIDDENS:
        mlp_in_occ = [v[1] for v in results[('MLP', h)]]
        mlp_mean = statistics.mean(mlp_in_occ)
        gap = (tl_mean - mlp_mean) * 100
        verdict = "TL wins" if gap > 1.0 else ("near-tie" if abs(gap) <= 1.0 else "MLP wins")
        print(f"  vs MLP h={h:<3d} mean={mlp_mean*100:6.2f}%  gap={gap:+.2f}pp  → {verdict}")


if __name__ == "__main__":
    main()
