"""
Phase 6a: Sample efficiency sweep — does TL's prior help in low-data regime?

Tests the bias-vs-capacity tradeoff: structural priors should win when
data is scarce; capacity should win at scale. We use 4 data sizes
(50, 200, 500, 2000 trajectories) and a FIXED COMPUTE BUDGET of 1000
gradient steps per training so the comparison is fair across sizes.
With smaller datasets, the model just sees the same data more times.

Same world as phase 5 (no collision, center occluder, single-channel,
N=2) so TL's per-cell assumption holds.

Models: TL + MLP h=64/128/256. 3 seeds each.

Decision rule:
  - TL wins at low data → structural prior pays off when data is the
    bottleneck. Cleanest positive case for TL on world modeling.
  - TL fails at low data → falsification deepens; TL's prior doesn't
    even compensate for missing data here.
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
N_GRAD_STEPS = 1000
BATCH_SIZE = 64
LR = 5e-3
SEEDS = [0, 1, 2]
MLP_HIDDENS = [64, 128, 256]
DATA_SIZES = [50, 200, 500, 2000]
EVAL_TRAJ = 400


def train_one(model, train_obs, train_true, train_a):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(train_obs)
    bs = min(BATCH_SIZE, n)
    for _ in range(N_GRAD_STEPS):
        idx = torch.randint(0, n, (bs,))
        beliefs, _ = rollout(model, train_obs[idx], train_a[idx])
        loss = F.mse_loss(beliefs[:, 1:], train_true[idx, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()


def evaluate(model, obs, true, actions):
    with torch.no_grad():
        beliefs, _ = rollout(model, obs, actions)
        preds = beliefs[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        trues = true[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        p = precision_at_k(preds, trues, N_OBJECTS)
        perm_res = in_occluder_recall(preds, trues, OCCLUDER, N_OBJECTS)
        perm = perm_res[0] if perm_res[0] is not None else float('nan')
    return p, perm


def print_table(title, results, idx=0):
    print(f"\n=== {title} (mean ± std across {len(SEEDS)} seeds) ===")
    header = f"  {'model':<12s} | " + " | ".join(f"{'n='+str(d):^13s}" for d in DATA_SIZES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in results:
        label = 'TL' if key[0] == 'TL' else f"MLP h={key[1]}"
        cells = []
        for ds in DATA_SIZES:
            vals = [v[idx] for v in results[key][ds] if v[idx] == v[idx]]
            if not vals:
                cells.append("    n/a    ")
                continue
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mean*100:5.1f}± {std*100:4.1f}")
        print(f"  {label:<12s} | " + " | ".join(cells))


def main():
    print("=== Phase 6a: Sample efficiency — TL vs MLP ===")
    print(f"  Data sizes: {DATA_SIZES} trajectories")
    print(f"  Fixed compute: {N_GRAD_STEPS} gradient steps per training")
    print(f"  Models: TL + MLP h={MLP_HIDDENS}, seeds={SEEDS}")
    print(f"  World: N={N_OBJECTS}, occluder={OCCLUDER}, NO collision\n")

    results = {}
    t_start = time.time()

    for seed in SEEDS:
        print(f"--- seed={seed} ---")
        world_train = GridWorld(size=8, n_objects=N_OBJECTS, seed=seed * 100,
                                collision=False, single_channel=True,
                                occluder_zone=OCCLUDER)
        all_obs, all_true, all_a = collect_episodes(
            world_train, n_traj=max(DATA_SIZES), length=TRAJ_LEN)

        world_eval = GridWorld(size=8, n_objects=N_OBJECTS, seed=seed * 100 + 999,
                               collision=False, single_channel=True,
                               occluder_zone=OCCLUDER)
        eval_obs, eval_true, eval_a = collect_episodes(
            world_eval, n_traj=EVAL_TRAJ, length=TRAJ_LEN)

        for data_size in DATA_SIZES:
            sub_obs = all_obs[:data_size]
            sub_true = all_true[:data_size]
            sub_a = all_a[:data_size]

            torch.manual_seed(seed)
            tl_model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
            t0 = time.time()
            train_one(tl_model, sub_obs, sub_true, sub_a)
            p, perm = evaluate(tl_model, eval_obs, eval_true, eval_a)
            t = time.time() - t0
            results.setdefault(('TL', None), {}).setdefault(data_size, []).append((p, perm))
            print(f"  n={data_size:>4d}  TL         P@2={p:.2%}  in-occ={perm:.2%}  ({t:.1f}s)")

            for h in MLP_HIDDENS:
                torch.manual_seed(seed)
                mlp_model = MLPForwardModel(grid=8, n_actions=N_ACTIONS, hidden=h)
                t0 = time.time()
                train_one(mlp_model, sub_obs, sub_true, sub_a)
                p, perm = evaluate(mlp_model, eval_obs, eval_true, eval_a)
                t = time.time() - t0
                results.setdefault(('MLP', h), {}).setdefault(data_size, []).append((p, perm))
                print(f"  n={data_size:>4d}  MLP h={h:<3d}  P@2={p:.2%}  in-occ={perm:.2%}  ({t:.1f}s)")
        print()

    print(f"Total wall time: {time.time()-t_start:.1f}s")

    print_table("P@2", results, idx=0)
    print_table("In-occluder recall", results, idx=1)

    print("\n=== Decision rule: TL vs MLP h=64 P@2 across data sizes ===")
    for ds in DATA_SIZES:
        tl_vals = [v[0] for v in results[('TL', None)][ds]]
        mlp_vals = [v[0] for v in results[('MLP', 64)][ds]]
        tl_mean = statistics.mean(tl_vals)
        mlp_mean = statistics.mean(mlp_vals)
        gap = (tl_mean - mlp_mean) * 100
        verdict = "TL wins" if gap > 1 else ("near-tie" if abs(gap) <= 1 else "MLP wins")
        print(f"  n={ds:>4d}  TL={tl_mean*100:5.1f}%  MLP h=64={mlp_mean*100:5.1f}%  gap={gap:+.1f}pp  → {verdict}")

    print("\n=== Decision rule: TL vs MLP h=64 in-occluder recall across data sizes ===")
    for ds in DATA_SIZES:
        tl_vals = [v[1] for v in results[('TL', None)][ds] if v[1] == v[1]]
        mlp_vals = [v[1] for v in results[('MLP', 64)][ds] if v[1] == v[1]]
        if not tl_vals or not mlp_vals:
            continue
        tl_mean = statistics.mean(tl_vals)
        mlp_mean = statistics.mean(mlp_vals)
        gap = (tl_mean - mlp_mean) * 100
        verdict = "TL wins" if gap > 1 else ("near-tie" if abs(gap) <= 1 else "MLP wins")
        print(f"  n={ds:>4d}  TL={tl_mean*100:5.1f}%  MLP h=64={mlp_mean*100:5.1f}%  gap={gap:+.1f}pp  → {verdict}")


if __name__ == "__main__":
    main()
