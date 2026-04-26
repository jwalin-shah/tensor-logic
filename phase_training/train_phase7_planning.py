"""
Phase 7: Multi-task transfer — does TL's sample efficiency compound to planning?

Tests the actual research ambition: does a sample-efficient forward model
(TL at low data) translate to better downstream task performance via
planning? This is the "teach navigation → easier pickup" question in
its simplest form.

Setup:
- World: 8×8 grid, 1 object, 4 actions, NO collision, NO occluder, fully
  observable (the SIMPLEST setup so the forward-model quality drives the
  result; harder worlds add confounds).
- Train forward model ONCE on n random-action trajectories.
- For each of 4 tasks (navigate to TL/TR/BL/BR corner):
    Run N episodes from random starts. At each step:
      1. Plan: sample K random action sequences of horizon H, predict each
         with the forward model, score by goal-cell probability at horizon.
      2. Execute the first action of the best sequence in the TRUE world.
      3. Repeat until reach goal or budget exhausted.
- Metrics: success rate, average steps to success.

Hypothesis (TL claim, steel-manned):
  At low n, TL's sample-efficient forward model gives better planning than
  MLP because TL's prior fills in dynamics knowledge MLP hasn't yet learned
  from the limited data. The advantage should shrink as n grows.

Models: TL + MLP h=64 + MLP h=128. Seeds: 3.
Data sizes: 50 (low), 200 (medium), 2000 (high).
"""

import time
import statistics
import torch
import torch.nn.functional as F

from world import GridWorld, N_ACTIONS, ACTIONS
from tl_model import TLForwardModel
from mlp_model import MLPForwardModel


TASKS = [
    ('goto_TL', (0, 0)),
    ('goto_TR', (0, 7)),
    ('goto_BL', (7, 0)),
    ('goto_BR', (7, 7)),
]
N_EPISODES = 30
MAX_STEPS = 15
PLAN_H = 10  # cover full Manhattan distance across 8×8 board
PLAN_K = 50  # more samples to cover 4^10 sequence space
TRAJ_LEN = 20
DATA_SIZES = [50, 200, 2000]
SEEDS = [0, 1, 2]
N_GRAD_STEPS = 1000
BATCH_SIZE = 64
LR = 5e-3
MLP_HIDDENS = [64, 128]


def collect_random_trajectories(world, n, length):
    states = []
    actions = []
    for _ in range(n):
        positions = world.reset()
        traj_states = [world.to_true_single(positions)]
        traj_actions = []
        for _ in range(length):
            a = world.rng.randint(0, N_ACTIONS - 1)
            positions = world.step(positions, a)
            traj_states.append(world.to_true_single(positions))
            traj_actions.append(a)
        states.append(torch.stack(traj_states))
        actions.append(torch.tensor(traj_actions, dtype=torch.long))
    return torch.stack(states), torch.stack(actions)


def train_forward_model(model, train_states, train_actions):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_traj = len(train_states)
    bs = min(BATCH_SIZE, n_traj)
    T = train_actions.shape[1]
    for _ in range(N_GRAD_STEPS):
        traj_idx = torch.randint(0, n_traj, (bs,))
        time_idx = torch.randint(0, T, (bs,))
        s_t = train_states[traj_idx, time_idx]
        a_t = train_actions[traj_idx, time_idx]
        s_next = train_states[traj_idx, time_idx + 1]
        s_pred = model(s_t, a_t)
        loss = F.mse_loss(s_pred, s_next)
        opt.zero_grad()
        loss.backward()
        opt.step()


def plan_action(model, state, goal_xy, horizon=PLAN_H, n_samples=PLAN_K):
    """Sampling-MPC: pick action sequence by discounted cumulative goal-prob.

    Why cumulative-not-final: with sharp forward-model predictions (TL),
    mass only sits AT the goal at one specific horizon step; final-step
    scoring gives near-zero signal for almost all sequences. Summing over
    the horizon credits any sequence whose predicted trajectory passes
    through the goal at any time.
    """
    action_seqs = torch.randint(0, N_ACTIONS, (n_samples, horizon))
    s = state.unsqueeze(0).expand(n_samples, -1, -1, -1).contiguous()
    gx, gy = goal_xy
    score = torch.zeros(n_samples)
    discount = 1.0
    with torch.no_grad():
        for h in range(horizon):
            a = action_seqs[:, h]
            s = model(s, a)
            score = score + discount * s[:, 0, gx, gy]
            discount *= 0.9
    best = score.argmax().item()
    return action_seqs[best, 0].item()


def run_episode(world, model, goal_xy, max_steps=MAX_STEPS):
    positions = world.reset()
    for step in range(max_steps):
        if positions[0] == goal_xy:
            return True, step
        state = world.to_true_single(positions)
        action = plan_action(model, state, goal_xy)
        positions = world.step(positions, action)
    if positions[0] == goal_xy:
        return True, max_steps
    return False, max_steps


def main():
    print("=== Phase 7: Multi-task planning with learned dynamics ===")
    print(f"  World: 8×8, 1 object, 4 actions, no collision, no occluder")
    print(f"  Tasks: {[t[0] for t in TASKS]}")
    print(f"  Data sizes: {DATA_SIZES}, Seeds: {SEEDS}")
    print(f"  Planning: H={PLAN_H}, K={PLAN_K} samples, max_steps={MAX_STEPS}")
    print(f"  Episodes per (model, task, seed): {N_EPISODES}\n")

    # results: (model_class, hidden, data_size, task) -> [success_rate per seed]
    results = {}
    avg_steps_results = {}
    t_start = time.time()

    model_configs = [('TL', None)] + [('MLP', h) for h in MLP_HIDDENS]

    for seed in SEEDS:
        print(f"--- seed={seed} ---")
        for ds in DATA_SIZES:
            world_train = GridWorld(size=8, n_objects=1, seed=seed * 100,
                                    collision=False, single_channel=True,
                                    occluder_zone=None)
            train_states, train_actions = collect_random_trajectories(
                world_train, n=ds, length=TRAJ_LEN)

            for model_class, hidden in model_configs:
                torch.manual_seed(seed)
                if model_class == 'TL':
                    model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
                else:
                    model = MLPForwardModel(grid=8, n_actions=N_ACTIONS, hidden=hidden)
                t0 = time.time()
                train_forward_model(model, train_states, train_actions)
                t_train = time.time() - t0

                model.eval()
                t1 = time.time()
                for task_name, goal_xy in TASKS:
                    eval_world = GridWorld(size=8, n_objects=1,
                                           seed=seed * 100 + 9000,
                                           collision=False, single_channel=True,
                                           occluder_zone=None)
                    successes = 0
                    total_steps = 0
                    for _ in range(N_EPISODES):
                        success, steps = run_episode(eval_world, model, goal_xy)
                        if success:
                            successes += 1
                            total_steps += steps
                    rate = successes / N_EPISODES
                    avg_steps = total_steps / max(1, successes)
                    label = 'TL' if model_class == 'TL' else f"MLP h={hidden}"
                    key = (model_class, hidden, ds, task_name)
                    results.setdefault(key, []).append(rate)
                    avg_steps_results.setdefault(key, []).append(avg_steps)
                model.train()
                t_eval = time.time() - t1
                print(f"  n={ds:>4d}  {label:<10s}  trained ({t_train:.1f}s)  evaled ({t_eval:.1f}s)")
        print()

    print(f"Total wall time: {time.time()-t_start:.1f}s\n")

    # Per-task success rates
    print("=== Success rate per task (mean across seeds) ===")
    for task_name, _ in TASKS:
        print(f"\n  Task: {task_name}")
        print(f"    {'model':<12s} | " + " | ".join(f"{'n='+str(d):^14s}" for d in DATA_SIZES))
        for model_class, hidden in model_configs:
            label = 'TL' if model_class == 'TL' else f"MLP h={hidden}"
            cells = []
            for ds in DATA_SIZES:
                key = (model_class, hidden, ds, task_name)
                vals = results.get(key, [])
                if not vals:
                    cells.append("    n/a    ")
                    continue
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                cells.append(f"{mean*100:5.1f}± {std*100:4.1f}")
            print(f"    {label:<12s} | " + " | ".join(cells))

    # Aggregated across tasks
    print("\n=== Aggregated success rate (mean across tasks AND seeds) ===")
    print(f"  {'model':<12s} | " + " | ".join(f"{'n='+str(d):^14s}" for d in DATA_SIZES))
    for model_class, hidden in model_configs:
        label = 'TL' if model_class == 'TL' else f"MLP h={hidden}"
        cells = []
        for ds in DATA_SIZES:
            per_seed_means = []
            for seed_idx in range(len(SEEDS)):
                task_rates = []
                for task_name, _ in TASKS:
                    key = (model_class, hidden, ds, task_name)
                    if seed_idx < len(results.get(key, [])):
                        task_rates.append(results[key][seed_idx])
                if task_rates:
                    per_seed_means.append(statistics.mean(task_rates))
            if not per_seed_means:
                cells.append("    n/a    ")
                continue
            mean = statistics.mean(per_seed_means)
            std = statistics.stdev(per_seed_means) if len(per_seed_means) > 1 else 0.0
            cells.append(f"{mean*100:5.1f}± {std*100:4.1f}")
        print(f"  {label:<12s} | " + " | ".join(cells))

    # Avg steps to success (efficiency, only over successful episodes)
    print("\n=== Avg steps to success (lower = more efficient planning) ===")
    print(f"  {'model':<12s} | " + " | ".join(f"{'n='+str(d):^14s}" for d in DATA_SIZES))
    for model_class, hidden in model_configs:
        label = 'TL' if model_class == 'TL' else f"MLP h={hidden}"
        cells = []
        for ds in DATA_SIZES:
            all_vals = []
            for task_name, _ in TASKS:
                key = (model_class, hidden, ds, task_name)
                all_vals.extend(avg_steps_results.get(key, []))
            if not all_vals:
                cells.append("    n/a    ")
                continue
            mean = statistics.mean(all_vals)
            std = statistics.stdev(all_vals) if len(all_vals) > 1 else 0.0
            cells.append(f"{mean:5.2f}± {std:4.2f}")
        print(f"  {label:<12s} | " + " | ".join(cells))

    # Decision rule: TL vs MLP h=64 success rate at low data
    print("\n=== Decision rule: TL vs MLP h=64 aggregated success rate ===")
    for ds in DATA_SIZES:
        tl_rates = []
        mlp_rates = []
        for seed_idx in range(len(SEEDS)):
            tl_task = []
            mlp_task = []
            for task_name, _ in TASKS:
                tl_key = ('TL', None, ds, task_name)
                mlp_key = ('MLP', 64, ds, task_name)
                if seed_idx < len(results.get(tl_key, [])):
                    tl_task.append(results[tl_key][seed_idx])
                if seed_idx < len(results.get(mlp_key, [])):
                    mlp_task.append(results[mlp_key][seed_idx])
            if tl_task:
                tl_rates.append(statistics.mean(tl_task))
            if mlp_task:
                mlp_rates.append(statistics.mean(mlp_task))
        tl_mean = statistics.mean(tl_rates) if tl_rates else 0
        mlp_mean = statistics.mean(mlp_rates) if mlp_rates else 0
        gap = (tl_mean - mlp_mean) * 100
        verdict = "TL wins" if gap > 2 else ("near-tie" if abs(gap) <= 2 else "MLP wins")
        print(f"  n={ds:>4d}  TL={tl_mean*100:5.1f}%  MLP h=64={mlp_mean*100:5.1f}%  gap={gap:+.1f}pp  → {verdict}")


if __name__ == "__main__":
    main()
