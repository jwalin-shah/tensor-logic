"""
Curiosity-driven exploration in a tiny gridworld.

Efficient setup: pure Python, no external dependencies.

Intrinsic reward is one-step prediction error of a tabular world model:
    r_int(s, a, s') = -log p_model(s' | s, a)

The curiosity policy chooses actions with highest estimated surprise,
and we compare it against a random baseline.
"""

import math
import random
import statistics
from dataclasses import dataclass


ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def state_to_idx(x: int, y: int, size: int) -> int:
    return x * size + y


@dataclass
class GridWorld:
    size: int = 7

    def step(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dx, dy = ACTIONS[action]
        nx = min(self.size - 1, max(0, pos[0] + dx))
        ny = min(self.size - 1, max(0, pos[1] + dy))
        return nx, ny


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def run_episode(seed: int, policy: str, steps: int = 250, grid_size: int = 7) -> dict:
    assert policy in {"curiosity", "random"}
    rng = random.Random(seed)

    env = GridWorld(size=grid_size)
    n_states = grid_size * grid_size
    n_actions = len(ACTIONS)

    # Transition model counts: counts[s][a][s_next]
    counts = [[[0 for _ in range(n_states)] for _ in range(n_actions)] for _ in range(n_states)]

    # Estimated novelty per (state, action), initialized optimistically high.
    novelty = [[4.0 for _ in range(n_actions)] for _ in range(n_states)]

    pos = (grid_size // 2, grid_size // 2)
    visited = {pos}
    coverage = []

    corners = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]
    first_corner_hit = {c: None for c in corners}

    for t in range(steps):
        s = state_to_idx(pos[0], pos[1], grid_size)

        if policy == "random":
            action = rng.randrange(n_actions)
        else:
            if rng.random() < 0.1:
                action = rng.randrange(n_actions)
            else:
                action = max(range(n_actions), key=lambda a: novelty[s][a])

        next_pos = env.step(pos, action)
        s_next = state_to_idx(next_pos[0], next_pos[1], grid_size)

        # Dirichlet-smoothed likelihood for prediction error.
        alpha = 1e-3
        row = counts[s][action]
        total = sum(row)
        p_next = (row[s_next] + alpha) / (total + alpha * n_states)
        pe = -math.log(p_next)

        # Update novelty estimate and world model.
        novelty[s][action] = 0.7 * novelty[s][action] + 0.3 * pe
        counts[s][action][s_next] += 1

        pos = next_pos
        visited.add(pos)

        for c in corners:
            if pos == c and first_corner_hit[c] is None:
                first_corner_hit[c] = t + 1

        coverage.append(len(visited) / n_states)

    return {
        "coverage": coverage,
        "first_corner_hit": first_corner_hit,
        "final_coverage": coverage[-1],
    }


def summarize(results: list[dict], checkpoints=(20, 50, 100, 200, 250)) -> None:
    print("  coverage (mean ± std)")
    for k in checkpoints:
        vals = [r["coverage"][k - 1] for r in results]
        mean, std = mean_std(vals)
        print(f"    step {k:>3}: {mean:.3f} ± {std:.3f}")

    corner_times = {c: [] for c in results[0]["first_corner_hit"].keys()}
    for r in results:
        for c, t in r["first_corner_hit"].items():
            if t is not None:
                corner_times[c].append(float(t))

    print("  first corner hit times (mean over reached runs)")
    for c, times in corner_times.items():
        if not times:
            print(f"    corner {c}: never reached")
        else:
            print(f"    corner {c}: {statistics.mean(times):.1f} steps (reached {len(times)}/{len(results)})")


if __name__ == "__main__":
    seeds = list(range(16))

    print("=== Curiosity-driven explorer (intrinsic reward = prediction error) ===")
    curiosity_runs = [run_episode(seed=s, policy="curiosity") for s in seeds]
    summarize(curiosity_runs)

    print("\n=== Random explorer baseline ===")
    random_runs = [run_episode(seed=s, policy="random") for s in seeds]
    summarize(random_runs)

    curiosity_final = statistics.mean(r["final_coverage"] for r in curiosity_runs)
    random_final = statistics.mean(r["final_coverage"] for r in random_runs)

    print("\n=== Takeaway ===")
    print(f"  mean final coverage @250: curiosity={curiosity_final:.3f} vs random={random_final:.3f}")
    if curiosity_final > random_final:
        print("  Curiosity explores more of the map by chasing its own modeling failures.")
    else:
        print("  In this setup, curiosity was not better than random; tune model/curriculum next.")
