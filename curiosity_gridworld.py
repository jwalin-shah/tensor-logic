"""
Curiosity-driven exploration in a tiny gridworld.

Intrinsic reward is the agent's own one-step prediction error:
    r_int(s, a) = CE( f_theta(s, a), s_next )

The agent prefers actions with higher expected prediction error, then updates
its world model online. We compare this against a random explorer baseline.

Goal of this demo: show what gets explored first when curiosity = surprise.
"""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def state_to_idx(x: int, y: int, size: int) -> int:
    return x * size + y


class ForwardModel(nn.Module):
    """Predict next grid cell from (state, action)."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_states),
        )

    def forward(self, state_oh: torch.Tensor, action_oh: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state_oh, action_oh], dim=-1))


@dataclass
class GridWorld:
    size: int = 7

    def step(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dx, dy = ACTIONS[action]
        nx = min(self.size - 1, max(0, pos[0] + dx))
        ny = min(self.size - 1, max(0, pos[1] + dy))
        return nx, ny


def one_hot(index: int, n: int) -> torch.Tensor:
    t = torch.zeros(1, n)
    t[0, index] = 1.0
    return t


def run_episode(seed: int, policy: str, steps: int = 250, grid_size: int = 7):
    assert policy in {"curiosity", "random"}
    random.seed(seed)
    torch.manual_seed(seed)

    env = GridWorld(size=grid_size)
    n_states = grid_size * grid_size
    n_actions = len(ACTIONS)

    model = ForwardModel(n_states, n_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Estimated novelty per (state, action): initialize high for optimism.
    novelty = torch.full((n_states, n_actions), 2.0)

    pos = (grid_size // 2, grid_size // 2)
    visited = {pos}
    coverage = []

    corners = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]
    first_corner_hit = {c: None for c in corners}

    for t in range(steps):
        s = state_to_idx(pos[0], pos[1], grid_size)

        if policy == "random":
            action = random.randrange(n_actions)
        else:
            if random.random() < 0.1:
                action = random.randrange(n_actions)
            else:
                action = int(torch.argmax(novelty[s]).item())

        next_pos = env.step(pos, action)
        s_next = state_to_idx(next_pos[0], next_pos[1], grid_size)

        s_oh = one_hot(s, n_states)
        a_oh = one_hot(action, n_actions)
        target = torch.tensor([s_next])

        logits = model(s_oh, a_oh)
        loss = F.cross_entropy(logits, target)

        # Curiosity signal = own prediction error before learning from this step.
        pe = float(loss.detach().item())
        novelty[s, action] = 0.9 * novelty[s, action] + 0.1 * pe

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def summarize(results: list[dict], checkpoints=(20, 50, 100, 200, 250)):
    print("  coverage (mean ± std)")
    for k in checkpoints:
        vals = torch.tensor([r["coverage"][k - 1] for r in results])
        print(f"    step {k:>3}: {vals.mean():.3f} ± {vals.std(unbiased=False):.3f}")

    corner_times = {c: [] for c in results[0]["first_corner_hit"].keys()}
    for r in results:
        for c, t in r["first_corner_hit"].items():
            corner_times[c].append(float(t) if t is not None else float("nan"))

    print("  first corner hit times (mean over reached runs)")
    for c, times in corner_times.items():
        ts = torch.tensor(times)
        reached = ts[~torch.isnan(ts)]
        if len(reached) == 0:
            print(f"    corner {c}: never reached")
        else:
            print(f"    corner {c}: {reached.mean():.1f} steps (reached {len(reached)}/{len(times)})")


if __name__ == "__main__":
    seeds = list(range(8))

    print("=== Curiosity-driven explorer (intrinsic reward = prediction error) ===")
    curiosity_runs = [run_episode(seed=s, policy="curiosity") for s in seeds]
    summarize(curiosity_runs)

    print("\n=== Random explorer baseline ===")
    random_runs = [run_episode(seed=s, policy="random") for s in seeds]
    summarize(random_runs)

    curiosity_final = torch.tensor([r["final_coverage"] for r in curiosity_runs]).mean().item()
    random_final = torch.tensor([r["final_coverage"] for r in random_runs]).mean().item()

    print("\n=== Takeaway ===")
    print(f"  mean final coverage @250: curiosity={curiosity_final:.3f} vs random={random_final:.3f}")
    if curiosity_final > random_final:
        print("  Curiosity explores more of the map by chasing its own modeling failures.")
    else:
        print("  In this setup, curiosity was not better than random; tune model/curriculum next.")
