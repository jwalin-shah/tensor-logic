"""
8x8 gridworld with N objects and 4 actions (N/S/E/W).

Each action shifts ALL objects by 1 cell in the chosen direction.
If the shift would go off-grid, the object clamps to the boundary.

State is a tensor R[obj, x, y] of shape (n_objects, size, size),
one-hot per object (1.0 at the cell the object occupies, 0.0 elsewhere).

Phase 1: simplest possible setup for testing whether a tensor-logic
forward model can learn deterministic dynamics from random rollouts.
"""

import random
import torch

# (dx, dy) for each action. 0=N, 1=S, 2=E, 3=W.
ACTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]
N_ACTIONS = 4


class GridWorld:
    def __init__(self, size=8, n_objects=2, seed=None, collision=False,
                 single_channel=False, occluder_zone=None):
        self.size = size
        self.n_objects = n_objects
        self.collision = collision
        self.single_channel = single_channel
        self.occluder_zone = occluder_zone  # (x_lo, x_hi, y_lo, y_hi) inclusive, or None
        self.rng = random.Random(seed)

    def reset(self):
        positions = []
        used = set()
        while len(positions) < self.n_objects:
            x = self.rng.randint(0, self.size - 1)
            y = self.rng.randint(0, self.size - 1)
            if (x, y) not in used:
                positions.append((x, y))
                used.add((x, y))
        return positions

    def step(self, positions, action):
        dx, dy = ACTIONS[action]
        intended = []
        for (x, y) in positions:
            nx = max(0, min(self.size - 1, x + dx))
            ny = max(0, min(self.size - 1, y + dy))
            intended.append((nx, ny))
        if self.collision and len(set(intended)) < len(intended):
            return positions
        return intended

    def to_tensor(self, positions):
        """Full true state with per-object channels. Shape (n_objects, size, size)."""
        state = torch.zeros(self.n_objects, self.size, self.size)
        for o, (x, y) in enumerate(positions):
            state[o, x, y] = 1.0
        return state

    def to_true_single(self, positions):
        """Single-channel true occupancy (no identity, no occlusion). Shape (1, size, size)."""
        state = torch.zeros(1, self.size, self.size)
        for (x, y) in positions:
            state[0, x, y] = 1.0
        return state

    def to_observed_single(self, positions):
        """Single-channel observed state (no identity; occluder applied). Shape (1, size, size)."""
        state = self.to_true_single(positions)
        if self.occluder_zone is not None:
            xl, xh, yl, yh = self.occluder_zone
            state[0, xl:xh + 1, yl:yh + 1] = 0.0
        return state

    def sample_trajectory(self, length=20):
        """Returns either:
          - (states, actions) where states is (T+1, n_obj, x, y), if single_channel=False.
          - (obs_states, true_states, actions) all single-channel, if single_channel=True.
            obs has occluder applied; true does not.
        """
        positions = [self.reset()]
        actions = []
        for _ in range(length):
            a = self.rng.randint(0, N_ACTIONS - 1)
            positions.append(self.step(positions[-1], a))
            actions.append(a)
        if self.single_channel:
            obs = torch.stack([self.to_observed_single(p) for p in positions])
            true = torch.stack([self.to_true_single(p) for p in positions])
            return obs, true, torch.tensor(actions, dtype=torch.long)
        states = torch.stack([self.to_tensor(p) for p in positions])
        return states, torch.tensor(actions, dtype=torch.long)
