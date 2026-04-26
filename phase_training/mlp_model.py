"""
MLP forward model — same input/output signature as TLForwardModel.

Takes (state, action) → predicted next state. Concatenates flattened state
with one-hot action, runs through a 1-hidden-layer MLP, sigmoids the
output, reshapes back to (B, 1, 8, 8).

Hidden=128 picked to give ~17K params (vs TL's 4*8^4 = 16,384) — slight
capacity edge for MLP, so any TL win can't be blamed on parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPForwardModel(nn.Module):
    def __init__(self, grid=8, n_actions=4, hidden=128, init_bias=-2.0):
        super().__init__()
        self.grid = grid
        self.n_actions = n_actions
        in_dim = grid * grid + n_actions
        out_dim = grid * grid
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, state, action):
        B = state.shape[0]
        flat = state.view(B, -1)
        a_onehot = F.one_hot(action, self.n_actions).float()
        x = torch.cat([flat, a_onehot], dim=-1)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return torch.sigmoid(out).view(B, 1, self.grid, self.grid)
