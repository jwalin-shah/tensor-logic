"""
Tensor-logic forward model for the 2D gridworld.

The forward step is a single einsum + nonlinearity:
    predicted[obj, p, q] = sigmoid(einsum('oxy, xypq -> opq',
                                           state, W[action]))

W has shape (n_actions, x, y, p, q). For each (action, x, y) the slice
W[action, x, y, :, :] should activate the cell where an object at (x, y)
will be after the action. The model has to learn this from data.

Same primitive as transitive_closure.py: einsum forward step + nonlinearity.
The action index just selects which transition tensor to apply.
"""

import torch
import torch.nn as nn


class TLForwardModel(nn.Module):
    def __init__(self, grid=8, n_actions=4, init_bias=-2.0):
        super().__init__()
        # W[action, x, y, x_next, y_next]
        self.W = nn.Parameter(
            torch.randn(n_actions, grid, grid, grid, grid) * 0.05 + init_bias
        )

    def forward(self, state, action):
        """
        state:  (..., n_obj, x, y) — one-hot occupancy per object
        action: int or tensor of action indices, shape (...,)
        returns: predicted next state, same shape as `state`
        """
        W_a = self.W[action]  # (..., x, y, p, q)
        scores = torch.einsum('...oxy,...xypq->...opq', state, W_a)
        return torch.sigmoid(scores)
