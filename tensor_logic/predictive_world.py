"""Small JEPA-style predictive world representation for experiment #101.

This is intentionally a minimal encoder-predictor scaffold, not a reproduction
of Meta's JEPA implementations. It predicts a target latent representation from
current typed-state features and an action/context vector, while the target
representation is stop-gradient.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PredictiveWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, latent_dim, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch,{self.state_dim}]"
            )
        return self.encoder(state)

    def predict_latent(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(
                f"action must have shape [batch,{self.action_dim}]"
            )
        if action.shape[0] != state.shape[0]:
            raise ValueError("state/action batch sizes differ")
        current = self.encode(state)
        return self.predictor(torch.cat((current, action), dim=-1))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        target_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self.predict_latent(state, action)
        with torch.no_grad():
            target = self.encode(target_state).detach()
        return predicted, target


def jepa_cosine_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted/target latent shapes differ")
    predicted_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return (1.0 - (predicted_norm * target_norm).sum(dim=-1)).mean()


def latent_transition_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-example latent distance for planning/evaluation."""
    if predicted.shape != target.shape:
        raise ValueError("predicted/target latent shapes differ")
    return torch.linalg.vector_norm(predicted - target, dim=-1)
