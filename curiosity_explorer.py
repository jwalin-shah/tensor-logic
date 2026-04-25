"""
Curiosity-driven explorer: intrinsic reward via prediction error.

An agent in a simple 1D grid world. The agent has a forward predictive model:
  state + action -> next_state

The intrinsic reward is the prediction error of the forward model. The agent
will naturally explore states where its model is poor, and move on once
it has learned the dynamics of that state.

Setup:
  - World: 1D grid with 10 positions (0..9).
  - Actions: -1 (left), +1 (right).
  - Model: PyTorch MLP predicting the next state from the current state and action.
  - Policy: "Babbling" (random actions initially), but we will observe the states
    it prefers to visit based on its model error. We'll simulate a simple
    exploration strategy: pick the action that maximizes predicted error
    (curiosity) or randomly babble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ============================================================
# 1. World and Agent Setup
# ============================================================
WORLD_SIZE = 10

class World:
    def __init__(self, size=WORLD_SIZE):
        self.size = size
        self.state = size // 2  # Start in the middle

    def step(self, action):
        # Action is -1 (left) or 1 (right)
        self.state = max(0, min(self.size - 1, self.state + action))
        return self.state

    def reset(self):
        self.state = self.size // 2
        return self.state


class ForwardModel(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ============================================================
# 2. Training Loop with Curiosity
# ============================================================
def train_curiosity_explorer(steps=2000):
    world = World()
    model = ForwardModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    state = world.reset()
    visitation_counts = [0] * WORLD_SIZE

    print("=== Training Curiosity-Driven Explorer ===")

    for step in range(steps):
        visitation_counts[state] += 1

        state_tensor = torch.tensor([[float(state)]])

        # We track empirical prediction errors per (state, action) pair to guide curiosity
        if not hasattr(train_curiosity_explorer, "empirical_errors"):
            train_curiosity_explorer.empirical_errors = {}

        # Initialize errors for unseen pairs to a high value to encourage initial exploration
        for a in [-1, 1]:
            if (state, a) not in train_curiosity_explorer.empirical_errors:
                train_curiosity_explorer.empirical_errors[(state, a)] = 100.0

        if random.random() < 0.3:
            # Random babbling 30% of the time
            action = random.choice([-1, 1])
        else:
            # Curious action 70% of the time: pick action with highest moving-average empirical error
            err_left = train_curiosity_explorer.empirical_errors[(state, -1)]
            err_right = train_curiosity_explorer.empirical_errors[(state, 1)]
            action = -1 if err_left > err_right else 1

        next_state = world.step(action)

        # Train the model
        action_tensor = torch.tensor([[float(action)]])
        target_tensor = torch.tensor([[float(next_state)]])

        pred = model(state_tensor, action_tensor)
        loss = F.mse_loss(pred, target_tensor)

        # Update empirical error estimate (moving average)
        current_error = abs(pred.item() - next_state)
        prev_error = train_curiosity_explorer.empirical_errors.get((state, action), current_error)
        train_curiosity_explorer.empirical_errors[(state, action)] = 0.9 * prev_error + 0.1 * current_error

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(f"Step {step:4d} | Model Loss (Surprise): {loss.item():.4f} | State: {state}")

        state = next_state

    print("\n=== Final Visitation Frequency ===")
    for s, count in enumerate(visitation_counts):
        print(f"State {s}: {'#' * (count // 10)} ({count})")

    print("\n=== Model Evaluation (Surprise Landscape) ===")
    print("Testing the model's prediction error across all states and actions.")
    for s in range(WORLD_SIZE):
        s_tensor = torch.tensor([[float(s)]])

        # Test action right
        a_right = torch.tensor([[1.0]])
        actual_right = max(0, min(WORLD_SIZE - 1, s + 1))
        pred_r = model(s_tensor, a_right).item()
        err_r = abs(pred_r - actual_right)

        # Test action left
        a_left = torch.tensor([[-1.0]])
        actual_left = max(0, min(WORLD_SIZE - 1, s - 1))
        pred_l = model(s_tensor, a_left).item()
        err_l = abs(pred_l - actual_left)

        avg_err = (err_r + err_l) / 2
        print(f"State {s}: Avg Error = {avg_err:.4f} (Pred L: {pred_l:.1f}, Pred R: {pred_r:.1f})")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    train_curiosity_explorer()
