"""
Object permanence via violation-of-expectation.

This demonstrates learning object permanence by training a forward model
on a simple 1D sequence where an object moves at a constant velocity,
disappears behind an occluder, and then reappears.

Once trained, we test the model on:
  1. A normal sequence (it reappears where it should).
  2. An impossible sequence (it reappears too early or vanishes).

We measure the model's surprise (prediction error).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Sequence Generation
# ============================================================
# World: positions 0 to 9.
# The occluder is at positions 4, 5, 6.
# If the object is at an occluder, its observed position is -1 (invisible).

OCCLUDER_START = 4
OCCLUDER_END = 6

def generate_sequence(start_pos=0, length=10, impossible_reappearance=None, vanish=False):
    """
    Generates a sequence of positions.
    impossible_reappearance: if set to an index, the object suddenly appears there instead of the correct spot.
    vanish: if True, the object never reappears after the occluder.
    """
    seq = []
    actual_pos = start_pos
    for t in range(length):
        if vanish and actual_pos > OCCLUDER_END:
            obs = -1.0 # Stays invisible
        elif impossible_reappearance is not None and t == (OCCLUDER_END + 1):
            actual_pos = impossible_reappearance
            obs = float(actual_pos)
        else:
            if OCCLUDER_START <= actual_pos <= OCCLUDER_END:
                obs = -1.0
            else:
                obs = float(actual_pos)

        seq.append(obs)
        actual_pos += 1  # Constant velocity of +1

    return torch.tensor(seq).unsqueeze(-1)


# ============================================================
# 2. Forward Model (RNN)
# ============================================================
class RNNForwardModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x is [batch, seq_len, 1]
        out, _ = self.rnn(x)
        pred = self.fc(out)
        return pred


# ============================================================
# 3. Training Loop
# ============================================================
def train_permanence():
    print("=== Training Forward Model on Normal Physics ===")
    model = RNNForwardModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # Generate standard training sequences
    # We train the model to predict the NEXT observation given the history.
    # Vary the start position slightly to create a diverse dataset.
    train_seqs = []
    for i in range(100):
        # start_pos ranges from -2 to 2 to generalize
        start_p = (i % 5) - 2
        train_seqs.append(generate_sequence(start_pos=start_p, length=10))

    X = torch.stack([seq[:-1] for seq in train_seqs]) # [100, 9, 1]
    Y = torch.stack([seq[1:] for seq in train_seqs])  # [100, 9, 1]

    for epoch in range(500):
        pred = model(X)
        loss = F.mse_loss(pred, Y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d} | MSE Loss: {loss.item():.4f}")

    return model


# ============================================================
# 4. Evaluation (Violation of Expectation)
# ============================================================
def evaluate(model):
    print("\n=== Violation of Expectation Test ===")

    # 1. Normal Sequence
    # 0, 1, 2, 3, -1, -1, -1, 7, 8, 9
    normal_seq = generate_sequence(start_pos=0, length=10)

    # 2. Impossible Sequence (reappears instantly at 8 right after 6)
    # 0, 1, 2, 3, -1, -1, -1, 8, 9, 10
    impossible_seq = generate_sequence(start_pos=0, length=10, impossible_reappearance=8)

    # 3. Vanish Sequence (never reappears)
    # 0, 1, 2, 3, -1, -1, -1, -1, -1, -1
    vanish_seq = generate_sequence(start_pos=0, length=10, vanish=True)

    model.eval()
    with torch.no_grad():
        def get_surprise(seq, name):
            x = seq[:-1].unsqueeze(0) # [1, 9, 1]
            y = seq[1:].unsqueeze(0)  # [1, 9, 1]
            pred = model(x)

            # We care about the surprise at the moment of reappearance (t=7, which is index 6 in Y)
            # Y indices: 0->t=1, 1->t=2, 2->t=3, 3->t=4(-1), 4->t=5(-1), 5->t=6(-1), 6->t=7(reappear)
            reappear_idx = 6

            pred_val = pred[0, reappear_idx, 0].item()
            actual_val = y[0, reappear_idx, 0].item()
            surprise = (pred_val - actual_val) ** 2

            print(f"\n{name} Sequence:")
            print(f"  Expected (Predicted) at t=7: {pred_val:.2f}")
            print(f"  Actual Observation at t=7: {actual_val:.2f}")
            print(f"  Surprise (Error^2): {surprise:.4f}")

        get_surprise(normal_seq, "Normal")
        get_surprise(impossible_seq, "Impossible (Teleport)")
        get_surprise(vanish_seq, "Vanish")

if __name__ == "__main__":
    torch.manual_seed(42)
    model = train_permanence()
    evaluate(model)
