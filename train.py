"""
Phase 1: train the TL forward model on random rollouts in the 8x8 gridworld.

Loop:
  1. Generate N rollouts of random actions.
  2. For each (state, action, next_state) triple, predict next_state
     from (state, action) and minimize MSE.
  3. Eval held-out trajectories: how often does argmax of predicted
     occupancy match argmax of true next_state, per object?

Win condition: held-out argmax-accuracy >= 80%.
"""

import time
import torch
import torch.nn.functional as F

from world import GridWorld, N_ACTIONS
from tl_model import TLForwardModel


def collect_rollouts(world, n_traj, length):
    s_list, a_list, sn_list = [], [], []
    for _ in range(n_traj):
        states, actions = world.sample_trajectory(length=length)
        s_list.append(states[:-1])
        a_list.append(actions)
        sn_list.append(states[1:])
    return (torch.cat(s_list, dim=0),
            torch.cat(a_list, dim=0),
            torch.cat(sn_list, dim=0))


def argmax_accuracy(pred, true):
    """For each object in each example, does argmax of pred match argmax of true?"""
    B, O, X, Y = pred.shape
    pred_idx = pred.view(B, O, X * Y).argmax(dim=-1)
    true_idx = true.view(B, O, X * Y).argmax(dim=-1)
    return (pred_idx == true_idx).float().mean().item()


def main():
    torch.manual_seed(0)
    world_train = GridWorld(size=8, n_objects=2, seed=0)
    world_val = GridWorld(size=8, n_objects=2, seed=1)

    print("Collecting rollouts (train + val)...")
    t0 = time.time()
    train_s, train_a, train_sn = collect_rollouts(world_train, n_traj=1000, length=20)
    val_s, val_a, val_sn = collect_rollouts(world_val, n_traj=200, length=20)
    print(f"  train: {len(train_s)} triples, val: {len(val_s)} triples "
          f"({time.time()-t0:.1f}s)")

    model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nTraining (params: {n_params:,})")
    print("  win condition: val_acc >= 80%")

    n_train = len(train_s)
    batch_size = 256
    n_epochs = 20

    for epoch in range(1, n_epochs + 1):
        idx = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            b = idx[start:start + batch_size]
            pred = model(train_s[b], train_a[b])
            loss = F.mse_loss(pred, train_sn[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        epoch_loss /= n_train

        with torch.no_grad():
            val_pred = model(val_s, val_a)
            val_loss = F.mse_loss(val_pred, val_sn).item()
            val_acc = argmax_accuracy(val_pred, val_sn)

        print(f"  epoch {epoch:2d}  train_loss={epoch_loss:.5f}  "
              f"val_loss={val_loss:.5f}  val_acc={val_acc:.2%}")

    print(f"\nFinal val_acc = {val_acc:.2%}")
    print(f"Win condition (>= 80%): "
          f"{'PASS' if val_acc >= 0.80 else 'FAIL'}")


if __name__ == "__main__":
    main()
