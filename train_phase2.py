"""
Phase 2: harder test for the TL forward model.

Two changes vs phase 1:
  1. Collision: objects can't share cells. If an action would put them
     on the same square, nobody moves. This breaks per-object factorization
     — the next state of object 0 now depends on where object 1 is.
  2. Multi-step rollout eval. We feed the model's own argmax-snapped
     predictions back as input, k steps deep. This tests whether per-step
     errors compound into rollout drift.

Same model, same training loop, same params (16,384). The win condition
is now nuanced: per-step accuracy will drop because the model's W tensor
has no way to represent "depends on the other object," and rollout will
amplify whatever per-step error exists.
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


def per_object_accuracy(pred, true):
    B, O, X, Y = pred.shape
    pred_idx = pred.view(B, O, X * Y).argmax(dim=-1)
    true_idx = true.view(B, O, X * Y).argmax(dim=-1)
    return (pred_idx == true_idx).float().mean().item()


def snap_to_argmax(pred):
    """Turn soft occupancy into one-hot per object via argmax."""
    B, O, X, Y = pred.shape
    flat = pred.view(B, O, X * Y)
    argmax = flat.argmax(dim=-1)
    snapped = torch.zeros_like(flat)
    snapped.scatter_(2, argmax.unsqueeze(-1), 1.0)
    return snapped.view(B, O, X, Y)


def multi_step_eval(model, world, n_episodes, ks):
    """Roll out model k steps, snapping to argmax between steps; report acc at each k."""
    max_k = max(ks)
    all_states, all_actions = [], []
    for _ in range(n_episodes):
        states, actions = world.sample_trajectory(length=max_k)
        all_states.append(states)
        all_actions.append(actions)
    all_states = torch.stack(all_states)         # (B, max_k+1, n_obj, x, y)
    all_actions = torch.stack(all_actions)        # (B, max_k)

    results = {}
    with torch.no_grad():
        pred_state = all_states[:, 0].clone()
        for t in range(max_k):
            pred = model(pred_state, all_actions[:, t])
            pred_state = snap_to_argmax(pred)
            k = t + 1
            if k in ks:
                results[k] = per_object_accuracy(pred_state, all_states[:, k])
    return results


def main():
    torch.manual_seed(0)
    print("=== Phase 2: collision dynamics + multi-step rollout eval ===")
    world_train = GridWorld(size=8, n_objects=2, seed=0, collision=True)
    world_val = GridWorld(size=8, n_objects=2, seed=1, collision=True)
    world_eval = GridWorld(size=8, n_objects=2, seed=2, collision=True)

    print("\nCollecting rollouts (train + val)...")
    t0 = time.time()
    train_s, train_a, train_sn = collect_rollouts(world_train, n_traj=2000, length=20)
    val_s, val_a, val_sn = collect_rollouts(world_val, n_traj=200, length=20)
    print(f"  train: {len(train_s)} triples, val: {len(val_s)} triples "
          f"({time.time()-t0:.1f}s)")

    # Quick stat: how often does collision actually happen?
    collisions = (train_s == train_sn).all(dim=(1, 2, 3)).float().mean().item()
    print(f"  fraction of steps where state didn't change (proxy for collisions): {collisions:.2%}")

    model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nTraining (params: {n_params:,})")

    n_train = len(train_s)
    batch_size = 256
    n_epochs = 30

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
            val_acc_1step = per_object_accuracy(val_pred, val_sn)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  train_loss={epoch_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  val_acc(k=1)={val_acc_1step:.2%}")

    print("\n=== Multi-step rollout eval ===")
    ks = [1, 2, 5, 10]
    rollout_acc = multi_step_eval(model, world_eval, n_episodes=500, ks=ks)
    print(f"  {'k':>3s}  {'per-object accuracy':>22s}")
    for k in ks:
        print(f"  {k:>3d}  {rollout_acc[k]:>21.2%}")

    print("\nWhat to look for:")
    print("  - val_acc(k=1) below 100% means the model can't perfectly fit collision dynamics")
    print("    (its W tensor factors per-object; collisions need joint state).")
    print("  - rollout accuracy decay tells us how per-step error compounds over k steps.")


if __name__ == "__main__":
    main()
