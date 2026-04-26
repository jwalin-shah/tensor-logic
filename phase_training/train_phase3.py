"""
Phase 3a: strip identity + add occlusion, NO memory yet.

This is the deliberately-broken baseline. The model now sees:
  - A single-channel occupancy map (no per-object identity)
  - With cells in an occluder zone always reported as 0

It must predict the TRUE (unoccluded) next-state occupancy from this
impoverished observation. Without memory, when an object is hidden,
the model has no way to know it's there. We expect this to fail in a
specific, informative way: predictions for in-occluder cells will be
wrong roughly proportional to how often objects pass through the zone.

The point is to get a baseline number that memory will need to recover.
"""

import time
import torch
import torch.nn.functional as F

from world import GridWorld, N_ACTIONS
from tl_model import TLForwardModel


OCCLUDER = (3, 5, 3, 5)  # 3x3 occluder zone covering the center
N_OBJECTS = 2


def collect_rollouts(world, n_traj, length):
    obs_list, true_list, a_list = [], [], []
    for _ in range(n_traj):
        obs, true, actions = world.sample_trajectory(length=length)
        obs_list.append(obs[:-1])
        true_list.append(true[1:])
        a_list.append(actions)
    return (torch.cat(obs_list, dim=0),
            torch.cat(a_list, dim=0),
            torch.cat(true_list, dim=0))


def precision_at_k(pred, true, k):
    """Of the k cells the model is most confident about, what fraction are actually occupied?"""
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    true_flat = true.view(B, -1).bool()
    _, topk_idx = pred_flat.topk(k, dim=-1)
    correct = true_flat.gather(1, topk_idx).float()
    return correct.mean().item()


def recall_at_k(pred, true, k):
    """Of the actually-occupied cells, what fraction made it into the top-k?"""
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    true_flat = true.view(B, -1).bool()
    _, topk_idx = pred_flat.topk(k, dim=-1)
    pred_topk = torch.zeros_like(pred_flat, dtype=torch.bool)
    pred_topk.scatter_(1, topk_idx, True)
    intersect = (pred_topk & true_flat).float().sum(dim=-1)
    actual = true_flat.float().sum(dim=-1).clamp(min=1.0)
    return (intersect / actual).mean().item()


def in_occluder_recall(pred, true, occluder, k):
    """Among trajectories where the TRUE next state has at least one object inside
    the occluder zone, how often does the model put one of its top-k predictions
    inside the occluder zone too? This is the object-permanence-ish metric."""
    B = pred.shape[0]
    xl, xh, yl, yh = occluder
    occ_mask = torch.zeros(true.shape[-2:], dtype=torch.bool)
    occ_mask[xl:xh + 1, yl:yh + 1] = True
    occ_flat = occ_mask.view(-1)

    true_flat = true.view(B, -1).bool()
    pred_flat = pred.view(B, -1)
    _, topk_idx = pred_flat.topk(k, dim=-1)
    pred_topk = torch.zeros_like(pred_flat, dtype=torch.bool)
    pred_topk.scatter_(1, topk_idx, True)

    has_hidden_truth = (true_flat & occ_flat.unsqueeze(0)).any(dim=-1)
    pred_in_occ = (pred_topk & occ_flat.unsqueeze(0)).any(dim=-1)
    if has_hidden_truth.sum() == 0:
        return None, 0
    return (pred_in_occ[has_hidden_truth].float().mean().item(),
            int(has_hidden_truth.sum()))


def main():
    torch.manual_seed(0)
    print("=== Phase 3a: single-channel + occlusion, NO memory ===")
    print(f"  Occluder zone: rows {OCCLUDER[0]}-{OCCLUDER[1]}, "
          f"cols {OCCLUDER[2]}-{OCCLUDER[3]} ({(OCCLUDER[1]-OCCLUDER[0]+1)*(OCCLUDER[3]-OCCLUDER[2]+1)}/64 cells)")

    world_train = GridWorld(size=8, n_objects=N_OBJECTS, seed=0, collision=True,
                            single_channel=True, occluder_zone=OCCLUDER)
    world_val = GridWorld(size=8, n_objects=N_OBJECTS, seed=1, collision=True,
                          single_channel=True, occluder_zone=OCCLUDER)

    print("\nCollecting rollouts...")
    t0 = time.time()
    train_obs, train_a, train_true = collect_rollouts(world_train, n_traj=2000, length=20)
    val_obs, val_a, val_true = collect_rollouts(world_val, n_traj=500, length=20)
    print(f"  train: {len(train_obs)} triples, val: {len(val_obs)} triples ({time.time()-t0:.1f}s)")

    # Stat: how often is the *true* next state entirely inside the visible region vs has hidden objects?
    xl, xh, yl, yh = OCCLUDER
    occ_mask = torch.zeros(8, 8, dtype=torch.bool)
    occ_mask[xl:xh + 1, yl:yh + 1] = True
    has_hidden = (train_true.squeeze(1).bool() & occ_mask).any(dim=(-1, -2))
    print(f"  fraction of true next-states with at least one hidden object: {has_hidden.float().mean():.2%}")

    # n_obj=1 channel; the existing tl_model handles this directly
    model = TLForwardModel(grid=8, n_actions=N_ACTIONS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    print(f"\nTraining (params: {sum(p.numel() for p in model.parameters()):,})")

    n_train = len(train_obs)
    batch_size = 256
    n_epochs = 30

    for epoch in range(1, n_epochs + 1):
        idx = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            b = idx[start:start + batch_size]
            pred = model(train_obs[b], train_a[b])
            loss = F.mse_loss(pred, train_true[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        epoch_loss /= n_train

        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                val_pred = model(val_obs, val_a)
                val_loss = F.mse_loss(val_pred, val_true).item()
                p_at_k = precision_at_k(val_pred.squeeze(1), val_true.squeeze(1), N_OBJECTS)
                r_at_k = recall_at_k(val_pred.squeeze(1), val_true.squeeze(1), N_OBJECTS)
            print(f"  epoch {epoch:2d}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}  "
                  f"P@{N_OBJECTS}={p_at_k:.2%}  R@{N_OBJECTS}={r_at_k:.2%}")

    print("\n=== Final eval ===")
    with torch.no_grad():
        val_pred = model(val_obs, val_a)
    p = precision_at_k(val_pred.squeeze(1), val_true.squeeze(1), N_OBJECTS)
    r = recall_at_k(val_pred.squeeze(1), val_true.squeeze(1), N_OBJECTS)
    print(f"  Precision@{N_OBJECTS}: {p:.2%}")
    print(f"  Recall@{N_OBJECTS}:    {r:.2%}")

    # Object permanence metric
    perm_recall, n_perm = in_occluder_recall(
        val_pred.squeeze(1), val_true.squeeze(1), OCCLUDER, N_OBJECTS)
    print(f"\nObject permanence test (held-out trajectories where truth has a hidden object):")
    print(f"  N such cases: {n_perm}")
    print(f"  Of these, fraction where model put a top-{N_OBJECTS} prediction "
          f"inside the occluder: {perm_recall:.2%}" if perm_recall is not None else "  (none in this batch)")

    print("\nWhat this number means:")
    print(f"  - If model has truly learned object permanence, it should put predictions inside the occluder")
    print(f"    when objects are there. Without memory, it can only do this when an object is *about to enter*")
    print(f"    (i.e., visible last frame, predicted to disappear into occluder this frame).")
    print(f"  - Multi-step hidden trajectories should fail.")


if __name__ == "__main__":
    main()
