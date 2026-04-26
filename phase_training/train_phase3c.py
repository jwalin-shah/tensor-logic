"""
Phase 3c: MLP world model with the same SSM-style belief tensor as 3b.

Equal-param-count baseline. Same world (single-channel + occluder + collision),
same recurrence (R̂[t+1] = obs[t+1] + (1 - obs[t+1]) * prior), same loss,
same training schedule — only the forward model changes.

This is the experiment that decides whether TL is the active ingredient or
whether memory + capacity alone is enough. If MLP ties or beats TL, we
cannot honestly claim the einsum-factored transition tensor matters here.

Phase 3a (no memory):       P@2 = 87.27%, in-occluder recall = 67.55%
Phase 3b (TL + memory):     P@2 = 90.63%, in-occluder recall = 98.84%
"""

import time
import torch
import torch.nn.functional as F

from world import GridWorld, N_ACTIONS
from mlp_model import MLPForwardModel


OCCLUDER = (3, 5, 3, 5)
N_OBJECTS = 2
TRAJ_LEN = 20


def collect_episodes(world, n_traj, length):
    obs_list, true_list, a_list = [], [], []
    for _ in range(n_traj):
        obs, true, actions = world.sample_trajectory(length=length)
        obs_list.append(obs)
        true_list.append(true)
        a_list.append(actions)
    return (torch.stack(obs_list),
            torch.stack(true_list),
            torch.stack(a_list))


def rollout(model, obs_batch, action_batch, return_all=True):
    B, T1 = obs_batch.shape[:2]
    T = action_batch.shape[1]
    belief = obs_batch[:, 0]
    beliefs = [belief]
    priors = []
    for t in range(T):
        action_t = action_batch[:, t]
        prior = model(belief, action_t)
        priors.append(prior)
        obs_next = obs_batch[:, t + 1]
        belief = obs_next + (1.0 - obs_next) * prior
        beliefs.append(belief)
    if return_all:
        return torch.stack(beliefs, dim=1), torch.stack(priors, dim=1)
    return belief


def precision_at_k(pred, true, k):
    pred = pred.view(pred.shape[0], -1)
    true = true.view(true.shape[0], -1).bool()
    _, topk = pred.topk(k, dim=-1)
    return true.gather(1, topk).float().mean().item()


def in_occluder_recall(pred, true, occluder, k):
    B = pred.shape[0]
    xl, xh, yl, yh = occluder
    occ_mask = torch.zeros(true.shape[-2:], dtype=torch.bool)
    occ_mask[xl:xh + 1, yl:yh + 1] = True
    occ_flat = occ_mask.view(-1)
    true_flat = true.view(B, -1).bool()
    pred_flat = pred.view(B, -1)
    _, topk = pred_flat.topk(k, dim=-1)
    pred_topk = torch.zeros_like(pred_flat, dtype=torch.bool)
    pred_topk.scatter_(1, topk, True)
    has_hidden = (true_flat & occ_flat.unsqueeze(0)).any(dim=-1)
    pred_in_occ = (pred_topk & occ_flat.unsqueeze(0)).any(dim=-1)
    if has_hidden.sum() == 0:
        return None, 0
    return (pred_in_occ[has_hidden].float().mean().item(),
            int(has_hidden.sum()))


def per_step_metrics(beliefs, trues, occluder, k):
    T1 = beliefs.shape[1]
    p_per_step, perm_per_step = [], []
    for t in range(1, T1):
        pred_t = beliefs[:, t].squeeze(1)
        true_t = trues[:, t].squeeze(1)
        p_per_step.append(precision_at_k(pred_t, true_t, k))
        perm = in_occluder_recall(pred_t, true_t, occluder, k)
        perm_per_step.append(perm[0] if perm[0] is not None else float('nan'))
    return p_per_step, perm_per_step


def main():
    torch.manual_seed(0)
    print("=== Phase 3c: MLP world model with SSM-style belief tensor ===")
    print(f"  Occluder: rows {OCCLUDER[0]}-{OCCLUDER[1]}, cols {OCCLUDER[2]}-{OCCLUDER[3]}")
    print(f"  Trajectory length: {TRAJ_LEN}, n_objects: {N_OBJECTS}, collision: True")
    print(f"  Phase 3a (no memory):    P@2 = 87.27%, in-occ-recall = 67.55%")
    print(f"  Phase 3b (TL + memory):  P@2 = 90.63%, in-occ-recall = 98.84%\n")

    world_train = GridWorld(size=8, n_objects=N_OBJECTS, seed=0, collision=True,
                            single_channel=True, occluder_zone=OCCLUDER)
    world_val = GridWorld(size=8, n_objects=N_OBJECTS, seed=1, collision=True,
                          single_channel=True, occluder_zone=OCCLUDER)

    print("Collecting trajectories...")
    t0 = time.time()
    train_obs, train_true, train_a = collect_episodes(world_train, n_traj=2000, length=TRAJ_LEN)
    val_obs, val_true, val_a = collect_episodes(world_val, n_traj=400, length=TRAJ_LEN)
    print(f"  train: {len(train_obs)} eps, val: {len(val_obs)} eps ({time.time()-t0:.1f}s)")

    model = MLPForwardModel(grid=8, n_actions=N_ACTIONS, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining (params: {n_params:,}, vs TL: 16,384)")

    n_epochs = 30
    batch_size = 64

    for epoch in range(1, n_epochs + 1):
        idx = torch.randperm(len(train_obs))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_obs), batch_size):
            b = idx[start:start + batch_size]
            beliefs, _ = rollout(model, train_obs[b], train_a[b])
            loss = F.mse_loss(beliefs[:, 1:], train_true[b, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= n_batches

        if epoch % 3 == 0 or epoch == 1:
            with torch.no_grad():
                val_beliefs, _ = rollout(model, val_obs, val_a)
                val_loss = F.mse_loss(val_beliefs[:, 1:], val_true[:, 1:]).item()
                preds_all = val_beliefs[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
                trues_all = val_true[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
                p = precision_at_k(preds_all, trues_all, N_OBJECTS)
                perm, n_perm = in_occluder_recall(preds_all, trues_all, OCCLUDER, N_OBJECTS)
            print(f"  epoch {epoch:2d}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}  "
                  f"P@{N_OBJECTS}={p:.2%}  in-occ-recall={perm:.2%} (n={n_perm})")

    print("\n=== Final eval ===")
    with torch.no_grad():
        val_beliefs, val_priors = rollout(model, val_obs, val_a)
        preds_all = val_beliefs[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        trues_all = val_true[:, 1:].reshape(-1, 1, 8, 8).squeeze(1)
        p = precision_at_k(preds_all, trues_all, N_OBJECTS)
        perm, n_perm = in_occluder_recall(preds_all, trues_all, OCCLUDER, N_OBJECTS)

    print(f"  Precision@{N_OBJECTS}:    {p:.2%}   (3a: 87.27% | 3b: 90.63%)")
    print(f"  In-occluder recall:  {perm:.2%}   (3a: 67.55% | 3b: 98.84%)")
    print(f"  N hidden cases:      {n_perm}")

    print("\n=== Per-step accuracy across the trajectory ===")
    with torch.no_grad():
        p_per, perm_per = per_step_metrics(val_beliefs, val_true, OCCLUDER, N_OBJECTS)
    print(f"  {'t':>3s}  {'P@2':>7s}  {'in-occ':>8s}")
    for t, (pt, prt) in enumerate(zip(p_per, perm_per), start=1):
        if t in (1, 2, 3, 5, 10, 15, 20):
            occ_str = f"{prt:.2%}" if not (prt != prt) else "n/a"
            print(f"  {t:>3d}  {pt:>7.2%}  {occ_str:>8s}")


if __name__ == "__main__":
    main()
