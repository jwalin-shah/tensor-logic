"""
Throwing physics: an embodied learner discovering "force → distance".

This is the simplest possible demonstration of how a concept like "force"
emerges from compression of action-outcome pairs. The agent has:
  - One innate ability: try a throw at force f, observe where it lands
  - One innate drive: predict outcomes, minimize error
  - No prior knowledge of physics

After ~100 random throws, it has cached enough (force, distance) pairs to
fit a forward model. Then it can do "thinking" = run the forward model
in its head to plan: "to hit target at distance D, what force f?"

The forward model is one tensor equation:
    landing_distance = f(throw_force)

The plan is the inverse:
    target_distance → throw_force  (via gradient descent through the model)

This is closed-loop sensorimotor learning in ~150 lines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. The "world" — projectile motion with a known closed form
#    (We KNOW the answer is d = v² · sin(2θ) / g, but the agent doesn't.)
# ============================================================
GRAVITY = 9.8
THROW_ANGLE = torch.pi / 4  # fixed 45 degree throw for simplicity


def world_throw(force: float) -> float:
    """The world. Takes a force, returns where the ball lands."""
    # We're treating "force" as initial velocity for simplicity
    v = force
    distance = (v ** 2) * torch.sin(2 * torch.tensor(THROW_ANGLE)) / GRAVITY
    return float(distance)


# ============================================================
# 2. The agent's forward model — a tiny MLP it trains on its own data
# ============================================================
class ForwardModel(nn.Module):
    """Predicts landing distance from throw force. The agent's 'physics'."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, force):
        return self.net(force).squeeze(-1)


def run_experiment(num_trials):
    print(f"\n======================================")
    print(f"Experiment: num_trials={num_trials}")
    print(f"======================================")

    # ============================================================
    # 3. Phase 1 — motor babbling: random throws, build dataset
    # ============================================================
    print("=== PHASE 1: Motor babbling (random throws) ===")
    torch.manual_seed(0)

    dataset = []
    for trial in range(num_trials):
        f = float(torch.rand(1)) * 30  # forces from 0 to 30
        d = world_throw(f)
        dataset.append((f, d))

    forces = torch.tensor([f for f, _ in dataset]).unsqueeze(1)
    distances = torch.tensor([d for _, d in dataset])
    print(f"  Collected {len(dataset)} (force, distance) pairs")
    print(f"  Range: f ∈ [{forces.min():.1f}, {forces.max():.1f}]   "
          f"d ∈ [{distances.min():.1f}, {distances.max():.1f}]")


    # ============================================================
    # 4. Phase 2 — train the forward model on self-collected data
    # ============================================================
    print("\n=== PHASE 2: Fit forward model (learn 'force → distance') ===")
    model = ForwardModel()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    for step in range(800):
        pred = model(forces)
        loss = F.mse_loss(pred, distances)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:3d}  prediction MSE = {loss.item():.3f}")

    print(f"  Final MSE: {loss.item():.4f}")


    # ============================================================
    # 5. Phase 3 — planning: 'how hard should I throw to hit target d?'
    #    The agent does NOT know the formula. It uses its learned model.
    #    Plan = gradient descent on force to minimize predicted-distance error.
    # ============================================================
    print("\n=== PHASE 3: Planning (use learned model to hit targets) ===")

    def plan_throw(target_d, model, n_iters=200):
        """Inverse the forward model via gradient descent on the action."""
        f = torch.tensor([10.0], requires_grad=True)
        inner_opt = torch.optim.Adam([f], lr=0.5)
        for _ in range(n_iters):
            pred = model(f.unsqueeze(0))
            err = (pred - target_d) ** 2
            inner_opt.zero_grad()
            err.backward()
            inner_opt.step()
            with torch.no_grad():
                f.clamp_(0, 50)  # physical bounds
        return float(f.detach())


    # Test on targets the agent has never seen
    test_targets = [10.0, 25.0, 50.0, 75.0]
    print(f"  {'target':>10s}  {'planned f':>10s}  {'actual d':>10s}  {'error':>8s}")
    for target in test_targets:
        planned_f = plan_throw(target, model)
        actual_d = world_throw(planned_f)
        err = abs(actual_d - target)
        print(f"  {target:>10.1f}  {planned_f:>10.2f}  {actual_d:>10.2f}  {err:>8.2f}")


    # ============================================================
    # 6. Phase 4 — what concept did the model discover?
    #    Probe the model's internal representation. Is there a 'force-like' axis?
    # ============================================================
    print("\n=== PHASE 4: Probe the learned representation ===")

    # Run a sweep of forces through the first hidden layer
    test_forces = torch.linspace(0, 30, 50).unsqueeze(1)
    with torch.no_grad():
        h1 = F.gelu(model.net[0](test_forces))   # [50, 32]

    # Find the hidden unit most correlated with force (the discovered "force" axis)
    correlations = torch.tensor([
        float(torch.corrcoef(torch.stack([h1[:, k], test_forces.squeeze()]))[0, 1])
        for k in range(h1.shape[1])
    ])
    best_unit = int(correlations.abs().argmax())
    print(f"  Hidden unit {best_unit} has correlation {correlations[best_unit]:.3f} with input force.")
    print(f"  This unit is the model's emergent 'force magnitude' representation.")
    print(f"  (We never told it about force — it discovered the most predictive axis.)")

for num_trials in [10, 50, 100, 500]:
    run_experiment(num_trials)


# ============================================================
# 7. The deeper lesson
# ============================================================
print("""
=== What just happened ===

1. Random motor babbling generated (action, outcome) pairs.
2. A predictive objective + bottleneck architecture compressed those pairs
   into a function: force → distance.
3. The function generalizes to unseen targets via gradient-based inverse.
4. The hidden representation discovered an axis that correlates with the
   'force magnitude' — the natural coordinate of this manifold.

The agent never read a physics textbook. It never heard the word 'force'.
It built a working theory of throwing through corrected practice.

This is the Ericsson-style deliberate-practice loop in code:
    try → observe → predict → error → update → repeat.

Concepts (force, distance) are coordinate axes of the manifold of useful
predictions. They emerged because they're the most efficient way to
compress this experience. Naming them comes later.
""")
