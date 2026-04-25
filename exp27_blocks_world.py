"""
Experiment 27: Tensor Logic as a World Model — Blocks World
===========================================================
World models need three things:
  1. State representation  — what the world looks like now
  2. Transition function   — given action A, what changes
  3. Planning              — simulate forward to reach a goal

Tensor logic maps directly onto all three:
  1. State = relation matrices: On[i,j], Clear[i], Holding[i]
  2. Actions = tensor rule applications (add + delete effects)
  3. Planning = BFS over states via rule applications

The blocks world is the classic AI planning domain (STRIPS, 1971):
  - N blocks + 1 table
  - Blocks stack on each other or sit on the table
  - One arm picks up / puts down one block at a time

Perception bridge (conceptual):
  pixels → CNN → patch features → Slot Attention → node embeddings E_i
  relation score(i,j) = E_i @ Ar @ E_j  (superposition, exp23)
  Ar is learned; E_i comes from visual perception of the block
"""

import torch
from collections import deque

torch.manual_seed(42)

# ── World representation ──────────────────────────────────────────────────────
# Nodes: 0=Table, 1=A, 2=B, 3=C, 4=D
TABLE = 0
N = 5
NAMES = {0: "Table", 1: "A", 2: "B", 3: "C", 4: "D"}

class BlocksWorld:
    """
    World state as tensor logic relation matrices.
      On[i,j]   = block i is directly on block j  (j=0 is table)
      Clear[i]  = block i has nothing on top of it
      Holding[i] = arm is holding block i
    """
    def __init__(self, on_pairs, holding=None):
        self.On = torch.zeros(N, N)
        for i, j in on_pairs:
            self.On[i, j] = 1.0
        # Clear[j] = 1 iff nothing is on j
        self.Clear = (1.0 - self.On.sum(dim=0).clamp(0, 1))
        self.Clear[TABLE] = 0.0        # table is not a valid stack target in Clear sense
        self.Holding = torch.zeros(N)
        if holding is not None:
            self.Holding[holding] = 1.0
        self.HandEmpty = float(1.0 - self.Holding.sum().clamp(0, 1))

    def clone(self):
        s = BlocksWorld.__new__(BlocksWorld)
        s.On      = self.On.clone()
        s.Clear   = self.Clear.clone()
        s.Holding = self.Holding.clone()
        s.HandEmpty = self.HandEmpty
        return s

    def to_key(self):
        return (tuple(self.On.flatten().int().tolist()),
                tuple(self.Holding.int().tolist()))

    def render(self):
        stacks = {}
        on_table = [i for i in range(1, N) if self.On[i, TABLE] > 0]
        for base in on_table:
            stack = [base]
            cur = base
            while True:
                above = [i for i in range(1, N) if self.On[i, cur] > 0]
                if not above:
                    break
                cur = above[0]
                stack.append(cur)
            stacks[base] = stack

        held = [i for i in range(1, N) if self.Holding[i] > 0]
        lines = []
        if held:
            lines.append(f"  [ARM holding: {NAMES[held[0]]}]")
        max_h = max((len(v) for v in stacks.values()), default=0)
        for h in range(max_h, 0, -1):
            row = "  "
            for base in sorted(stacks.keys()):
                s = stacks[base]
                row += f"[{NAMES[s[h-1]]}] " if h <= len(s) else "    "
            lines.append(row)
        cols = max(len(stacks), 1)
        lines.append("  " + "----" * cols)
        lines.append("  TABLE")
        return "\n".join(lines)


# ── Action rules = tensor logic transitions ───────────────────────────────────
# Each action checks preconditions (tensor entry tests) then updates matrices.

def pickup(s, b):
    """PickUp(b): On[b,table] ∧ Clear[b] ∧ HandEmpty  →  Holding[b], ¬On[b,table]"""
    if not (s.On[b, TABLE] > 0 and s.Clear[b] > 0 and s.HandEmpty > 0):
        return None
    ns = s.clone()
    ns.On[b, TABLE] = 0.0
    ns.Holding[b]   = 1.0
    ns.HandEmpty    = 0.0
    return ns

def putdown(s, b):
    """PutDown(b): Holding[b]  →  On[b,table], ¬Holding[b], HandEmpty"""
    if not (s.Holding[b] > 0):
        return None
    ns = s.clone()
    ns.On[b, TABLE] = 1.0
    ns.Holding[b]   = 0.0
    ns.HandEmpty    = 1.0
    ns.Clear[b]     = 1.0
    return ns

def stack(s, b, x):
    """Stack(b,x): Holding[b] ∧ Clear[x]  →  On[b,x], ¬Holding[b], ¬Clear[x]"""
    if not (s.Holding[b] > 0 and s.Clear[x] > 0 and x != TABLE and b != x):
        return None
    ns = s.clone()
    ns.On[b, x]   = 1.0
    ns.Holding[b] = 0.0
    ns.HandEmpty  = 1.0
    ns.Clear[x]   = 0.0
    ns.Clear[b]   = 1.0
    return ns

def unstack(s, b, x):
    """Unstack(b,x): On[b,x] ∧ Clear[b] ∧ HandEmpty  →  Holding[b], ¬On[b,x], Clear[x]"""
    if not (s.On[b, x] > 0 and s.Clear[b] > 0 and s.HandEmpty > 0 and x != TABLE):
        return None
    ns = s.clone()
    ns.Holding[b] = 1.0
    ns.HandEmpty  = 0.0
    ns.On[b, x]   = 0.0
    ns.Clear[x]   = 1.0
    return ns

def successors(state):
    blocks = list(range(1, N))
    moves = []
    for b in blocks:
        ns = pickup(state, b)
        if ns: moves.append((f"PickUp({NAMES[b]})", ns))
        ns = putdown(state, b)
        if ns: moves.append((f"PutDown({NAMES[b]})", ns))
        for x in blocks:
            ns = stack(state, b, x)
            if ns: moves.append((f"Stack({NAMES[b]},{NAMES[x]})", ns))
            ns = unstack(state, b, x)
            if ns: moves.append((f"Unstack({NAMES[b]},{NAMES[x]})", ns))
    return moves


# ── BFS planner ───────────────────────────────────────────────────────────────
def bfs(start, goal_fn, max_states=50000):
    queue   = deque([(start, [])])
    visited = {start.to_key()}
    explored = 0
    while queue:
        state, plan = queue.popleft()
        explored += 1
        if goal_fn(state):
            return plan, state, explored
        if explored >= max_states:
            break
        for name, ns in successors(state):
            k = ns.to_key()
            if k not in visited:
                visited.add(k)
                queue.append((ns, plan + [name]))
    return None, None, explored


# ── Scenes ────────────────────────────────────────────────────────────────────
print("Experiment 27: Tensor Logic World Model — Blocks World")
print("=" * 65)
print()
print("  State  = relation matrices: On[i,j], Clear[i], Holding[i]")
print("  Action = rule application (updates to relation matrices)")
print("  Plan   = BFS over tensor logic state space")
print()

# ── Scene 1: reverse a tower ──────────────────────────────────────────────────
print("─" * 65)
print("  SCENE 1: Reverse a tower")
print("  Start: C on B on A    →    Goal: A on B on C")
print()
# C(3) on B(2) on A(1) on table
s1 = BlocksWorld([(1, TABLE), (2, 1), (3, 2)])
print("  Start:")
print(s1.render())
print(f"\n  On matrix (rows=what, cols=where):")
header = "       " + " ".join(f"{NAMES[j]:>5}" for j in range(N))
print(f"  {header}")
for i in range(1, N):
    row = "  " + f"{NAMES[i]:>5}  " + " ".join(f"{int(s1.On[i,j]):>5}" for j in range(N))
    print(row)
print(f"\n  Clear: { {NAMES[i]: int(s1.Clear[i].item()) for i in range(1,N)} }")
print()

def goal1(s):
    return (s.On[1, 2] > 0 and s.On[2, 3] > 0 and
            s.On[3, TABLE] > 0 and s.HandEmpty > 0)

plan1, end1, exp1 = bfs(s1, goal1)
print(f"  Plan ({len(plan1)} steps, {exp1} states explored):")
for i, a in enumerate(plan1, 1):
    print(f"    {i:2}. {a}")
print()
print("  Goal:")
print(end1.render())

# ── Scene 2: build a tower from scratch ───────────────────────────────────────
print()
print("─" * 65)
print("  SCENE 2: Build a tower from scattered blocks")
print("  Start: A B C D all on table  →  Goal: D on C on B on A")
print()
s2 = BlocksWorld([(1, TABLE), (2, TABLE), (3, TABLE), (4, TABLE)])
print("  Start:")
print(s2.render())
print()

def goal2(s):
    return (s.On[2, 1] > 0 and s.On[3, 2] > 0 and
            s.On[4, 3] > 0 and s.On[1, TABLE] > 0 and s.HandEmpty > 0)

plan2, end2, exp2 = bfs(s2, goal2)
print(f"  Plan ({len(plan2)} steps, {exp2} states explored):")
for i, a in enumerate(plan2, 1):
    print(f"    {i:2}. {a}")
print()
print("  Goal:")
print(end2.render())

# ── Scene 3: rearrange two separate stacks ────────────────────────────────────
print()
print("─" * 65)
print("  SCENE 3: Merge two stacks")
print("  Start: B on A,  D on C (two stacks)  →  Goal: D on C on B on A")
print()
s3 = BlocksWorld([(1, TABLE), (2, 1), (3, TABLE), (4, 3)])
print("  Start:")
print(s3.render())
print()

def goal3(s):
    return (s.On[2, 1] > 0 and s.On[3, 2] > 0 and
            s.On[4, 3] > 0 and s.On[1, TABLE] > 0 and s.HandEmpty > 0)

plan3, end3, exp3 = bfs(s3, goal3)
print(f"  Plan ({len(plan3)} steps, {exp3} states explored):")
for i, a in enumerate(plan3, 1):
    print(f"    {i:2}. {a}")
print()
print("  Goal:")
print(end3.render())


# ── Show matrix transitions step by step ──────────────────────────────────────
print()
print("=" * 65)
print("  TENSOR VIEW: trace how matrices change through plan steps")
print("  (Scene 1, first 3 steps)")
print()

state = s1.clone()
steps_to_show = plan1[:3]
for step_i, action in enumerate(steps_to_show):
    print(f"  Step {step_i+1}: {action}")
    print(f"    On nonzero: { [(NAMES[i],NAMES[j]) for i in range(N) for j in range(N) if state.On[i,j]>0] }")
    print(f"    Clear:      { {NAMES[i]: int(state.Clear[i].item()) for i in range(1,N)} }")
    held = [NAMES[i] for i in range(1,N) if state.Holding[i]>0]
    print(f"    Holding:    {held if held else 'nothing'}")
    # Apply action
    for name, ns in successors(state):
        if name == action:
            state = ns
            break
    print()

print(f"  Step {len(steps_to_show)+1}: (after above)")
print(f"    On nonzero: { [(NAMES[i],NAMES[j]) for i in range(N) for j in range(N) if state.On[i,j]>0] }")
print(f"    Clear:      { {NAMES[i]: int(state.Clear[i].item()) for i in range(1,N)} }")
held = [NAMES[i] for i in range(1,N) if state.Holding[i]>0]
print(f"    Holding:    {held if held else 'nothing'}")


# ── Perception bridge ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  PERCEPTION BRIDGE: how pixels become node embeddings")
print()
print("  The full pipeline:")
print()
print("  [image of blocks]")
print("       ↓  CNN (or ViT) — extract visual features per image patch")
print("  [patch feature grid]")
print("       ↓  Slot Attention — K slots compete to explain image regions")
print("  [E_1, E_2, ..., E_K]  ← one embedding vector per detected object")
print("       ↓  Superposition: score(i,j) = sigmoid(E_i @ Ar @ E_j)")
print("  [On[i,j] predictions]  ← relation matrices from visual perception")
print("       ↓  Tensor logic planning (this experiment)")
print("  [action sequence]")
print()
print("  Slot Attention (Locatello et al., 2020) is the key module:")
print("  - Initialises K 'slots' (learned or random)")
print("  - Slots compete via attention to explain different image patches")
print("  - Each slot 'captures' one object — unsupervised, no labels needed")
print("  - Output: K vectors of dim d, one per object = node embeddings E_i")
print()

import torch.nn as nn

# Minimal demonstration: position+color features → On predictions
class PerceptionModel(nn.Module):
    """
    Toy perception: (x, y, r, g, b) visual features → embed → predict On[i,j].
    Real version: pixels → ViT → Slot Attention → E_i
    Relation scoring: E_i @ Ar @ E_j  (superposition construction, exp23)
    """
    def __init__(self, feat_dim=5, embed_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, embed_dim))
        self.Ar = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.1)

    def forward(self, feats):
        E = self.encoder(feats)                         # [N, embed_dim]
        scores = torch.einsum("id,de,je->ij", E, self.Ar, E)
        return torch.sigmoid(scores), E                 # On predictions + embeddings

# Visual features for start1 state (x_pos, y_pos, r, g, b)
# Table: center-bottom, gray; A: left-low, red; B: left-mid, blue;
# C: left-high, green; D: right, yellow (on table separately)
feats = torch.tensor([
    [0.50, 0.00, 0.5, 0.5, 0.5],   # Table
    [0.20, 0.10, 1.0, 0.2, 0.2],   # A (on table)
    [0.20, 0.20, 0.2, 0.2, 1.0],   # B (on A)
    [0.20, 0.30, 0.2, 1.0, 0.2],   # C (on B)
    [0.75, 0.10, 1.0, 1.0, 0.2],   # D (separate, on table)
])

# Train briefly on ground-truth On matrix from s1
model = PerceptionModel()
opt = torch.optim.Adam(model.parameters(), lr=0.02)
target = s1.On.clone()
for _ in range(500):
    pred, _ = model(feats)
    loss = nn.functional.binary_cross_entropy(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    pred_on, E = model(feats)
    pred_bin = (pred_on > 0.5).float()
    correct = (pred_bin == target).float().mean().item()

print(f"  Trained PerceptionModel on 5 blocks × (x,y,r,g,b) features:")
print(f"  Embedding dim: {E.shape[1]}, Ar shape: {tuple(model.Ar.shape)}")
print(f"  On[i,j] prediction accuracy: {correct:.1%}  (vs ground truth)")
print()
print(f"  Predicted On matrix (rows=what, cols=where):")
header = "       " + " ".join(f"{NAMES[j]:>5}" for j in range(N))
print(f"  {header}")
for i in range(1, N):
    row = "  " + f"{NAMES[i]:>5}  " + " ".join(f"{pred_bin[i,j].int().item():>5}" for j in range(N))
    print(row)
print()
print(f"  Ground truth On matrix:")
print(f"  {header}")
for i in range(1, N):
    row = "  " + f"{NAMES[i]:>5}  " + " ".join(f"{int(target[i,j].item()):>5}" for j in range(N))
    print(row)

print("""
=== Key Insights ===

1. World state IS relation matrices. On[i,j], Clear[i], Holding[i] are
   exactly the same tensors we've used in exp1-exp26. Tensor logic was
   always a world model — we just didn't call it that.

2. Actions = tensor rule applications with add + delete effects.
   Each STRIPS action is a pattern match (precondition check on matrix
   entries) followed by a matrix update. No new mechanism needed —
   this is the same einsum/update pattern from all prior experiments.

3. Planning = forward BFS over the tensor logic state space.
   Each step expands by applying all valid action rules. The plan is
   the sequence of rule applications that reach the goal state.
   This is the ancestor fixpoint (exp23) generalised to action spaces.

4. STRIPS (1971) vs tensor logic: STRIPS hand-codes the rules.
   Tensor logic can LEARN them from observed (state, action, next_state)
   triples — same rule induction as exp21/exp22, now over action effects.

5. Perception bridge: Slot Attention (Locatello et al. 2020) segments
   images into object slots without supervision. Each slot = one node.
   Slot vector = E_i. Then relation score(i,j) = E_i @ Ar @ E_j
   (superposition from exp23). Ar is the only learned relational weight.

6. Why this beats pure neural world models (DreamerV3/MuZero):
   - Plans are explicit and readable: you can inspect each step
   - Rules transfer: the same PickUp rule works for any block
   - Planning is exact BFS, not probabilistic rollout
   - But: requires correct object segmentation (Slot Attention is imperfect
     on complex scenes) and assumes discrete, structured action space.

7. The complete grounded world model stack:
   Perception:  pixels → Slot Attention → E_i
   Relations:   E_i @ Ar @ E_j → On[i,j] (learned from experience)
   Transitions: action rules update relation matrices (this experiment)
   Planning:    BFS over tensor logic state transitions
   This is the architecture the field is converging toward — and the
   two pieces (tensor logic + slot attention) already exist separately.
""")
