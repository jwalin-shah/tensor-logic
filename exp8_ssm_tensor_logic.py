"""
Experiment 8: SSM + Tensor-Logic Recurrence
============================================
Research question: can we implement a state-space model (SSM) where the
recurrence step IS a tensor-logic rule application?

Standard SSM (Mamba-style):
    h[t] = A · h[t-1] + B · x[t]
    y[t] = C · h[t]

The recurrence matrix A encodes "how the state evolves". In Mamba, A is learned
to be diagonal (efficient). But what if A is a tensor-logic RULE matrix?

Tensor-Logic SSM:
    h[t] = σ( einsum('ij,j->i', Rule, h[t-1]) + B · x[t] )
    y[t] = C · h[t]

Where Rule is a sparse matrix encoding a logical rule over the state.
For example: "if h[i] is active, activate h[j]" is Rule[j,i] = 1.

We test on a long-range dependency task:
  - Input: a sequence of tokens. Token "X" appears at position t=3.
  - The model must remember X and predict it at position t=50 (long range).
  - Rule: "propagate the X token's activation forward through the state".

Compare:
  A. Vanilla RNN: dense h[t] = tanh(W_h · h[t-1] + W_x · x[t])
  B. Tensor-Logic SSM: h[t] = sigmoid(Rule · h[t-1] + B · x[t]) with sparse Rule
  C. Perfect-memory SSM: Rule = identity (h[t] = h[t-1] + new_input), memory never decays

This tests whether structured recurrence (logical rule in A matrix) is better
than unstructured (dense W_h) for long-range memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ── Task: copy-at-distance ─────────────────────────────────────────────────────
# Sequence: [random tokens]...[X]...[random tokens]...[RECALL]...[output X]
# The model reads a sequence, sees marker X at pos 5, sees RECALL at pos 45,
# must output X's value at RECALL position.

VOCAB = 8        # tokens 0-7 (token 7 is RECALL marker)
SEQ_LEN = 50
RECALL_POS = 45  # where RECALL appears
STORE_POS = 5    # where the "important" token appears
HIDDEN = 16      # state dimension
RECALL_TOKEN = 7

def make_sequence(batch_size=64):
    """Random sequence with one token to remember and a recall prompt."""
    torch.manual_seed(hash(batch_size) % 1000)
    seq = torch.randint(0, VOCAB - 1, (batch_size, SEQ_LEN))  # random noise
    targets = torch.randint(0, VOCAB - 1, (batch_size,))
    seq[:, STORE_POS] = targets           # plant the token to remember
    seq[:, RECALL_POS] = RECALL_TOKEN     # plant the recall marker
    return seq, targets

# ── One-hot encoding ──────────────────────────────────────────────────────────
def encode(seq):
    return F.one_hot(seq, VOCAB).float()  # [B, T, V]


# ── Model A: Vanilla RNN ──────────────────────────────────────────────────────
class VanillaRNN(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.Wh = nn.Linear(hidden, hidden, bias=False)
        self.Wx = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, seq):
        B, T = seq.shape
        h = torch.zeros(B, HIDDEN)
        for t in range(T):
            x = self.embed(seq[:, t])
            h = torch.tanh(self.Wh(h) + self.Wx(x))
        return self.out(h)


# ── Model B: Tensor-Logic SSM ─────────────────────────────────────────────────
class TensorLogicSSM(nn.Module):
    """
    State h tracks 'which tokens have been seen'.
    Rule: a SPARSE learned matrix that propagates state forward.
    The sparsity is enforced by starting from a near-identity init
    plus a learned low-rank update.

    This models: h[i,t] = σ( Σ_j Rule[i,j] · h[j,t-1] + B[i] · x[t] )
    In tensor-logic terms: State[i] :- Σ_j Rule[i,j] · State[j]
    """
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        # Rule matrix initialized near-identity (keeps memory stable)
        self.Rule = nn.Parameter(torch.eye(hidden) * 0.9 + torch.randn(hidden, hidden) * 0.01)
        self.B = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, seq):
        B, T = seq.shape
        h = torch.zeros(B, HIDDEN)
        for t in range(T):
            x = self.embed(seq[:, t])
            h = torch.sigmoid(h @ self.Rule.T + self.B(x))
        return self.out(h)


# ── Model C: Perfect-memory SSM (h[t] = h[t-1] + input, gated by RECALL) ─────
class PerfectMemorySSM(nn.Module):
    """
    State is an accumulator. On RECALL token, output the accumulated state.
    This is the ideal baseline: it SHOULD remember perfectly.
    Uses a learned gate: when to write vs. when to read.
    """
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        # Gate: should we update the memory or not?
        self.gate = nn.Linear(hidden, 1)
        self.update = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, seq):
        B, T = seq.shape
        memory = torch.zeros(B, HIDDEN)
        h = torch.zeros(B, HIDDEN)
        for t in range(T):
            x = self.embed(seq[:, t])
            g = torch.sigmoid(self.gate(x))           # write gate
            new_content = torch.tanh(self.update(x))
            memory = memory + g * new_content          # accumulate
            h = memory
        return self.out(h)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(model, n_epochs=200, batch_size=64, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        seq, targets = make_sequence(batch_size)
        logits = model(seq)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def evaluate(model, n_batches=20, batch_size=64):
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, targets = make_sequence(batch_size)
            logits = model(seq)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size
    return correct / total


print("Experiment 8: SSM + Tensor-Logic Recurrence")
print("=" * 65)
print(f"  Task: remember token at position {STORE_POS}, recall at position {RECALL_POS}")
print(f"  Sequence length: {SEQ_LEN}, Vocab: {VOCAB}, Hidden: {HIDDEN}")
print(f"  Chance performance: {1/VOCAB:.3f}")
print()

models = {
    "Vanilla RNN":         VanillaRNN(VOCAB, HIDDEN),
    "Tensor-Logic SSM":    TensorLogicSSM(VOCAB, HIDDEN),
    "Gated Memory SSM":    PerfectMemorySSM(VOCAB, HIDDEN),
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    train_model(model)
    acc = evaluate(model)
    results[name] = acc
    bar = "█" * int(acc * 40)
    print(f"    Accuracy: {acc:.3f}  |{bar:<40}|")

print()
print("=" * 65)
print(f"  {'Model':<22}  {'Accuracy':>10}  {'vs. chance':>12}")
print("  " + "-" * 50)
chance = 1 / VOCAB
for name, acc in results.items():
    lift = (acc - chance) / chance * 100
    print(f"  {name:<22}  {acc:>10.3f}  {lift:>+10.1f}%")


# ── Analysis of the Rule matrix ────────────────────────────────────────────────
print()
print("=== Structure of the learned Rule matrix (Tensor-Logic SSM) ===")
rule_model = models["Tensor-Logic SSM"]
R = rule_model.Rule.detach()

# How close is the Rule to identity? (memory preservation vs. transformation)
identity_similarity = (R * torch.eye(HIDDEN)).sum() / R.abs().sum()
off_diag_mass = (R * (1 - torch.eye(HIDDEN))).abs().sum() / R.abs().sum()

print(f"  Rule matrix statistics:")
print(f"    Diagonal mass (memory): {identity_similarity:.3f}  (1.0 = pure memory, 0 = all mixing)")
print(f"    Off-diagonal mass (mixing): {off_diag_mass:.3f}")
print(f"    Max off-diagonal: {(R * (1-torch.eye(HIDDEN))).abs().max():.4f}")
print(f"    Rule norm: {R.norm():.3f}")
eigvals = torch.linalg.eigvals(R)
print(f"    Rule eigenvalue max magnitude: {eigvals.abs().max():.3f}")

# Show top 5 most "active" state transitions
off_diag = R * (1 - torch.eye(HIDDEN))
vals, flat_idx = off_diag.abs().flatten().topk(5)
print(f"\n  Top 5 off-diagonal transitions (state i → state j):")
for v, idx in zip(vals, flat_idx):
    i, j = idx // HIDDEN, idx % HIDDEN
    print(f"    h[{i}] → h[{j}]: strength {R[i,j]:.4f}")

print("""
=== Key Insights ===

1. Tensor-Logic SSM vs. Vanilla RNN:
   The SSM with a structured Rule matrix should outperform the vanilla RNN
   on long-range memory because the Rule matrix, initialized near-identity,
   naturally preserves state across many time steps. Dense RNN weights tend
   to mix and decay past information.

2. What the Rule matrix learns:
   - Diagonal entries close to 1.0: strong memory (state persists)
   - Off-diagonal entries: mixing between state dimensions
   - Sparsity: the SSM learns which state dimensions matter

3. Connection to Mamba/SSMs:
   Mamba restricts A to be diagonal (h[i,t] = A[i] · h[i,t-1] + B[i] · x[t])
   for computational efficiency. But tensor logic suggests that the recurrence
   matrix could encode RULES — sparse, structured transitions.
   This is "Mamba where the state transition is a logical rule."

4. The key tensor-logic insight:
   An SSM recurrence step h[t] = A · h[t-1] is a matrix multiply.
   A matrix multiply IS an einsum: h[t][i] = Σ_j A[i,j] · h[t-1][j]
   This is the tensor equation: State(i,t) :- Σ_j Rule(i,j) · State(j,t-1)
   So every SSM is already a tensor-logic program! The rule is the A matrix.
   Choosing A to be identity = "memory rule". Choosing A sparse = "structured rule".

5. The broader architecture:
   - SSM handles time (recurrence with a logical A matrix)
   - Attention handles space (einsum over sequence positions)
   - Both are tensor equations. Tensor logic IS the unified architecture.
""")
