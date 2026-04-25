"""
Joint LM + KG training as ONE tensor logic program.

Setup:
  - 8 people in a family tree, indexed 0..7. Same as train_kg.py.
  - Vocabulary = {person tokens 0..7} ∪ {<parent>, <grandparent>, <is>, <of>, <?>, <pad>}.
  - Text corpus: short generated sentences like "0 <parent> <of> 2"
                 and questions "<grandparent> <of> 7 <is> ?"
  - KG: Parent facts (boolean tensor), and the rule
        Grandparent(x,z) :- Parent(x,y), Parent(y,z)

Joint loss:
   α · CE( LM(text), next_token )
 + β · BCE( rule_apply_in_embedding_space(emb), Grandparent_truth )

Bridge: token embeddings for person-tokens ARE the KG object embeddings.
The same parameter row serves both the transformer and the rule. Gradients from
both losses sculpt one shared space.

Anneal: T (sigmoid temperature in the rule) goes 1.0 -> 0.05 over training.

Eval: ask "<grandparent> <of> 7 <is> ?" — does the LM produce token 2?
      Compare with the KG-only embedding-space query.
"""

import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

# ============================================================
# 1. Domain: family tree
# ============================================================
parent_edges = [(0, 2), (1, 2), (2, 4), (2, 5), (3, 5), (4, 6), (5, 7)]
N_PEOPLE = 8

P_true = torch.zeros(N_PEOPLE, N_PEOPLE)
for u, v in parent_edges:
    P_true[u, v] = 1.0
GP_true = ((P_true @ P_true) > 0).float()

# ============================================================
# 2. Vocabulary + corpus
# ============================================================
# Tokens 0..7 = person IDs (these embeddings are SHARED with the KG)
# Tokens 8..14 = special tokens
PARENT, GP, OF, IS, Q, PAD, BOS = 8, 9, 10, 11, 12, 13, 14
VOCAB = N_PEOPLE + 7
SEQ_LEN = 6

def encode_parent_fact(u, v):
    return [BOS, u, PARENT, OF, v, PAD]

def encode_gp_question(z, answer):
    # "<gp> <of> z <is> answer"
    return [BOS, GP, OF, z, IS, answer]

# Training corpus: parent facts + grandparent QA examples (small subset for training)
train_corpus = []
for u, v in parent_edges:
    train_corpus.append(encode_parent_fact(u, v))

# Use HALF the grandparent facts as text training, hold the rest for eval
gp_pairs = [(i, j) for i in range(N_PEOPLE) for j in range(N_PEOPLE) if GP_true[i, j]]
train_gp = gp_pairs[:len(gp_pairs)//2]
eval_gp = gp_pairs[len(gp_pairs)//2:]
for x, z in train_gp:
    train_corpus.append(encode_gp_question(z, x))

train_tensor = torch.tensor(train_corpus, dtype=torch.long)
print(f"Vocab={VOCAB}  corpus={len(train_corpus)} sequences  seq_len={SEQ_LEN}")
print(f"Train GP: {train_gp}")
print(f"Held-out GP for eval: {eval_gp}")

# ============================================================
# 3. Tiny transformer expressed as einsum equations
# ============================================================
D = 32       # embedding dim (= KG embedding dim too)
H = 4        # heads
DH = D // H  # head dim
N_LAYERS = 2

class EinsumTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The TIE: this single matrix is both the LM token embedding and the
        # KG object embedding for tokens 0..7.
        self.tok_emb = torch.nn.Parameter(torch.randn(VOCAB, D) * 0.1)
        self.pos_emb = torch.nn.Parameter(torch.randn(SEQ_LEN, D) * 0.1)

        self.Wq = torch.nn.Parameter(torch.randn(N_LAYERS, D, D) * 0.1)
        self.Wk = torch.nn.Parameter(torch.randn(N_LAYERS, D, D) * 0.1)
        self.Wv = torch.nn.Parameter(torch.randn(N_LAYERS, D, D) * 0.1)
        self.Wo = torch.nn.Parameter(torch.randn(N_LAYERS, D, D) * 0.1)
        self.W1 = torch.nn.Parameter(torch.randn(N_LAYERS, D, 4 * D) * 0.1)
        self.W2 = torch.nn.Parameter(torch.randn(N_LAYERS, 4 * D, D) * 0.1)

    def forward(self, tokens):  # tokens [B, S]
        B, S = tokens.shape
        # Eq 1: X[b,s,d] = tok_emb[token,d] + pos_emb[s,d]
        X = self.tok_emb[tokens] + self.pos_emb[:S]                     # [B,S,D]

        causal_mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)

        for L in range(N_LAYERS):
            Xn = F.layer_norm(X, (D,))
            # Multi-head attention as einsums:
            # Q[b,s,h,k] = Xn[b,s,d] · Wq[L,d,h*k]   (split heads)
            Q = torch.einsum("bsd,de->bse", Xn, self.Wq[L]).view(B, S, H, DH)
            K = torch.einsum("bsd,de->bse", Xn, self.Wk[L]).view(B, S, H, DH)
            V = torch.einsum("bsd,de->bse", Xn, self.Wv[L]).view(B, S, H, DH)

            # A[b,h,s,t] = softmax_t( Q[b,s,h,k] · K[b,t,h,k] / sqrt(DH) )
            scores = torch.einsum("bshk,bthk->bhst", Q, K) / math.sqrt(DH)
            scores = scores + causal_mask
            A = scores.softmax(dim=-1)

            # O[b,s,h,k] = A[b,h,s,t] · V[b,t,h,k]
            O = torch.einsum("bhst,bthk->bshk", A, V).reshape(B, S, D)
            # Y = O · Wo
            X = X + torch.einsum("bsd,de->bse", O, self.Wo[L])

            # MLP: H = gelu(X · W1); X = X + H · W2
            Xn = F.layer_norm(X, (D,))
            HID = F.gelu(torch.einsum("bsd,de->bse", Xn, self.W1[L]))
            X = X + torch.einsum("bse,ed->bsd", HID, self.W2[L])

        X = F.layer_norm(X, (D,))
        # Tied output projection: logits[b,s,v] = X[b,s,d] · tok_emb[v,d]
        logits = torch.einsum("bsd,vd->bsv", X, self.tok_emb)
        return logits

# ============================================================
# 4. KG rule head — uses the SAME tok_emb rows (people are tokens 0..7)
# ============================================================
def kg_rule_loss(model, T):
    """
    EmbParent[i,j] = sum_{(u,v) in P} emb[u,i] · emb[v,j]
    EmbGrandparent = EmbParent @ EmbParent     (the rule, in embedding space)
    pred[a,b] = sigmoid( EmbGP · emb[a] · emb[b] / T )
    """
    e = F.normalize(model.tok_emb[:N_PEOPLE], dim=1)
    EmbP = torch.einsum("uv,ui,vj->ij", P_true, e, e)
    EmbGP = EmbP @ EmbP
    scores = torch.einsum("ij,ai,bj->ab", EmbGP, e, e)
    pred = torch.sigmoid(scores / T)
    return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), GP_true), pred

def run_experiment(BETA):
    print(f"\n======================================")
    print(f"Experiment: BETA={BETA}")
    print(f"======================================")

    # ============================================================
    # 5. Train
    # ============================================================
    model = EinsumTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    ALPHA = 1.0   # weight on LM loss
    N_STEPS = 600

    print("\nTraining (joint LM + KG, T annealed 1.0 -> 0.05)...")
    for step in range(N_STEPS):
        # Anneal T
        T = 1.0 * (0.05 / 1.0) ** (step / max(1, N_STEPS - 1))

        # LM loss: predict next token
        logits = model(train_tensor[:, :-1])
        targets = train_tensor[:, 1:]
        lm_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=PAD
        )

        # KG rule loss
        rule_loss, _ = kg_rule_loss(model, T)

        loss = ALPHA * lm_loss + BETA * rule_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"  step {step:3d}  T={T:.3f}  lm={lm_loss.item():.3f}  "
                  f"rule={rule_loss.item():.3f}")

    # ============================================================
    # 6. Eval
    # ============================================================
    print("\n=== EVAL ===")
    model.eval()
    with torch.no_grad():
        # 6a. LM-side: ask grandparent questions including HELD-OUT pairs
        print("\nLM completion of '<gp> <of> z <is> ?' (greedy argmax over person tokens):")
        for x, z in gp_pairs:
            prompt = torch.tensor([[BOS, GP, OF, z, IS, PAD]])
            logits = model(prompt)
            # Restrict to person tokens (0..7)
            person_logits = logits[0, -2, :N_PEOPLE]
            pred = int(person_logits.argmax())
            seen = "TRAIN" if (x, z) in train_gp else "HELD-OUT"
            ok = "OK " if pred == x else "MISS"
            print(f"  {ok} [{seen:8s}] grandparent of {z} -> pred={pred}  truth={x}")

        # 6b. KG-side: query in embedding space directly (no transformer)
        print("\nKG embedding-space query (rule applied to learned emb, no LM):")
        e = F.normalize(model.tok_emb[:N_PEOPLE], dim=1)
        EmbP = torch.einsum("uv,ui,vj->ij", P_true, e, e)
        EmbGP = EmbP @ EmbP
        for x, z in gp_pairs:
            # "who are grandparents of z?" -> contract second index with e[z]
            q = torch.einsum("ij,j->i", EmbGP, e[z])
            scores = e @ q
            pred = int(scores.argmax())
            ok = "OK " if pred == x else "MISS"
            print(f"  {ok} grandparent of {z} -> pred={pred}  truth={x}")

        # 6c. Show the KG rule produces the right boolean matrix (T -> 0)
        _, pred_GP = kg_rule_loss(model, T=0.05)
        print("\nFinal predicted Grandparent matrix (threshold 0.5) vs truth:")
        print((pred_GP > 0.5).int())
        print("Truth:")
        print(GP_true.int())
        acc = ((pred_GP > 0.5).float() == GP_true).float().mean()
        print(f"Cell-wise accuracy: {acc.item():.3f}")

for BETA in [0.0, 1.0, 10.0]:
    run_experiment(BETA)
