"""
exp32 — Composed-attention probe in GPT-2 small

HYPOTHESIS: A pretrained transformer's attention matrices encode meaningful
relations (e.g. "X's father is Y"), and composing two heads (A_i @ A_j) over
those matrices recovers higher-order relations (e.g. "X's grandfather is Y")
without any training.

FALSIFIED IF: The best (single-head, head-pair-composition) result over all
12*12 heads / pairs gives < 0.3 Pearson correlation with the ground-truth
grandparent matrix.

SMALLEST TEST: Synthetic kinship sentences, GPT-2 small (124M, pretrained,
loaded from HF), extract attention, ground-truth correlation.

Setup details:
- Sentences are constructed so causal-mask allows the relation: parent name
  precedes child name. Template: "<G>. <P> is <G>'s son. <C> is <P>'s son."
  Then <C> appears last, can attend backward to <P> (parent) and <G> (grandparent).
- We use 30 distinct (G, P, C) triples drawn from a pool of common names.
- Ground truth: parent_matrix[child_pos, parent_pos] = 1 in each sentence;
  grandparent_matrix[child_pos, grandparent_pos] = 1.
- For each of the 144 attention heads, average across sentences.
- Find best single head for "parent" relation, best for "grandparent" relation.
- Then take the best parent head's matrix A and compute A @ A.
  Compare A@A to ground-truth grandparent matrix via Pearson correlation.
"""
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model

torch.manual_seed(0); np.random.seed(0)

# ── Build kinship corpus ─────────────────────────────────────────────────────
CANDIDATE_NAMES = ["Bob", "Carl", "Dan", "Eve", "Frank", "Greg", "Hank", "Ivan",
                   "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Pat", "Quinn",
                   "Rita", "Sam", "Tina", "Uma", "Vic", "Wade", "Xena", "Yara",
                   "Tom", "Ann", "Sue", "Joe", "Tim", "Ron", "Ned", "Lee", "Ted",
                   "Max", "Ben", "Amy", "Ray", "Eli"]

def make_sentence(g, p, c):
    # Order: G, P, C — earliest-first so child can attend BACK to parent/grandparent.
    # Prefix "Story:" avoids putting a name at sentence-start (BPE split issue).
    return f"Story: {g}'s son is {p}. {p}'s son is {c}."

tok_for_filter = GPT2Tokenizer.from_pretrained("gpt2")
NAMES = [n for n in CANDIDATE_NAMES if len(tok_for_filter.encode(" " + n)) == 1]
print(f"Filtered to {len(NAMES)} single-token names: {NAMES}")
triples = []
np.random.shuffle(NAMES)
for i in range(0, len(NAMES) - 2, 3):
    triples.append((NAMES[i], NAMES[i+1], NAMES[i+2]))
sentences = [make_sentence(*t) for t in triples]
print(f"Built {len(sentences)} kinship sentences. Sample:\n  {sentences[0]}\n")

# ── Load GPT-2 small ─────────────────────────────────────────────────────────
print("Loading GPT-2 small (124M)...")
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", attn_implementation="eager")  # eager so we get attentions out
model.eval()
N_LAYERS = model.config.n_layer  # 12
N_HEADS = model.config.n_head    # 12
print(f"  Loaded. {N_LAYERS} layers x {N_HEADS} heads = {N_LAYERS*N_HEADS} attention matrices.\n")

# ── Run model and collect attention + ground-truth positions ─────────────────
# For each sentence, find the token indices of G, P, C (last occurrence of each
# name token, since the C name appears last)
def find_name_positions(input_ids, name):
    """Find all token indices whose decoded text (with surrounding whitespace) contains the name.
    Returns list of indices in order."""
    positions = []
    for i, tid in enumerate(input_ids.tolist()):
        decoded = tok.decode([tid]).strip()
        if decoded == name:
            positions.append(i)
    return positions

sample_attentions = []  # list of [12, 12, T, T] tensors
sample_positions = []   # list of dicts {'g_pos': int, 'p_pos': int, 'c_pos': int, 'T': int}

for idx, (sent, (g, p, c)) in enumerate(zip(sentences, triples)):
    enc = tok(sent, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attn = torch.stack([a[0] for a in out.attentions])  # [12, 12, T, T]
    g_positions = find_name_positions(input_ids, g)
    p_positions = find_name_positions(input_ids, p)
    c_positions = find_name_positions(input_ids, c)
    if not (g_positions and p_positions and c_positions):
        continue
    # We want C (last) attending back to P (earlier) and G (earliest).
    # Use LAST occurrence of each — this puts everything maximally close to C and
    # ensures each name has been "seen" before C.
    g_pos = g_positions[-1]; p_pos = p_positions[-1]; c_pos = c_positions[-1]
    if not (g_pos < p_pos < c_pos):
        continue
    sample_attentions.append(attn)
    sample_positions.append({"g_pos": g_pos, "p_pos": p_pos, "c_pos": c_pos, "T": len(input_ids)})

print(f"Collected {len(sample_attentions)} usable sentences (others dropped due to tokenization quirks).\n")

# ── Score each head: how much does C attend to P (parent), and to G (grandparent)? ──
print("Per-head attention from child token to parent / grandparent (averaged across sentences):\n")
parent_scores = np.zeros((N_LAYERS, N_HEADS))
grand_scores = np.zeros((N_LAYERS, N_HEADS))
for attn, pos in zip(sample_attentions, sample_positions):
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            parent_scores[L, H] += attn[L, H, pos["c_pos"], pos["p_pos"]].item()
            grand_scores[L, H] += attn[L, H, pos["c_pos"], pos["g_pos"]].item()
parent_scores /= len(sample_attentions)
grand_scores /= len(sample_attentions)

# Find top heads for each
best_parent = np.unravel_index(np.argmax(parent_scores), parent_scores.shape)
best_grand = np.unravel_index(np.argmax(grand_scores), grand_scores.shape)
print(f"  Best PARENT-relation head:      L{best_parent[0]} H{best_parent[1]}   score={parent_scores[best_parent]:.3f}")
print(f"  Best GRANDPARENT-relation head: L{best_grand[0]} H{best_grand[1]}    score={grand_scores[best_grand]:.3f}\n")

# Top 5 of each for context
def top_k(scores, k=5):
    flat = [(s, np.unravel_index(i, scores.shape)) for i, s in enumerate(scores.flatten())]
    return sorted(flat, key=lambda x: -x[0])[:k]
print("  Top-5 heads for PARENT relation:")
for s, idx in top_k(parent_scores): print(f"    L{idx[0]:>2} H{idx[1]:>2}  score={s:.3f}")
print("\n  Top-5 heads for GRANDPARENT relation:")
for s, idx in top_k(grand_scores): print(f"    L{idx[0]:>2} H{idx[1]:>2}  score={s:.3f}")

# ── Composition test ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Composition test: does (best parent head) @ (best parent head) recover grandparent?")
print("=" * 70 + "\n")

# For each sentence, compute A @ A for the best parent head, and compare
# its [c_pos, g_pos] entry to the ground-truth (which is high) and to the
# average over [c_pos, all_other_positions] (which should be low).
compose_scores_g = []  # composed[c, g] for each sentence
compose_scores_baseline = []  # mean of composed[c, other_pos]

for attn, pos in zip(sample_attentions, sample_positions):
    A = attn[best_parent[0], best_parent[1]]  # [T, T]
    A2 = A @ A  # composition
    g_score = A2[pos["c_pos"], pos["g_pos"]].item()
    other_positions = [i for i in range(pos["T"]) if i not in (pos["g_pos"], pos["p_pos"], pos["c_pos"])]
    if not other_positions: continue
    baseline = A2[pos["c_pos"], other_positions].mean().item()
    compose_scores_g.append(g_score)
    compose_scores_baseline.append(baseline)

mean_g = np.mean(compose_scores_g)
mean_baseline = np.mean(compose_scores_baseline)
print(f"  Mean A^2[c, g]                   = {mean_g:.4f}")
print(f"  Mean A^2[c, random other token]  = {mean_baseline:.4f}")
print(f"  Ratio                            = {mean_g / max(mean_baseline, 1e-9):.2f}x")

# Pearson correlation between A@A scores and ground-truth grandparent indicator
# Build vectors over (sentence, position) pairs
gt = []
pred = []
for attn, pos in zip(sample_attentions, sample_positions):
    A = attn[best_parent[0], best_parent[1]]
    A2 = A @ A
    for j in range(pos["T"]):
        gt.append(1.0 if j == pos["g_pos"] else 0.0)
        pred.append(A2[pos["c_pos"], j].item())
gt = np.array(gt); pred = np.array(pred)
pearson = np.corrcoef(gt, pred)[0, 1]
print(f"\n  Pearson correlation (A^2[c,j] vs is-grandparent[j]) = {pearson:.3f}")

# ── Clean test: search only over heads that DON'T already encode grandparent ─
GRAND_MASK_THRESHOLD = 0.05  # if a head's solo grandparent score exceeds this, exclude it
clean_heads = [(L, H) for L in range(N_LAYERS) for H in range(N_HEADS)
               if grand_scores[L, H] <= GRAND_MASK_THRESHOLD]
print(f"\nClean test: searching only over {len(clean_heads)} heads with solo grandparent score <= {GRAND_MASK_THRESHOLD}")
print(f"  (excludes the {N_LAYERS*N_HEADS - len(clean_heads)} heads that already encode grandparent on their own)\n")

best_pair = None
best_pair_score = -1.0
for L1, H1 in clean_heads:
    for L2, H2 in clean_heads:
                # only consider L1 < L2 (composition flows up the layers naturally)
                if L1 > L2: continue
                # compute A2 = A1 @ A2 for one sentence to keep this tractable
                # actually do the average across all sentences
                avg_g = 0.0; avg_base = 0.0; n = 0
                for attn, pos in zip(sample_attentions[:5], sample_positions[:5]):
                    A1 = attn[L1, H1]; A2 = attn[L2, H2]
                    P_comp = A2 @ A1  # apply A1 first, then A2 on top
                    g_s = P_comp[pos["c_pos"], pos["g_pos"]].item()
                    other = [i for i in range(pos["T"]) if i not in (pos["g_pos"], pos["p_pos"], pos["c_pos"])]
                    if not other: continue
                    b_s = P_comp[pos["c_pos"], other].mean().item()
                    avg_g += g_s; avg_base += b_s; n += 1
                if n == 0: continue
                avg_g /= n; avg_base /= n
                # score = ratio (capped) — high means composed attention prefers grandparent
                ratio = avg_g / max(avg_base, 1e-9)
                if avg_g > best_pair_score:
                    best_pair_score = avg_g
                    best_pair = (L1, H1, L2, H2, avg_g, avg_base, ratio)

L1, H1, L2, H2, g, b, r = best_pair
print(f"  Best head-pair composition: L{L1}H{H1} ∘ L{L2}H{H2}")
print(f"    A^2[c,g] = {g:.4f}    baseline = {b:.4f}    ratio = {r:.2f}x")

# Now compute Pearson over full set with this best pair
gt = []; pred = []
for attn, pos in zip(sample_attentions, sample_positions):
    A1 = attn[L1, H1]; A2 = attn[L2, H2]
    P_comp = A2 @ A1
    for j in range(pos["T"]):
        gt.append(1.0 if j == pos["g_pos"] else 0.0)
        pred.append(P_comp[pos["c_pos"], j].item())
gt = np.array(gt); pred = np.array(pred)
pearson_pair = np.corrcoef(gt, pred)[0, 1]
print(f"    Pearson correlation                              = {pearson_pair:.3f}")

# ── Hypothesis check ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Hypothesis check")
print("=" * 70)
print(f"  Best single-head A@A Pearson:       {pearson:.3f}")
print(f"  Best head-pair (A1∘A2) Pearson:     {pearson_pair:.3f}")
threshold = 0.3
best_corr = max(pearson, pearson_pair)
verdict = "CONFIRMED — symbolic composition of attention recovers higher-order relations." \
          if best_corr > threshold else \
          "FALSIFIED — composition does not recover grandparent above 0.3 correlation."
print(f"\n  Threshold for confirmation: {threshold}")
print(f"  Verdict: {verdict}")
