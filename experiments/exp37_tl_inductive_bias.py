"""
exp37: does tensor-logic consistency loss act as a useful inductive bias?

Train two tiny from-scratch transformers on the SAME synthetic call-graph data:
  (V) Vanilla — next-token cross-entropy only
  (T) +TL    — next-token CE + λ * MSE(model_relation_head, tensor_logic_closure)

Train on graphs with shallow queries (hop depth 1-2, some 3).
Test generalization on hop depth 3-5 (held-out depths).

Win condition: if (T) > (V) on Test-deep without losing on Test-shallow,
tensor-logic loss is shaping compositional generalization.
Falsify: equal accuracy on Test-deep across both runs.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- config ----------
SEED = 0
N_FUNCS_VOCAB = 50          # f00..f49
MAX_NODES_PER_GRAPH = 15
MIN_NODES_PER_GRAPH = 8
EDGE_P_LO, EDGE_P_HI = 0.15, 0.25
N_TRAIN_GRAPHS = 8000
N_VAL_GRAPHS = 500
N_TEST_GRAPHS = 1000
QUERIES_PER_GRAPH_TRAIN = 4
QUERIES_PER_GRAPH_EVAL = 6
TRAIN_DEPTH_DIST = {1: 0.45, 2: 0.40, 3: 0.15}     # mostly shallow + a little 3
TEST_SHALLOW_DEPTHS = [1, 2]
TEST_DEEP_DEPTHS = [3, 4, 5]

# model
D_MODEL = 128
N_LAYER = 4
N_HEAD = 4
DROPOUT = 0.0
MAX_SEQ_LEN = 256

# train
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 6
TL_LAMBDA = 1.0
N_SEEDS = 3

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"device={device}")

# ---------- vocab ----------
SPECIAL = ["<pad>", "<bos>", "<eos>"]
FUNCS = [f"f{i:02d}" for i in range(N_FUNCS_VOCAB)]
WORDS = ["calls", ".", "Query:", "transitively", "?", "Answer:", "Yes", "No"]
VOCAB = SPECIAL + FUNCS + WORDS
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
PAD = TOK2ID["<pad>"]
BOS = TOK2ID["<bos>"]
EOS = TOK2ID["<eos>"]
YES_ID = TOK2ID["Yes"]
NO_ID = TOK2ID["No"]
VOCAB_SIZE = len(VOCAB)
print(f"vocab size: {VOCAB_SIZE}")


# ---------- data generation ----------
def sample_dag(rng: random.Random) -> Tuple[List[str], List[Tuple[str, str]]]:
    n = rng.randint(MIN_NODES_PER_GRAPH, MAX_NODES_PER_GRAPH)
    node_ids = rng.sample(range(N_FUNCS_VOCAB), n)
    names = [FUNCS[i] for i in node_ids]
    p = rng.uniform(EDGE_P_LO, EDGE_P_HI)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((names[i], names[j]))
    return names, edges


def adjacency(names, edges):
    n = len(names)
    idx = {x: i for i, x in enumerate(names)}
    M = torch.zeros(n, n)
    for a, b in edges:
        M[idx[a], idx[b]] = 1.0
    return M, idx


def closure(M):
    R = M.clone()
    n = R.shape[0]
    for _ in range(n):
        Rn = ((R @ R + R) > 0).float()
        if torch.equal(Rn, R):
            break
        R = Rn
    return R


def shortest_hop(M, idx, a, b, max_hops=8):
    cur = M.clone()
    if cur[idx[a], idx[b]] > 0:
        return 1
    for k in range(2, max_hops + 1):
        cur = ((cur @ M) > 0).float()
        if cur[idx[a], idx[b]] > 0:
            return k
    return 0


def sample_query(names, M, R, idx, depth_dist: Dict[int, float], rng: random.Random):
    """Pick a (a, b, label, depth) query honoring depth distribution for Yes labels;
    No labels get depth=0 (we still record nominal depth = sampled depth bucket for splitting)."""
    n = len(names)
    # bucket reachable pairs by depth
    pos_by_depth: Dict[int, List[Tuple[str, str]]] = {}
    for i in range(n):
        for j in range(n):
            if i == j or R[i, j] == 0:
                continue
            d = shortest_hop(M, idx, names[i], names[j])
            pos_by_depth.setdefault(d, []).append((names[i], names[j]))
    neg_pairs = [(names[i], names[j]) for i in range(n) for j in range(n)
                 if i != j and R[i, j] == 0]

    depths = sorted(depth_dist.keys())
    weights = [depth_dist[d] for d in depths]
    label = rng.random() < 0.5

    if label:
        for _ in range(20):
            d = rng.choices(depths, weights=weights, k=1)[0]
            if d in pos_by_depth and pos_by_depth[d]:
                a, b = rng.choice(pos_by_depth[d])
                return a, b, True, d
        return None
    else:
        if not neg_pairs:
            return None
        a, b = rng.choice(neg_pairs)
        d = rng.choices(depths, weights=weights, k=1)[0]
        return a, b, False, d


def encode_example(names, edges, query):
    a, b, label, depth = query
    toks = [BOS]
    for u, v in edges:
        toks += [TOK2ID[u], TOK2ID["calls"], TOK2ID[v], TOK2ID["."]]
    toks += [TOK2ID["Query:"], TOK2ID[a], TOK2ID["transitively"], TOK2ID["calls"],
             TOK2ID[b], TOK2ID["?"], TOK2ID["Answer:"]]
    answer_pos = len(toks)            # position whose target is Yes/No
    toks += [YES_ID if label else NO_ID, EOS]
    return toks, answer_pos, label, depth


@dataclass
class Example:
    input_ids: torch.Tensor      # (L,)
    target_ids: torch.Tensor     # (L,) shifted by 1, -100 for ignore
    answer_pos: int              # position to read for query yes/no
    label: int                   # 1 yes, 0 no
    depth: int
    node_ids: torch.Tensor       # (M,) token ids for nodes in this graph (M <= MAX_NODES_PER_GRAPH)
    node_positions: torch.Tensor # (M,) last occurrence position of each node in the facts span
    closure_mat: torch.Tensor    # (M, M) ground-truth transitive closure
    n_nodes: int


def build_example(names, edges, query) -> Example:
    M_adj, idx = adjacency(names, edges)
    R = closure(M_adj)
    toks, ans_pos, label, depth = encode_example(names, edges, query)
    n_nodes = len(names)
    # compute last occurrence of each node within the facts span (positions before "Query:")
    facts_end = toks.index(TOK2ID["Query:"])
    last_pos = {}
    for p in range(facts_end):
        if ID2TOK[toks[p]] in TOK2ID and ID2TOK[toks[p]] in [n for n in names]:
            last_pos[ID2TOK[toks[p]]] = p
    # nodes that never appear in facts (no edges) get position = first appearance in query span;
    # for cleanliness, give them facts_start (BOS+1) as a fallback — they'll be masked as well
    for nm in names:
        if nm not in last_pos:
            last_pos[nm] = 0  # BOS position; we mask via closure_mask if needed
    node_positions = torch.tensor([last_pos[nm] for nm in names], dtype=torch.long)
    node_ids = torch.tensor([TOK2ID[nm] for nm in names], dtype=torch.long)

    input_ids = torch.tensor(toks, dtype=torch.long)
    target_ids = torch.full_like(input_ids, -100)
    target_ids[:-1] = input_ids[1:]   # next-token targets
    target_ids[-1] = -100             # final position has no target (we appended EOS already)

    return Example(
        input_ids=input_ids,
        target_ids=target_ids,
        answer_pos=ans_pos,
        label=int(label),
        depth=depth,
        node_ids=node_ids,
        node_positions=node_positions,
        closure_mat=R,
        n_nodes=n_nodes,
    )


def gen_dataset(n_graphs, queries_per_graph, depth_dist, rng) -> List[Example]:
    out: List[Example] = []
    attempts = 0
    while len(out) < n_graphs * queries_per_graph and attempts < n_graphs * 4:
        attempts += 1
        names, edges = sample_dag(rng)
        if not edges:
            continue
        M_adj, idx = adjacency(names, edges)
        R = closure(M_adj)
        for _ in range(queries_per_graph):
            q = sample_query(names, M_adj, R, idx, depth_dist, rng)
            if q is None:
                continue
            ex = build_example(names, edges, q)
            if len(ex.input_ids) > MAX_SEQ_LEN:
                continue
            out.append(ex)
    return out


# ---------- collate ----------
def collate(batch: List[Example]):
    L = max(e.input_ids.size(0) for e in batch)
    M = max(e.n_nodes for e in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, L), PAD, dtype=torch.long)
    target_ids = torch.full((bsz, L), -100, dtype=torch.long)
    answer_pos = torch.zeros(bsz, dtype=torch.long)
    labels = torch.zeros(bsz, dtype=torch.long)
    depths = torch.zeros(bsz, dtype=torch.long)
    node_pos = torch.zeros(bsz, M, dtype=torch.long)
    node_mask = torch.zeros(bsz, M, dtype=torch.bool)
    closure_mat = torch.zeros(bsz, M, M)
    pair_mask = torch.zeros(bsz, M, M)
    for i, e in enumerate(batch):
        ll = e.input_ids.size(0)
        input_ids[i, :ll] = e.input_ids
        target_ids[i, :ll] = e.target_ids
        answer_pos[i] = e.answer_pos
        labels[i] = e.label
        depths[i] = e.depth
        n = e.n_nodes
        node_pos[i, :n] = e.node_positions
        node_mask[i, :n] = True
        closure_mat[i, :n, :n] = e.closure_mat
        # pair_mask: 1 for valid (i,j) with i != j
        pm = torch.ones(n, n) - torch.eye(n)
        pair_mask[i, :n, :n] = pm
    return dict(
        input_ids=input_ids, target_ids=target_ids,
        answer_pos=answer_pos, labels=labels, depths=depths,
        node_pos=node_pos, node_mask=node_mask,
        closure_mat=closure_mat, pair_mask=pair_mask,
    )


# ---------- model ----------
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, max_seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # relation head: pairwise scoring of node hidden states
        self.rel_left = nn.Linear(d_model, d_model // 2, bias=False)
        self.rel_right = nn.Linear(d_model, d_model // 2, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        # causal mask
        mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = (input_ids == PAD)
        h = self.encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, h

    def relation_logits(self, h, node_pos, node_mask):
        # h: (B, L, D); node_pos: (B, M); node_mask: (B, M)
        B, M = node_pos.shape
        idx = node_pos.unsqueeze(-1).expand(-1, -1, h.size(-1))   # (B, M, D)
        h_nodes = h.gather(1, idx)                                # (B, M, D)
        a = self.rel_left(h_nodes)                                # (B, M, D/2)
        b = self.rel_right(h_nodes)                               # (B, M, D/2)
        scores = torch.einsum("bmd,bnd->bmn", a, b) / math.sqrt(a.size(-1))
        return scores  # logits over edges; sigmoid to get prob


# ---------- train / eval ----------
def step(model, batch, use_tl_loss, tl_lambda):
    logits, h = model(batch["input_ids"])
    ce = F.cross_entropy(logits.view(-1, VOCAB_SIZE), batch["target_ids"].view(-1), ignore_index=-100)
    tl = torch.tensor(0.0, device=logits.device)
    if use_tl_loss:
        rel = model.relation_logits(h, batch["node_pos"], batch["node_mask"])
        target = batch["closure_mat"]
        mask = batch["pair_mask"] * batch["node_mask"].unsqueeze(1).float() * batch["node_mask"].unsqueeze(2).float()
        tl = F.binary_cross_entropy_with_logits(rel, target, weight=mask, reduction="sum") / mask.sum().clamp(min=1)
    return ce + tl_lambda * tl, ce.detach(), tl.detach() if use_tl_loss else torch.tensor(0.0)


@torch.no_grad()
def evaluate(model, dataset, depth_buckets):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False)
    correct = {tuple(b): 0 for b in depth_buckets}
    total = {tuple(b): 0 for b in depth_buckets}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(batch["input_ids"])
        # gather logits at answer_pos for each example
        # logits at position p predict input_ids[p+1]; to score the Yes/No token
        # at answer_pos, read logits at answer_pos - 1.
        ap = batch["answer_pos"] - 1
        gathered = logits[torch.arange(logits.size(0), device=device), ap]   # (B, V)
        yes = gathered[:, YES_ID]
        no = gathered[:, NO_ID]
        pred = (yes > no).long()
        ok = (pred == batch["labels"]).cpu()
        depths = batch["depths"].cpu()
        for bucket in depth_buckets:
            mask = torch.zeros_like(depths, dtype=torch.bool)
            for d in bucket:
                mask |= (depths == d)
            correct[tuple(bucket)] += int(ok[mask].sum().item())
            total[tuple(bucket)] += int(mask.sum().item())
    return {tuple(b): (correct[tuple(b)] / total[tuple(b)] if total[tuple(b)] else 0.0,
                       total[tuple(b)]) for b in depth_buckets}


def train_one(seed, use_tl_loss, train_set, val_set, test_sets):
    torch.manual_seed(seed)
    random.seed(seed)
    model = TinyTransformer(VOCAB_SIZE, D_MODEL, N_LAYER, N_HEAD, MAX_SEQ_LEN, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)

    tag = "+TL " if use_tl_loss else "Vani"
    print(f"  [{tag} seed={seed}] params={n_params:,}")

    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        ce_acc, tl_acc, n = 0.0, 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, ce, tl = step(model, batch, use_tl_loss, TL_LAMBDA)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ce_acc += ce.item() * batch["input_ids"].size(0)
            tl_acc += tl.item() * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
        dt = time.time() - t0
        print(f"  [{tag} seed={seed}] epoch {ep+1}/{EPOCHS}  ce={ce_acc/n:.3f}  tl={tl_acc/n:.3f}  ({dt:.1f}s)")

    # final eval
    val_buckets = [TEST_SHALLOW_DEPTHS, TEST_DEEP_DEPTHS]
    res = {}
    for name, ds in test_sets.items():
        out = evaluate(model, ds, val_buckets)
        res[name] = out
    return res


def main():
    rng = random.Random(SEED)
    print("generating data…")
    train_set = gen_dataset(N_TRAIN_GRAPHS, QUERIES_PER_GRAPH_TRAIN, TRAIN_DEPTH_DIST, rng)
    val_set = gen_dataset(N_VAL_GRAPHS, QUERIES_PER_GRAPH_EVAL, TRAIN_DEPTH_DIST, rng)
    # test sets: ALL depths 1-5 sampled from new graphs (held-out) — we'll bucket at eval
    test_dist_full = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
    test_set = gen_dataset(N_TEST_GRAPHS, QUERIES_PER_GRAPH_EVAL, test_dist_full, rng)
    print(f"  train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")
    by_depth_train = {}
    for e in train_set:
        by_depth_train[e.depth] = by_depth_train.get(e.depth, 0) + 1
    print(f"  train depth dist: {dict(sorted(by_depth_train.items()))}")
    by_depth_test = {}
    for e in test_set:
        by_depth_test[e.depth] = by_depth_test.get(e.depth, 0) + 1
    print(f"  test  depth dist: {dict(sorted(by_depth_test.items()))}")

    test_sets = {"test": test_set}
    results = {"vanilla": [], "tl": []}
    for s in range(N_SEEDS):
        print(f"\n=== seed {s}: vanilla ===")
        r = train_one(seed=100 + s, use_tl_loss=False, train_set=train_set,
                      val_set=val_set, test_sets=test_sets)
        results["vanilla"].append(r)
        print(f"\n=== seed {s}: +TL ===")
        r = train_one(seed=100 + s, use_tl_loss=True, train_set=train_set,
                      val_set=val_set, test_sets=test_sets)
        results["tl"].append(r)

    # summarize
    def mean_acc(runs, split, bucket):
        accs = [r[split][tuple(bucket)][0] for r in runs]
        return sum(accs) / len(accs), accs

    print("\n=========== summary (test set, mean over seeds) ===========")
    for bucket_name, bucket in [("shallow (1-2)", TEST_SHALLOW_DEPTHS),
                                ("deep (3-5)", TEST_DEEP_DEPTHS)]:
        v_mean, v_all = mean_acc(results["vanilla"], "test", bucket)
        t_mean, t_all = mean_acc(results["tl"], "test", bucket)
        print(f"  {bucket_name:<14} vanilla={v_mean:.3f}  {[f'{a:.3f}' for a in v_all]}"
              f"   +TL={t_mean:.3f}  {[f'{a:.3f}' for a in t_all]}"
              f"   delta={t_mean - v_mean:+.3f}")


if __name__ == "__main__":
    main()
