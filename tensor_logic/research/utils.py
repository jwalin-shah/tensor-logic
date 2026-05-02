import itertools
import random
import time
from dataclasses import dataclass
import torch

# ---------- Schema + world generation ----------

@dataclass
class Schema:
    name: str
    relations: dict  # rel_name -> (arg_type_a, arg_type_b)

    def rel_names(self):
        return list(self.relations.keys())

def gen_world(schema: Schema, n_entities: int = 8, density: float = 0.25, seed: int = 0):
    """Generate random tensors for each base relation in schema."""
    rng = random.Random(seed)
    rels = {}
    for name, (ta, tb) in schema.relations.items():
        T = torch.zeros(n_entities, n_entities)
        for i in range(n_entities):
            for j in range(n_entities):
                if i != j and rng.random() < density:
                    T[i, j] = 1.0
        rels[name] = T
    return rels

# ---------- Rule Application & Scoring ----------

def enumerate_rules(rel_names: list, max_len: int):
    """All rule bodies of length 1..max_len as lists of relation names (w/ transposes)."""
    candidates = []
    primitives = []
    for r in rel_names:
        primitives.append(r)
        primitives.append(r + "^T")
    for k in range(1, max_len + 1):
        for combo in itertools.product(primitives, repeat=k):
            candidates.append(list(combo))
    return candidates

def apply_body(body: list, base: dict) -> torch.Tensor:
    """Compose relation matrices in body via Boolean-style matmul."""
    def get(name):
        if name.endswith("^T"):
            return base[name[:-2]].T.contiguous()
        return base[name]
    cur = get(body[0])
    for name in body[1:]:
        cur = (torch.einsum("xy,yz->xz", cur, get(name)) > 0).float()
    return cur

def f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute F1 score between prediction and target tensors."""
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    pr = tp / max(tp + fp, 1e-9)
    re = tp / max(tp + fn, 1e-9)
    return 2 * pr * re / max(pr + re, 1e-9)

def semantic_equiv(induced_body: list, gold_body: list, schema: Schema,
                   n_worlds: int = 50, n_entities: int = 8) -> float:
    """Fraction of random worlds where induced and gold produce identical tensors."""
    if induced_body is None:
        return 0.0
    matches = 0
    for s in range(n_worlds):
        base = gen_world(schema, n_entities=n_entities, seed=1000 + s)
        pred = apply_body(induced_body, base)
        gold = apply_body(gold_body, base)
        if torch.equal(pred, gold):
            matches += 1
    return matches / n_worlds

# ---------- Examples & Induction ----------

def sample_examples(target: torch.Tensor, n_pos: int, n_neg: int, seed: int):
    """Sample (i, j) pairs for positive and negative cases (excluding self-loops)."""
    rng = random.Random(seed)
    n = target.shape[0]
    pos_pairs = [(i, j) for i in range(n) for j in range(n) if target[i, j] > 0 and i != j]
    neg_pairs = [(i, j) for i in range(n) for j in range(n) if target[i, j] == 0 and i != j]
    rng.shuffle(pos_pairs); rng.shuffle(neg_pairs)
    return pos_pairs[:n_pos], neg_pairs[:n_neg]

def corrupt_examples(pos: list, neg: list, noise: float, seed: int) -> tuple[list, list]:
    """Flip `noise` fraction of labels between pos and neg."""
    if noise == 0.0:
        return pos, neg
    rng = random.Random(seed + 9999)
    pos, neg = list(pos), list(neg)
    n_flip = max(1, int(noise * min(len(pos), len(neg))))
    flip_pos = rng.sample(range(len(pos)), min(n_flip, len(pos)))
    flip_neg = rng.sample(range(len(neg)), min(n_flip, len(neg)))
    for pi, ni in zip(flip_pos, flip_neg):
        pos[pi], neg[ni] = neg[ni], pos[pi]
    return pos, neg

def sample_exclusive_examples(gold: torch.Tensor, base: dict, n_pos: int, n_neg: int,
                               seed: int) -> tuple[list, list]:
    """Sample positives that require multi-hop reasoning (not in any single base rel)."""
    rng = random.Random(seed)
    n = gold.shape[0]
    single_hop = torch.zeros(n, n)
    for t in base.values():
        if t.shape == (n, n):
            single_hop = (single_hop + t).clamp(0, 1)

    pos_pairs = [(i, j) for i in range(n) for j in range(n)
                 if gold[i, j] > 0 and single_hop[i, j] == 0 and i != j]
    neg_pairs = [(i, j) for i in range(n) for j in range(n)
                 if gold[i, j] == 0 and single_hop[i, j] == 0 and i != j]

    if len(pos_pairs) < n_pos or len(neg_pairs) < n_neg:
        return sample_examples(gold, n_pos, n_neg, seed)

    rng.shuffle(pos_pairs); rng.shuffle(neg_pairs)
    return pos_pairs[:n_pos], neg_pairs[:n_neg]

def induce_from_examples(base: dict, positive: list, negative: list,
                         n_entities: int, max_len: int = 3,
                         allowed_rels: list | None = None) -> dict:
    """Brute-force induction scored ONLY on the provided pos/neg pairs."""
    pos_set = set((i, j) for i, j in positive)
    neg_set = set((i, j) for i, j in negative)
    if not pos_set:
        return {"f1": 0.0, "body": None, "search_cost": 0, "search_time_s": 0.0}
    t0 = time.perf_counter()
    rel_names = allowed_rels if allowed_rels is not None else list(base.keys())
    candidates = enumerate_rules(rel_names, max_len)
    best = (-1.0, None)
    for body in candidates:
        try:
            pred = apply_body(body, base)
        except KeyError:
            continue
        tp = sum(1 for ij in pos_set if pred[ij[0], ij[1]] > 0)
        fn = len(pos_set) - tp
        fp = sum(1 for ij in neg_set if pred[ij[0], ij[1]] > 0)
        pr = tp / max(tp + fp, 1e-9)
        re = tp / max(tp + fn, 1e-9)
        score = 2 * pr * re / max(pr + re, 1e-9)
        if score > best[0] or (score == best[0] and best[1] and len(body) < len(best[1])):
            best = (score, body)
    return {
        "f1": best[0],
        "body": best[1],
        "search_cost": len(candidates),
        "search_time_s": time.perf_counter() - t0,
    }

def fixpoint_stable(base: dict, rules: list, max_iters: int = 10) -> bool:
    """Return True if derived tensors converge under iterative rule application."""
    derived: dict[str, torch.Tensor] = {}
    for _ in range(max_iters):
        new_derived: dict[str, torch.Tensor] = {}
        for name, body in rules:
            full_base = {**base, **derived}
            needed = [r[:-2] if r.endswith("^T") else r for r in body]
            if not all(k in full_base for k in needed):
                continue
            try:
                t = apply_body(body, full_base)
            except KeyError:
                continue
            new_derived[name] = t
        converged = all(
            name in derived and torch.equal(derived[name], t)
            for name, t in new_derived.items()
        )
        derived = new_derived
        if converged:
            return True
    return False
