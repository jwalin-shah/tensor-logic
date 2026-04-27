"""
exp78: TL-only rule induction across schemas.

Extends exp21_rule_induction (hardcoded family) to arbitrary typed schemas.
v1 has no LM — pure brute-force search. v2 will add LM-as-pruner.

Falsification (per docs/exp78_rule_induction_spec.md):
  TL-only must recover gold rules for grandparent/uncle/great_uncle from
  <=20 pos + <=20 neg examples in <=1s search per target. If not, the
  search space needs typing/pruning before LM can help.

Eval splits:
  seen_schema       — family schema (parent, sibling); same shapes as exp21.
  heldout_schema    — workplace schema (manages, reports_to); same rule shapes.
  heldout_compose   — 3-hop chains (great_uncle, skip_skip_manager) over both.
  distractor        — schema padded with 5 unused relations.

Metrics per target:
  rule_found        — top rule has F1 >= 0.9 on examples.
  semantic_equiv    — top rule answers match gold rule on 50 random worlds.
  search_cost       — number of candidates evaluated.
  search_time_s     — wall time.
"""
import argparse
import itertools
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

HERE = Path(__file__).parent


# ---------- Schema + world generation ----------

@dataclass
class Schema:
    name: str
    relations: dict  # rel_name -> (arg_type_a, arg_type_b)

    def rel_names(self):
        return list(self.relations.keys())


FAMILY = Schema("family", {
    "parent": ("person", "person"),
    "sibling": ("person", "person"),
})

WORKPLACE = Schema("workplace", {
    "manages": ("person", "person"),
    "peer_of": ("person", "person"),
})

DISTRACTORS = {
    "lives_in": ("person", "city"),
    "born_in": ("person", "city"),
    "owns": ("person", "thing"),
    "knows_lang": ("person", "lang"),
    "speaks": ("person", "lang"),
}


def gen_world(schema: Schema, n_entities: int = 8, density: float = 0.25, seed: int = 0):
    """Generate random tensors for each base relation in schema.

    Returns dict: rel_name -> N×N tensor. Only person×person rels for now;
    distractor rels with non-person types get random but unused tensors.
    """
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


def derive_target(base: dict, rule_body: list) -> torch.Tensor:
    """Apply a rule body (list of (rel, var_a, var_b)) to base relations.

    Body must be a chain X -> Y -> Z (-> W) sharing intermediate vars.
    Returns the head tensor.
    """
    # Compose left-to-right.
    cur = base[rule_body[0][0]]
    for atom in rule_body[1:]:
        T = base[atom[0]]
        cur = (torch.einsum("xy,yz->xz", cur, T) > 0).float()
    return cur


# ---------- Brute-force induction ----------

def enumerate_rules(rel_names: list, max_len: int):
    """All rule bodies of length 1..max_len as lists of relation names.

    Each rel can appear as itself (forward) or transposed (reverse). We model
    transpose by appending '^T' to the name; caller maps that to T.T.
    """
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
    def get(name):
        if name.endswith("^T"):
            return base[name[:-2]].T.contiguous()
        return base[name]
    cur = get(body[0])
    for name in body[1:]:
        cur = (torch.einsum("xy,yz->xz", cur, get(name)) > 0).float()
    return cur


def f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    pr = tp / max(tp + fp, 1e-9)
    re = tp / max(tp + fn, 1e-9)
    return 2 * pr * re / max(pr + re, 1e-9)


def induce(base: dict, target: torch.Tensor, max_len: int = 3) -> dict:
    """Brute-force search for the rule body that best matches target.

    Returns dict with best body, F1, search cost, time.
    """
    t0 = time.perf_counter()
    candidates = enumerate_rules(list(base.keys()), max_len)
    best = (-1.0, None)
    for body in candidates:
        pred = apply_body(body, base)
        score = f1(pred, target)
        if score > best[0] or (score == best[0] and best[1] and len(body) < len(best[1])):
            best = (score, body)
    return {
        "f1": best[0],
        "body": best[1],
        "search_cost": len(candidates),
        "search_time_s": time.perf_counter() - t0,
    }


# ---------- Examples-only induction (the realistic setup) ----------

def induce_from_examples(base: dict, positive: list, negative: list,
                         n_entities: int, max_len: int = 3) -> dict:
    """Induce rule from pos/neg example pairs.

    Score F1 over LABELED pairs only (standard ILP setup): TP = predicted+pos,
    FP = predicted+neg, FN = not-predicted+pos. Pairs not in pos∪neg are
    unlabeled and don't count either way.
    """
    pos_set = set((i, j) for i, j in positive)
    neg_set = set((i, j) for i, j in negative)
    if not pos_set:
        return {"f1": 0.0, "body": None, "search_cost": 0, "search_time_s": 0.0}
    t0 = time.perf_counter()
    candidates = enumerate_rules(list(base.keys()), max_len)
    best = (-1.0, None)
    for body in candidates:
        pred = apply_body(body, base)
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


# ---------- Eval splits ----------

GOLD_RULES = {
    # name -> (schema, body as list of relation atoms with ^T marker)
    "grandparent": (FAMILY, ["parent", "parent"]),
    "uncle":       (FAMILY, ["sibling", "parent"]),
    "great_uncle": (FAMILY, ["sibling", "parent", "parent"]),
    "skip_manager":     (WORKPLACE, ["manages", "manages"]),
    "skip_peer":        (WORKPLACE, ["peer_of", "manages"]),
    "skip_skip_manager":(WORKPLACE, ["manages", "manages", "manages"]),
}


def schema_with_distractors(schema: Schema) -> Schema:
    rels = dict(schema.relations)
    rels.update(DISTRACTORS)
    return Schema(schema.name + "+distract", rels)


def sample_examples(target: torch.Tensor, n_pos: int, n_neg: int, seed: int):
    rng = random.Random(seed)
    n = target.shape[0]
    pos_pairs = [(i, j) for i in range(n) for j in range(n) if target[i, j] > 0]
    neg_pairs = [(i, j) for i in range(n) for j in range(n) if target[i, j] == 0 and i != j]
    rng.shuffle(pos_pairs); rng.shuffle(neg_pairs)
    return pos_pairs[:n_pos], neg_pairs[:n_neg]


def semantic_equiv(induced_body: list, gold_body: list, schema: Schema,
                   n_worlds: int = 50, n_entities: int = 8) -> float:
    """Fraction of random worlds where induced and gold produce the same target tensor."""
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


def eval_split(name: str, targets: list, schema_for_world, n_pos: int = 20,
               n_neg: int = 20, n_entities: int = 8, max_len: int = 3) -> dict:
    """Run induction on each target; return per-target metrics."""
    out = {}
    for tgt in targets:
        gold_schema, gold_body = GOLD_RULES[tgt]
        eval_schema = schema_for_world(gold_schema)
        base = gen_world(eval_schema, n_entities=n_entities, seed=42)
        target_tensor = apply_body(gold_body, base)
        pos, neg = sample_examples(target_tensor, n_pos, n_neg, seed=7)
        if len(pos) == 0:
            out[tgt] = {"skipped": "no positives in sampled world"}
            continue
        result = induce_from_examples(base, pos, neg, n_entities, max_len=max_len)
        equiv = semantic_equiv(result["body"], gold_body, eval_schema,
                               n_worlds=20, n_entities=n_entities)
        out[tgt] = {
            "induced": result["body"],
            "gold": gold_body,
            "f1": round(result["f1"], 3),
            "semantic_equiv": equiv,
            "search_cost": result["search_cost"],
            "search_time_s": round(result["search_time_s"], 3),
            "rule_found": result["f1"] >= 0.9,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "exp78_data" / "results.json"))
    ap.add_argument("--n-pos", type=int, default=20)
    ap.add_argument("--n-neg", type=int, default=20)
    ap.add_argument("--n-entities", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3)
    args = ap.parse_args()

    splits = {
        "seen_schema":      (["grandparent", "uncle"],
                             lambda s: s),
        "heldout_schema":   (["skip_manager", "skip_peer"],
                             lambda s: s),
        "heldout_compose":  (["great_uncle", "skip_skip_manager"],
                             lambda s: s),
        "distractor":       (["grandparent", "uncle", "skip_manager"],
                             schema_with_distractors),
    }

    all_results = {}
    print(f"exp78: TL-only rule induction (max_len={args.max_len}, "
          f"n_pos={args.n_pos}, n_neg={args.n_neg})")
    print("=" * 70)
    for split_name, (targets, schema_fn) in splits.items():
        print(f"\n--- {split_name} ---")
        res = eval_split(split_name, targets, schema_fn,
                         n_pos=args.n_pos, n_neg=args.n_neg,
                         n_entities=args.n_entities, max_len=args.max_len)
        all_results[split_name] = res
        for tgt, m in res.items():
            if m.get("skipped"):
                print(f"  {tgt}: skipped ({m['skipped']})")
                continue
            mark = "✓" if m["rule_found"] and m["semantic_equiv"] >= 0.95 else "✗"
            print(f"  {mark} {tgt:<22} F1={m['f1']:.3f}  equiv={m['semantic_equiv']:.2f}  "
                  f"cost={m['search_cost']:>4}  t={m['search_time_s']:.3f}s")
            print(f"      gold:    {m['gold']}")
            print(f"      induced: {m['induced']}")

    # Aggregate falsification check.
    print("\n=== Falsification check ===")
    family_targets = ["grandparent", "uncle", "great_uncle"]
    family_results = {}
    for split, r in all_results.items():
        for t in family_targets:
            if t in r and not r[t].get("skipped"):
                family_results[t] = r[t]
    if len(family_results) == len(family_targets):
        all_found = all(m["rule_found"] for m in family_results.values())
        all_fast = all(m["search_time_s"] <= 1.0 for m in family_results.values())
        all_equiv = all(m["semantic_equiv"] >= 0.95 for m in family_results.values())
        print(f"  family rules found        : {'PASS' if all_found else 'FAIL'}")
        print(f"  family search <=1s        : {'PASS' if all_fast else 'FAIL'}")
        print(f"  family semantic equiv     : {'PASS' if all_equiv else 'FAIL'}")
        verdict = all_found and all_fast and all_equiv
        print(f"  v1 thesis (TL-only works) : "
              f"{'PASS — proceed to v2 LM pruner' if verdict else 'FAIL — prune search space first'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  results written: {out_path}")


if __name__ == "__main__":
    main()
