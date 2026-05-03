"""
exp79: Self-play rule factory loop (SGS pattern) — hardened test suite.
Refactored to use tensor_logic.research utilities and NegativeProof trigger.
"""
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensor_logic.research.utils import (
    Schema, gen_world, apply_body, induce_from_examples,
    semantic_equiv, sample_examples, corrupt_examples,
    sample_exclusive_examples, fixpoint_stable, f1 as compute_f1,
    enumerate_rules
)
from tensor_logic.research.constants import (
    CONTACTS, schema_with_distractors, QUERY_TARGETS
)

from tensor_logic import Program, prove_negative

# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    schema: Schema
    n_pos: int
    n_neg: int
    noise: float
    n_entities: int
    max_steps: int
    min_equiv: float
    f1_threshold: float
    n_attempts: int        # induction attempts per step for majority vote
    vote_threshold: float  # min F1 for a body to count as a vote
    min_support: int       # votes needed for winner to be proposed
    exclusive: bool = False  # use exclusive (multi-hop-only) examples

CONTACTS_WITH_DISTRACTORS = schema_with_distractors(CONTACTS)

EASY = Config(
    name="easy",
    schema=CONTACTS,
    n_pos=10, n_neg=10, noise=0.0,
    n_entities=50, max_steps=20,
    min_equiv=0.95, f1_threshold=0.85,
    n_attempts=1, vote_threshold=0.85, min_support=1,
)

MEDIUM = Config(
    name="medium",
    schema=CONTACTS_WITH_DISTRACTORS,
    n_pos=5, n_neg=5, noise=0.0,
    n_entities=50, max_steps=20,
    min_equiv=0.95, f1_threshold=0.85,
    n_attempts=5, vote_threshold=0.75, min_support=2,
)

HARD = Config(
    name="hard",
    schema=CONTACTS_WITH_DISTRACTORS,
    n_pos=5, n_neg=5, noise=0.20,
    n_entities=50, max_steps=20,
    min_equiv=0.95, f1_threshold=0.75,
    n_attempts=10, vote_threshold=0.50, min_support=3,
    exclusive=True,
)

VERY_HARD = Config(
    name="very_hard",
    schema=CONTACTS_WITH_DISTRACTORS,
    n_pos=3, n_neg=3, noise=0.20,
    n_entities=50, max_steps=20,
    min_equiv=0.95, f1_threshold=0.75,
    n_attempts=15, vote_threshold=0.40, min_support=3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def candidate_relation_names(schema: Schema) -> list[str]:
    """Only person-person relations can compose into query targets."""
    return [
        name for name, types in schema.relations.items()
        if types == ("person", "person")
    ]


def candidate_prediction_cache(base: dict, rel_names: list[str], max_len: int = 3) -> tuple[list[tuple], torch.Tensor]:
    keys = []
    preds = []
    for body in enumerate_rules(rel_names, max_len=max_len):
        try:
            preds.append(apply_body(body, base))
            keys.append(tuple(body))
        except KeyError:
            continue
    return keys, torch.stack(preds)


def cross_world_generalization(body: list, gold_body: list,
                                schema: Schema, seeds: list[int]) -> dict:
    """Test equiv on held-out worlds."""
    results = {}
    for s in seeds:
        base = gen_world(schema, n_entities=15, seed=s)
        # Only use base rels
        base_clean = {k: v for k, v in base.items() if k in schema.relations
                      or k.rstrip("^T") in schema.relations}
        try:
            pred = apply_body(body, base_clean)
            gold = apply_body(gold_body, base_clean)
            results[s] = int(torch.equal(pred, gold))
        except KeyError:
            results[s] = 0
    return results

# ---------------------------------------------------------------------------
# KB
# ---------------------------------------------------------------------------

class KB:
    def __init__(self, base: dict, schema: Schema, n: int):
        self.base = base
        self.schema = schema
        self.n = n
        self.rules: list[tuple[str, list]] = []

        self.program = Program()
        self.symbols = [f"p{i}" for i in range(n)]
        self.domain = self.program.domain("person", self.symbols)

        for name, tensor in base.items():
            if name not in self.program.relations:
                self.program.relation(name, "person", "person")
            indices = tensor.nonzero()
            for row, col in indices:
                self.program.fact(name, f"p{row.item()}", f"p{col.item()}")

        for target_name, _ in QUERY_TARGETS:
            if target_name not in self.program.relations:
                self.program.relation(target_name, "person", "person")

    def derive(self, name: str) -> torch.Tensor:
        for rname, body in self.rules:
            if rname == name:
                return apply_body(body, self.base)
        return torch.zeros(self.n, self.n)

    def has_rule(self, name: str) -> bool:
        return any(rname == name for rname, _ in self.rules)

    def add_rule(self, name: str, body: list) -> bool:
        if not fixpoint_stable(self.base, self.rules + [(name, body)]):
            return False
        self.rules.append((name, body))

        vars = ["X", "Y", "Z", "W", "V"]
        body_parts = []
        for idx, rel in enumerate(body):
            src_var = vars[idx]
            dst_var = vars[idx+1]
            if rel.endswith("^T"):
                real_rel = rel[:-2]
                body_parts.append(f"{real_rel}({dst_var}, {src_var})")
            else:
                body_parts.append(f"{rel}({src_var}, {dst_var})")

        tl_rule = f"{name}(X, {vars[len(body)]}) := " + " * ".join(body_parts)
        self.program.rule(tl_rule)
        return True

# ---------------------------------------------------------------------------
# Core self-play loop
# ---------------------------------------------------------------------------

def run_self_play(cfg: Config, seed: int = 42) -> dict:
    base = gen_world(cfg.schema, n_entities=cfg.n_entities, density=0.08, seed=seed)
    contacts_base = {k: v for k, v in base.items() if k in CONTACTS.relations}
    kb = KB(base, cfg.schema, cfg.n_entities)
    rel_names = candidate_relation_names(cfg.schema)
    body_keys, pred_stack = candidate_prediction_cache(base, rel_names, max_len=3)
    body_index = {body: i for i, body in enumerate(body_keys)}

    gold: dict[str, torch.Tensor] = {
        name: apply_body(body, contacts_base) for name, body in QUERY_TARGETS
    }

    def needs_induction(name: str) -> bool:
        pos_indices = gold[name].nonzero()
        if len(pos_indices) == 0:
            neg_proof = prove_negative(kb.program, name, "p0", "p1")
            return neg_proof is not None and neg_proof.reason == "no_rules"

        idx = pos_indices[0]
        src, dst = f"p{idx[0].item()}", f"p{idx[1].item()}"
        neg_proof = prove_negative(kb.program, name, src, dst)
        return neg_proof is not None and neg_proof.reason == "no_rules"

    def answerable(name: str) -> bool:
        if not kb.has_rule(name):
            return False
        pred = kb.derive(name)
        return compute_f1(pred, gold[name]) >= cfg.f1_threshold

    answered_at_0 = sum(1 for n, _ in QUERY_TARGETS if answerable(n))
    step_log = []

    for step in range(cfg.max_steps):
        target = next(((n, b) for n, b in QUERY_TARGETS if needs_induction(n)), None)
        if target is None:
            break

        t_name, gold_body = target
        t0 = time.perf_counter()

        # Majority vote induction logic
        votes: dict[tuple, int] = {}
        for attempt in range(cfg.n_attempts):
            attempt_seed = step * 100 + attempt
            if cfg.exclusive:
                p, n = sample_exclusive_examples(gold[t_name], base,
                                                  cfg.n_pos, cfg.n_neg, attempt_seed)
            else:
                p, n = sample_examples(gold[t_name], cfg.n_pos, cfg.n_neg, attempt_seed)
            if not p:
                continue
            p, n = corrupt_examples(p, n, cfg.noise, seed=attempt_seed)
            pos_i = torch.tensor([ij[0] for ij in p], dtype=torch.long)
            pos_j = torch.tensor([ij[1] for ij in p], dtype=torch.long)
            neg_i = torch.tensor([ij[0] for ij in n], dtype=torch.long)
            neg_j = torch.tensor([ij[1] for ij in n], dtype=torch.long)
            tp = pred_stack[:, pos_i, pos_j].sum(dim=1)
            fp = pred_stack[:, neg_i, neg_j].sum(dim=1)
            fn = len(p) - tp
            pr = tp / (tp + fp).clamp_min(1e-9)
            re = tp / (tp + fn).clamp_min(1e-9)
            scores = 2 * pr * re / (pr + re).clamp_min(1e-9)

            for idx in (scores >= cfg.vote_threshold).nonzero().flatten().tolist():
                body_key = body_keys[idx]
                votes[body_key] = votes.get(body_key, 0) + 1

        qualified = {k: v for k, v in votes.items() if v >= cfg.min_support}
        if qualified:
            max_votes = max(qualified.values())
            best_body = list(min(
                (k for k, v in qualified.items() if v == max_votes),
                key=lambda b: (len({r.rstrip("^T") for r in b}), len(b), b)
            ))
        else:
            best_body = None

        if best_body is not None:
            pred_report = pred_stack[body_index[tuple(best_body)]]
            f1_score = compute_f1(pred_report, gold[t_name])
            result = {"f1": f1_score, "body": best_body}
        else:
            result = {"f1": 0.0, "body": None}

        outcome = "failed_f1"
        equiv = 0.0
        accepted = False
        gen_scores: Optional[dict] = None

        if result["body"] is not None:
            equiv = semantic_equiv(result["body"], gold_body, cfg.schema,
                                   n_worlds=30, n_entities=12)
            if equiv >= cfg.min_equiv:
                accepted = kb.add_rule(t_name, result["body"])
                outcome = "accepted" if accepted else "rejected_unstable"
                if accepted:
                    gen_scores = cross_world_generalization(
                        result["body"], gold_body, CONTACTS,
                        seeds=[100, 101, 102, 103, 104],
                    )
            else:
                outcome = "failed_equiv"

        step_log.append({
            "step":         step,
            "target":       t_name,
            "outcome":      outcome,
            "induced":      result["body"],
            "gold":         gold_body,
            "f1":           round(result["f1"], 3),
            "equiv":        round(equiv, 3),
            "elapsed_s":    round(time.perf_counter() - t0, 3),
            "gen_scores":   gen_scores,
        })

    answered_after = sum(1 for n, _ in QUERY_TARGETS if answerable(n))
    return {
        "mode":           cfg.name,
        "answered_at_0":  answered_at_0,
        "answered_after": answered_after,
        "total_queries":  len(QUERY_TARGETS),
        "rules_induced":  len(kb.rules),
        "steps_taken":    len(step_log),
        "step_log":       step_log,
        "rules":          [(n, b) for n, b in kb.rules],
    }

# ---------------------------------------------------------------------------
# Adversarial check
# ---------------------------------------------------------------------------

def run_adversarial(cfg: Config, seed: int = 42, n_impossible: int = 3) -> dict:
    base = gen_world(cfg.schema, n_entities=cfg.n_entities, density=0.08, seed=seed)
    kb = KB(base, cfg.schema, cfg.n_entities)
    rel_names = candidate_relation_names(cfg.schema)

    rng = random.Random(seed + 77)
    attempts = []

    for i in range(n_impossible):
        target = torch.zeros(cfg.n_entities, cfg.n_entities)
        for r in range(cfg.n_entities):
            for c in range(cfg.n_entities):
                if r != c and rng.random() < 0.07:
                    target[r, c] = 1.0

        t_name = f"impossible_{i}"
        # Register in Program for NegativeProof check
        if t_name not in kb.program.relations:
            kb.program.relation(t_name, "person", "person")

        pos_pairs = [(r, c) for r in range(cfg.n_entities)
                     for c in range(cfg.n_entities) if target[r, c] > 0]
        neg_pairs = [(r, c) for r in range(cfg.n_entities)
                     for c in range(cfg.n_entities) if target[r, c] == 0 and r != c]
        rng.shuffle(pos_pairs); rng.shuffle(neg_pairs)
        pos = pos_pairs[:cfg.n_pos]
        neg = neg_pairs[:cfg.n_neg]

        if not pos:
            attempts.append({"target": t_name, "outcome": "skipped_no_pos"})
            continue

        res = induce_from_examples(base, pos, neg, cfg.n_entities, max_len=3,
                                   allowed_rels=rel_names)

        equiv = 0.0
        accepted = False
        outcome = "correctly_rejected"

        if res["f1"] >= cfg.f1_threshold and res["body"] is not None:
            matches = 0
            for s in range(20):
                b2 = gen_world(cfg.schema, n_entities=12, seed=200 + s)
                pred2 = apply_body(res["body"], b2)
                tgt2 = torch.zeros(12, 12)
                rng2 = random.Random(seed + s + 300)
                for r2 in range(12):
                    for c2 in range(12):
                        if r2 != c2 and rng2.random() < 0.07:
                            tgt2[r2, c2] = 1.0
                if torch.equal(pred2, tgt2):
                    matches += 1
            equiv = matches / 20
            if equiv >= cfg.min_equiv:
                accepted = kb.add_rule(t_name, res["body"])
                outcome = "FALSIFIED_accepted" if accepted else "rejected_unstable"
            else:
                outcome = "correctly_rejected"

        attempts.append({
            "target":    t_name,
            "outcome":   outcome,
            "best_f1":   round(res["f1"], 3),
            "equiv":     round(equiv, 3),
            "induced":   res["body"],
            "accepted":  accepted,
        })

    falsified = any(a.get("accepted") for a in attempts)
    return {
        "mode":       "adversarial",
        "falsified":  falsified,
        "attempts":   attempts,
    }

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def print_mode_result(r: dict):
    print(f"\n{'='*60}")
    print(f"Mode: {r['mode'].upper()}")
    print(f"{'='*60}")
    print(f"Queries answered: {r['answered_at_0']} → {r['answered_after']} / {r['total_queries']}")
    print(f"Rules induced:    {r['rules_induced']}  |  Steps taken: {r['steps_taken']}")
    print()
    print(f"{'Step':>4}  {'Target':<22}  {'Outcome':<20}  {'F1':>5}  {'Equiv':>5}  {'Gen':>5}  Body")
    print("-" * 95)
    for e in r["step_log"]:
        body_str = " ∘ ".join(e["induced"]) if e["induced"] else "—"
        gen_str = ""
        if e.get("gen_scores"):
            gen_mean = sum(e["gen_scores"].values()) / len(e["gen_scores"])
            gen_str = f"{gen_mean:.2f}"
        print(f"{e['step']:>4}  {e['target']:<22}  {e['outcome']:<20}  "
              f"{e['f1']:>5.3f}  {e['equiv']:>5.3f}  {gen_str:>5}  {body_str}")

def verdict(r: dict, cfg: Config) -> tuple[bool, str]:
    improvement = r["answered_after"] > r["answered_at_0"]
    accepted_steps = [e for e in r["step_log"] if e["outcome"] == "accepted"]
    all_equiv_ok = all(e["equiv"] >= cfg.min_equiv for e in accepted_steps)
    gen_ok = all(
        sum(e["gen_scores"].values()) / len(e["gen_scores"]) >= 0.8
        for e in accepted_steps if e.get("gen_scores")
    )
    reasons = []
    if not improvement and r['answered_after'] < r['total_queries']:
        reasons.append("coverage did not improve")
    if not all_equiv_ok:
        reasons.append(f"accepted rule(s) with equiv < {cfg.min_equiv}")
    if not gen_ok:
        reasons.append("poor cross-world generalization (<0.8 on held-out worlds)")
    ok = not reasons
    return ok, (f"✓ PASS" if ok else f"✗ FAIL — {'; '.join(reasons)}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path("experiments") / "exp79_data" / "results.json"))
    ap.add_argument("--mode", choices=["all", "easy", "medium", "hard", "very_hard"], default="all")
    args = ap.parse_args()

    configs = {"easy": EASY, "medium": MEDIUM, "hard": HARD, "very_hard": VERY_HARD}
    modes = list(configs.keys()) if args.mode == "all" else [args.mode]

    print("exp79: self-play rule factory loop — refactored")
    print(f"Seed={args.seed}  Queries={len(QUERY_TARGETS)}")

    all_results = {}
    overall_pass = True

    for mode_name in modes:
        cfg = configs[mode_name]
        t0 = time.perf_counter()
        r = run_self_play(cfg, seed=args.seed)
        r["wall_s"] = round(time.perf_counter() - t0, 2)
        print_mode_result(r)
        if mode_name == "very_hard":
            ok = True
            msg = "~ EXPECTED FAIL (identifiability boundary)"
        else:
            ok, msg = verdict(r, cfg)
        print(f"\n{msg}  ({r['wall_s']}s)")
        if not ok:
            overall_pass = False
        all_results[mode_name] = r

    if args.mode in ("all", "hard", "very_hard"):
        adv = run_adversarial(HARD, seed=args.seed)
        if adv["falsified"]:
            overall_pass = False
        all_results["adversarial"] = adv

    print(f"\n{'='*60}")
    print(f"Overall: {'✓ ALL PASS' if overall_pass else '✗ FAILURES DETECTED'}")
    print(f"{'='*60}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults → {out}")

if __name__ == "__main__":
    main()
