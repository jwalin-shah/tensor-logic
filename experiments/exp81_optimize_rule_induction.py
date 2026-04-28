"""
exp81: LLM-guided rule induction using the optimize loop.

Qwen2.5-0.5B acts as a pruner: given ASI feedback from the previous evaluation,
it narrows the relation set for the next brute-force induction pass. The optimize
loop drives the search; brute-force runs on the reduced space each iteration.

Falsification: if exp81 needs >= brute-force template count steps on easy mode,
the proposer is not helping and the loop design needs revision.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # project root for tensor_logic


# ---------- Artifact helpers ----------

def parse_artifact(artifact: str) -> Optional[dict]:
    """Parse proposer output. Returns None on invalid JSON or missing keys."""
    try:
        d = json.loads(artifact)
    except (json.JSONDecodeError, ValueError):
        return None
    if "relevant_relations" not in d or not isinstance(d["relevant_relations"], list):
        return None
    if "max_len" not in d or not isinstance(d["max_len"], int):
        return None
    return d


# ---------- ASI: explain why a positive pair is missed ----------

def explain_miss(body: list[str], base: dict, src: int, dst: int) -> str:
    """
    Explain why (src, dst) is not derived by rule body at the tensor level.
    Used to generate human-readable ASI for the Qwen proposer.
    """
    reachable = {src}
    for hop, rel in enumerate(body):
        T = base[rel]
        nxt: set[int] = set()
        for u in reachable:
            reached = (T[u] > 0.5).nonzero(as_tuple=True)[0].tolist()
            nxt.update(reached)
        if not nxt:
            return (
                f"chain breaks at hop {hop} ({rel}): "
                f"no {rel}-edges from entities {sorted(reachable)}"
            )
        reachable = nxt
    if dst not in reachable:
        return (
            f"chain reaches {sorted(reachable)} via {body}, "
            f"but dst={dst} not included"
        )
    return ""


# ---------- Proposer ----------

def make_proposer(schema, target_rel: str, positive: list, negative: list,
                  n_entities: int = 8,
                  model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Returns a propose(feedback: str) -> artifact: str closure.

    Wraps exp78's lm_prune. On each call, prepends ASI feedback to the
    prompt so the model can narrow the relation set based on previous failures.
    Artifact format: JSON string {"relevant_relations": [...], "max_len": int}
    """
    from exp78_rule_induction import lm_prune

    def propose(feedback: str) -> str:
        annotated_target = target_rel
        if feedback:
            annotated_target = f"{target_rel}  [prev failure: {feedback[:300]}]"

        result = lm_prune(
            schema=schema,
            target_rel=annotated_target,
            positive=positive,
            negative=negative,
            n_entities=n_entities,
            model_name=model_name,
        )
        artifact = json.dumps({
            "relevant_relations": result.get("relevant_relations", schema.rel_names()),
            "max_len": result.get("max_len", 3),
        })
        return artifact

    return propose


# ---------- Evaluator ----------

def _precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    tp = float((pred * target).sum())
    pp = float(pred.sum())
    return tp / pp if pp > 0 else 0.0


def make_evaluator(base: dict, target: torch.Tensor, positive: list, negative: list,
                   n_entities: int, target_rel: str):
    """
    Returns an evaluate(artifact: str) -> EvalResult closure.

    For each call:
      1. Parse artifact to get {relevant_relations, max_len}.
      2. Run induce_from_examples on the reduced relation set.
      3. Compute F1 (primary) and precision (secondary).
      4. For each missed positive pair, call explain_miss → append to ASI.
    """
    from exp78_rule_induction import induce_from_examples, apply_body, f1 as compute_f1
    from tensor_logic.optimize import EvalResult

    def evaluate(artifact: str) -> EvalResult:
        parsed = parse_artifact(artifact)
        if parsed is None:
            return EvalResult(
                artifact=artifact, score=0.0,
                asi="invalid artifact — could not parse JSON or missing keys",
                asi_kind="engine_error",
            )

        allowed_rels = parsed["relevant_relations"]
        max_len = parsed["max_len"]

        result = induce_from_examples(
            base=base,
            positive=positive,
            negative=negative,
            n_entities=n_entities,
            allowed_rels=allowed_rels if allowed_rels else None,
            max_len=max_len,
        )

        body = result.get("body")
        if body is None:
            return EvalResult(
                artifact=artifact, score=0.0,
                asi=f"no rule found in {allowed_rels} up to len {max_len}",
                asi_kind="why_not",
            )

        pred = apply_body(body, base)
        f1_score = compute_f1(pred, target)
        prec = _precision(pred, target)

        missed_explanations = []
        for src, dst in positive:
            if pred[src, dst] <= 0.5:
                expl = explain_miss(body, base, src, dst)
                if expl:
                    missed_explanations.append(f"missed ({src},{dst}): {expl}")

        asi_text = "\n".join(missed_explanations) if missed_explanations else f"rule {body} covers all positives"
        asi_kind = "why_not" if missed_explanations else "proof"

        return EvalResult(
            artifact=artifact,
            score=f1_score,
            secondary_score=prec,
            asi=asi_text,
            asi_kind=asi_kind,
        )

    return evaluate


# ---------- Full loop ----------

@dataclass
class Exp81Config:
    # target must be a key in exp78's GOLD_RULES: grandparent, uncle, skip_manager, etc.
    target: str = "grandparent"
    mode: str = "easy"            # "easy" | "medium" (distractors) | "hard" (noise+distractors)
    n_pos: int = 10
    n_neg: int = 10
    seed: int = 42
    max_steps: int = 50
    frontier_size: int = 5        # 15 for hard mode
    stagnation_k: int = 5
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    n_entities: int = 8


def run_exp81(cfg: Exp81Config) -> dict:
    """
    Run the optimize loop for one target/mode combination.
    Returns dict with: steps_to_f1_1, brute_force_template_count, accepted_f1, frontier_size.
    """
    import random
    from exp78_rule_induction import (
        GOLD_RULES, gen_world, apply_body, sample_examples, enumerate_rules,
        schema_with_distractors,
    )
    from tensor_logic.optimize import optimize, EvalResult

    random.seed(cfg.seed)

    schema, gold_body = GOLD_RULES[cfg.target]

    if cfg.mode in ("medium", "hard"):
        schema = schema_with_distractors(schema)

    base = gen_world(schema, n_entities=cfg.n_entities, seed=cfg.seed)
    target_tensor = apply_body(gold_body, base)

    if cfg.mode == "hard":
        from exp79_self_play_loop import corrupt_examples
        positive, negative = sample_examples(target_tensor, n_pos=3, n_neg=3, seed=cfg.seed)
        positive, negative = corrupt_examples(positive, negative, noise=0.2, seed=cfg.seed)
        frontier_size = 15
    else:
        positive, negative = sample_examples(target_tensor, n_pos=cfg.n_pos, n_neg=cfg.n_neg,
                                             seed=cfg.seed)
        frontier_size = cfg.frontier_size

    brute_force_count = sum(1 for _ in enumerate_rules(schema.rel_names(), max_len=3))

    propose = make_proposer(schema=schema, target_rel=cfg.target,
                            positive=positive, negative=negative,
                            n_entities=cfg.n_entities, model_name=cfg.model)
    evaluate = make_evaluator(base=base, target=target_tensor, positive=positive,
                              negative=negative, n_entities=cfg.n_entities,
                              target_rel=cfg.target)

    steps_to_success = None
    step_counter = [0]

    def accept(r: EvalResult) -> bool:
        step_counter[0] += 1
        if r.score >= 1.0:
            nonlocal steps_to_success
            steps_to_success = step_counter[0]
            return True
        return False

    frontier = optimize(
        propose, evaluate, accept,
        pareto_axes=("f1", "precision"),
        max_steps=cfg.max_steps,
        frontier_size=frontier_size,
        stagnation_k=cfg.stagnation_k,
    )

    accepted_f1 = frontier[0].score if frontier else 0.0
    return {
        "target": cfg.target,
        "mode": cfg.mode,
        "steps_to_f1_1": steps_to_success,
        "brute_force_template_count": brute_force_count,
        "accepted_f1": accepted_f1,
        "frontier_size": len(frontier),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="exp81: LLM-guided rule induction")
    parser.add_argument("--target", default="grandparent",
                        choices=["grandparent", "uncle", "skip_manager", "skip_peer",
                                 "great_uncle", "skip_skip_manager"])
    parser.add_argument("--mode", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Exp81Config(
        target=args.target,
        mode=args.mode,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    result = run_exp81(cfg)
    print(f"\n=== exp81 result ===")
    print(f"Target: {result['target']}  Mode: {result['mode']}")
    print(f"Steps to F1=1.0: {result['steps_to_f1_1']} / {cfg.max_steps}")
    print(f"Brute-force template count: {result['brute_force_template_count']}")
    print(f"Accepted F1: {result['accepted_f1']:.3f}")
    if result['steps_to_f1_1'] is not None:
        ratio = result['steps_to_f1_1'] / result['brute_force_template_count']
        verdict = "PASS" if ratio < 1.0 else "FAIL — loop not faster than brute force"
        print(f"Steps/Templates ratio: {ratio:.2f}  → {verdict}")
    else:
        print("F1=1.0 not reached within max_steps → FAIL")


if __name__ == "__main__":
    main()
