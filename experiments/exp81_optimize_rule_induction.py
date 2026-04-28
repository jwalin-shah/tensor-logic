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
    def get_rel(name):
        if name.endswith("^T"):
            return base[name[:-2]].T.contiguous()
        return base[name]

    reachable = {src}
    for hop, rel in enumerate(body):
        T = get_rel(rel)
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
                  model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                  min_max_len: int = 1):
    """
    Returns a propose(feedback: str) -> artifact: str closure.

    Wraps exp78's lm_prune. On each call, prepends ASI feedback to the
    prompt so the model can narrow the relation set based on previous failures.
    Artifact format: JSON string {"relevant_relations": [...], "max_len": int}
    min_max_len: floor on max_len in artifact (set to 4 for extra_hard).

    Programmatic exclusion: when wrong_rule ASI fires, the spurious relation
    names are parsed and filtered from all future proposals — no model
    instruction-following required.
    """
    import re as _re
    from exp78_rule_induction import lm_prune

    excluded_rels: set[str] = set()
    _stagnant_count = [0]
    _last_wrong: dict = {}   # {rels_key, f1} from previous step

    def propose(feedback: str) -> str:
        # Accumulate excluded rels from wrong_rule ASI in two ways:
        # 1. Immediate: F1 ≤ 0.45 — clearly bad rule, exclude right away
        # 2. Stagnation: same wrong_rule fires 3+ steps in a row — stuck, exclude even if F1 > 0.45
        if "distractor" in feedback:
            f1_match = _re.search(r"target F1=([\d.]+)", feedback)
            f1_val = float(f1_match.group(1)) if f1_match else 1.0
            rels_match = _re.search(r"rule uses (\[[^\]]+\])", feedback)
            new_rels = []
            if rels_match:
                try:
                    new_rels = json.loads(rels_match.group(1).replace("'", '"'))
                except (json.JSONDecodeError, ValueError):
                    pass

            rels_key = tuple(sorted(new_rels))
            if f1_val <= 0.45:
                excluded_rels.update(new_rels)
                _stagnant_count[0] = 0
            elif rels_key == _last_wrong.get("rels_key") and f1_val == _last_wrong.get("f1"):
                _stagnant_count[0] += 1
                if _stagnant_count[0] >= 3:
                    excluded_rels.update(new_rels)
                    _stagnant_count[0] = 0
            else:
                _stagnant_count[0] = 0
            _last_wrong["rels_key"] = rels_key
            _last_wrong["f1"] = f1_val

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

        proposed = result.get("relevant_relations", schema.rel_names())
        filtered = [r for r in proposed if r not in excluded_rels]
        if not filtered:
            filtered = [r for r in schema.rel_names() if r not in excluded_rels] or schema.rel_names()

        artifact = json.dumps({
            "relevant_relations": filtered,
            "max_len": max(min_max_len, result.get("max_body_length", 3)),
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

        if missed_explanations:
            asi_text = "\n".join(missed_explanations)
            asi_kind = "why_not"
        elif f1_score < 1.0:
            # Sample fully covered but target F1 is bad — spurious rule
            rels_used = sorted({r.replace("^T", "") for r in body})
            asi_text = (
                f"found rule {body} (sample F1=1.0 but target F1={f1_score:.2f}) — "
                f"rule uses {rels_used} which may be distractor relations; "
                f"try more specific predicates for '{target_rel}' and ignore {rels_used}"
            )
            asi_kind = "wrong_rule"
        else:
            asi_text = f"rule {body} covers all positives"
            asi_kind = "proof"

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
    mode: str = "easy"            # "easy" | "medium" | "hard" (noise+distractors, n=3) | "hard_v2" (noise+distractors, n=10) | "extra_hard"
    n_pos: int = 10
    n_neg: int = 10
    seed: int = 42
    max_steps: int = 50
    frontier_size: int = 5        # 15 for hard mode
    stagnation_k: int = 5
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    n_entities: int = 8
    debug: bool = False


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

    if cfg.mode in ("medium", "hard", "hard_v2", "extra_hard"):
        schema = schema_with_distractors(schema)

    base = gen_world(schema, n_entities=cfg.n_entities, seed=cfg.seed)
    target_tensor = apply_body(gold_body, base)

    if cfg.mode == "hard":
        from exp79_self_play_loop import corrupt_examples
        positive, negative = sample_examples(target_tensor, n_pos=3, n_neg=3, seed=cfg.seed)
        positive, negative = corrupt_examples(positive, negative, noise=0.2, seed=cfg.seed)
        frontier_size = 15
        max_len = 3
    elif cfg.mode == "hard_v2":
        from exp79_self_play_loop import corrupt_examples
        positive, negative = sample_examples(target_tensor, n_pos=10, n_neg=10, seed=cfg.seed)
        positive, negative = corrupt_examples(positive, negative, noise=0.2, seed=cfg.seed)
        frontier_size = 15
        max_len = 3
    elif cfg.mode == "extra_hard":
        from exp79_self_play_loop import corrupt_examples
        positive, negative = sample_examples(target_tensor, n_pos=3, n_neg=3, seed=cfg.seed)
        positive, negative = corrupt_examples(positive, negative, noise=0.4, seed=cfg.seed)
        frontier_size = 20
        max_len = 4
    else:
        positive, negative = sample_examples(target_tensor, n_pos=cfg.n_pos, n_neg=cfg.n_neg,
                                             seed=cfg.seed)
        frontier_size = cfg.frontier_size
        max_len = 3

    # Pre-flight checks
    missing_rels = [r for r in gold_body if r.replace("^T", "") not in schema.rel_names()]
    if missing_rels:
        raise ValueError(f"gold_body rels {missing_rels} not in schema {schema.rel_names()}")
    if len(gold_body) > max_len:
        raise ValueError(
            f"gold_body length {len(gold_body)} > max_len {max_len} — rule unreachable; "
            f"increase max_len or use extra_hard mode"
        )
    available_pos = int((target_tensor > 0).float().fill_diagonal_(0).sum().item())
    if available_pos < cfg.n_pos:
        raise ValueError(
            f"target has only {available_pos} non-self-loop positive pairs, "
            f"need {cfg.n_pos} — try a different seed or reduce n_pos"
        )

    brute_force_count = sum(1 for _ in enumerate_rules(schema.rel_names(), max_len=max_len))

    propose = make_proposer(schema=schema, target_rel=cfg.target,
                            positive=positive, negative=negative,
                            n_entities=cfg.n_entities, model_name=cfg.model,
                            min_max_len=max_len)
    evaluate = make_evaluator(base=base, target=target_tensor, positive=positive,
                              negative=negative, n_entities=cfg.n_entities,
                              target_rel=cfg.target)

    steps_to_success = None
    step_counter = [0]
    step_log: list[dict] = []  # always tracked
    total_rels = len(schema.rel_names())

    _base_propose = propose
    _base_evaluate = evaluate
    _last_artifact_box = [None]

    def propose(feedback: str) -> str:
        art = _base_propose(feedback)
        _last_artifact_box[0] = art
        return art

    def evaluate(artifact: str) -> EvalResult:
        result = _base_evaluate(artifact)
        parsed = parse_artifact(artifact)
        proposed_rels = parsed["relevant_relations"] if parsed else schema.rel_names()
        gold_covered = all(r.replace("^T", "") in proposed_rels for r in gold_body)
        step_log.append({
            "gold_covered": gold_covered,
            "n_proposed": len(proposed_rels),
            "f1": result.score,
            "asi_kind": result.asi_kind,
        })
        return result

    if cfg.debug:
        print(f"\ngold_body: {gold_body}")
        print(f"positive: {positive}")
        print(f"negative: {negative}")
        print(f"\n{'step':>4}  {'gold_covered':>12}  {'proposed_rels':<40}  {'rule_found':<30}  {'f1':>5}  asi_kind")
        print("-" * 120)

        _orig_propose = propose
        _orig_evaluate = evaluate
        _last_artifact = [None]

        def propose(feedback: str) -> str:
            art = _orig_propose(feedback)
            _last_artifact[0] = art
            return art

        def evaluate(artifact: str) -> EvalResult:
            result = _orig_evaluate(artifact)
            parsed = parse_artifact(artifact)
            proposed_rels = parsed["relevant_relations"] if parsed else []
            gold_covered = all(r in proposed_rels for r in gold_body)
            rule_found = json.loads(artifact).get("relevant_relations", []) if parsed else []
            # Extract rule from ASI text if proof
            rule_str = ""
            if result.asi_kind == "proof" or (result.score > 0 and "rule" in result.asi):
                rule_str = result.asi[:30]
            elif result.asi_kind == "engine_error":
                rule_str = "ERROR"
            print(
                f"{step_counter[0] + 1:>4}  {'YES' if gold_covered else 'NO':>12}  "
                f"{str(proposed_rels):<40}  {rule_str:<30}  {result.score:>5.3f}  {result.asi_kind}"
            )
            return result

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

    gold_covered_rate = (
        sum(1 for s in step_log if s["gold_covered"]) / len(step_log)
        if step_log else 0.0
    )
    pruning_rate = (
        sum(1 - s["n_proposed"] / total_rels for s in step_log) / len(step_log)
        if step_log else 0.0
    )
    asi_kinds = [s["asi_kind"] for s in step_log]
    if steps_to_success is not None:
        failure_mode = None
    elif gold_covered_rate < 0.5:
        failure_mode = "gold_excluded"
    elif accepted_f1 == 0.0:
        failure_mode = "no_rule"
    elif "wrong_rule" in asi_kinds:
        failure_mode = "wrong_rule"
    else:
        failure_mode = "stagnated"

    return {
        "target": cfg.target,
        "mode": cfg.mode,
        "steps_to_f1_1": steps_to_success,
        "brute_force_template_count": brute_force_count,
        "accepted_f1": accepted_f1,
        "frontier_size": len(frontier),
        "failure_mode": failure_mode,
        "gold_covered_rate": round(gold_covered_rate, 3),
        "proposer_pruning_rate": round(pruning_rate, 3),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="exp81: LLM-guided rule induction")
    parser.add_argument("--target", default="grandparent",
                        choices=["grandparent", "uncle", "skip_manager", "skip_peer",
                                 "great_uncle", "skip_skip_manager"])
    parser.add_argument("--mode", default="easy", choices=["easy", "medium", "hard", "hard_v2", "extra_hard"])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    cfg = Exp81Config(
        target=args.target,
        mode=args.mode,
        max_steps=args.max_steps,
        seed=args.seed,
        debug=args.debug,
        model=args.model,
    )
    result = run_exp81(cfg)
    print(f"\n=== exp81 result ===")
    print(f"Target: {result['target']}  Mode: {result['mode']}")
    print(f"Steps to F1=1.0: {result['steps_to_f1_1']} / {cfg.max_steps}")
    print(f"Brute-force template count: {result['brute_force_template_count']}")
    print(f"Accepted F1: {result['accepted_f1']:.3f}")
    print(f"Gold covered rate:  {result['gold_covered_rate']:.0%}")
    print(f"Proposer pruning:   {result['proposer_pruning_rate']:.0%}")
    print(f"Failure mode:       {result['failure_mode']}")
    if result['steps_to_f1_1'] is not None:
        ratio = result['steps_to_f1_1'] / result['brute_force_template_count']
        verdict = "PASS" if ratio < 1.0 else "FAIL — loop not faster than brute force"
        print(f"Steps/Templates ratio: {ratio:.2f}  → {verdict}")
    else:
        print("F1=1.0 not reached within max_steps → FAIL")


if __name__ == "__main__":
    main()
