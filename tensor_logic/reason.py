"""
Entry point for the `memjuice reason` subprocess call.

Usage:
    python -m tensor_logic.reason --query "..." --facts-file /path/facts.jsonl [--max-steps 20]

Stdout: JSON {"obs_ids": [int, ...], "proofs": [str, ...], "query": str}
Stderr: progress/debug info
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# EAV field mapping: observation dict key → TL relation name
_FIELD_MAP = {
    "kind":    "obs_kind",
    "project": "obs_project",
    "file":    "obs_file",
    "ts":      "obs_ts",
    "text":    "obs_text",
    "harness": "obs_harness",
    "sha":     "obs_sha",
}


def obs_to_facts(obs: dict[str, Any]) -> str:
    """Convert a single observation dict to TL EAV fact lines."""
    obs_id = obs["id"]
    lines = []
    for key, rel in _FIELD_MAP.items():
        if key in obs:
            val = str(obs[key]).replace('"', '\\"')
            lines.append(f'{rel}({obs_id}, "{val}").')
    return "\n".join(lines)


def facts_to_tl_source(observations: list[dict[str, Any]]) -> str:
    """Convert list of observations to full TL source with EAV facts."""
    sections = [obs_to_facts(obs) for obs in observations]
    return "\n".join(sections)


def _extract_obs_ids(tl_source: str) -> list[int]:
    """Extract all observation IDs mentioned in TL source."""
    return sorted({int(m) for m in re.findall(r"obs_\w+\((\d+),", tl_source)})


def make_query_evaluator(observations: list[dict[str, Any]]):
    """
    Returns an evaluate(query_str: str) -> EvalResult closure.

    Appends the proposed 'result(X) :- ...' rule to the EAV facts,
    runs it through the TL engine, collects matching obs IDs,
    and scores by coverage (fraction of all obs IDs matched).
    """
    from tensor_logic.language import parse_source
    from tensor_logic.program import Program
    from tensor_logic.proofs import prove, prove_negative, fmt_proof_tree, fmt_negative_proof_tree
    from tensor_logic.optimize import EvalResult

    facts_source = facts_to_tl_source(observations)
    all_ids = _extract_obs_ids(facts_source)
    total = len(all_ids)

    def evaluate(query: str) -> EvalResult:
        full_source = facts_source + "\n" + query
        try:
            program = Program.from_source(full_source)
        except Exception as exc:
            return EvalResult(
                artifact=json.dumps({"obs_ids": [], "query": query, "proofs": []}),
                score=0.0,
                asi=f"parse error: {exc}",
                asi_kind="engine_error",
            )

        matched_ids = []
        proof_texts = []
        for obs_id in all_ids:
            try:
                p = prove(program, "result", str(obs_id), "_")
                if p is not None:
                    matched_ids.append(obs_id)
                    proof_texts.append(fmt_proof_tree(p))
            except Exception:
                pass

        why_not_texts = []
        unmatched = [i for i in all_ids if i not in matched_ids]
        for obs_id in unmatched[:3]:
            try:
                np_ = prove_negative(program, "result", str(obs_id), "_")
                if np_ is not None:
                    why_not_texts.append(fmt_negative_proof_tree(np_))
            except Exception:
                pass

        score = len(matched_ids) / total if total > 0 else 0.0
        asi_parts = []
        if why_not_texts:
            asi_parts.append("Unmatched observations:\n" + "\n".join(why_not_texts[:3]))
        asi = "\n".join(asi_parts) if asi_parts else f"matched {len(matched_ids)}/{total}"

        return EvalResult(
            artifact=json.dumps({"obs_ids": matched_ids, "query": query, "proofs": proof_texts}),
            score=score,
            secondary_score=score,
            asi=asi,
            asi_kind="why_not" if why_not_texts else "proof",
        )

    return evaluate


def _make_reason_proposer(user_query: str, all_obs_ids: list[int],
                          model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Returns a propose(feedback: str) -> datalog_query_str closure."""
    field_names = list(_FIELD_MAP.values())

    def propose(feedback: str) -> str:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch as _torch
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            prompt = (
                f"Available TL relations: {', '.join(field_names)}\n"
                f"User query: {user_query}\n"
                f"Previous failure: {feedback[:400] if feedback else 'none'}\n"
                "Write a single Datalog rule: result(X) :- ... using the available relations.\n"
                "Rule: "
            )
            inputs = tok(prompt, return_tensors="pt")
            with _torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            rule = generated.strip().split("\n")[0]
            if not rule.endswith("."):
                rule += "."
            return rule
        except Exception:
            words = user_query.lower().split()
            for kind in ("decision", "fix", "edit", "commit"):
                if kind in words:
                    return f'result(X) :- obs_kind(X, "{kind}").'
            return 'result(X) :- obs_kind(X, "decision").'

    return propose


def reason(
    observations: list[dict[str, Any]],
    user_query: str,
    max_steps: int = 20,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> tuple[list[int], list[str], str]:
    """
    Run the optimize loop to find TL query matching user_query over observations.

    Returns: (matched_obs_ids, proof_texts, best_query_str)
    """
    from tensor_logic.optimize import EvalResult, optimize

    all_ids = _extract_obs_ids(facts_to_tl_source(observations))
    propose = _make_reason_proposer(user_query, all_ids, model_name)
    evaluate = make_query_evaluator(observations)

    def accept(r: EvalResult) -> bool:
        return r.score > 0 and r.asi_kind != "engine_error"

    frontier = optimize(
        propose, evaluate, accept,
        pareto_axes=("coverage", "coverage"),
        max_steps=max_steps,
        frontier_size=5,
        stagnation_k=3,
    )

    if not frontier or frontier[0].score == 0.0:
        return [], [], ""

    best = frontier[0]
    payload = json.loads(best.artifact)
    return payload["obs_ids"], payload["proofs"], payload["query"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="tensor_logic reason subprocess entry point")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--facts-file", required=True, help="Path to JSONL observations file")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    facts_path = Path(args.facts_file)
    observations = [json.loads(line) for line in facts_path.read_text().splitlines() if line.strip()]
    for i, obs in enumerate(observations):
        obs["id"] = i

    obs_ids, proofs, query = reason(
        observations=observations,
        user_query=args.query,
        max_steps=args.max_steps,
        model_name=args.model,
    )
    print(json.dumps({"obs_ids": obs_ids, "proofs": proofs, "query": query}))


if __name__ == "__main__":
    main()
