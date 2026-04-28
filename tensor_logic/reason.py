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
import tempfile
import os
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

# Only these fields can be TL domain members (valid identifiers, finite set of values)
_CATEGORICAL_FIELDS = {
    "kind":    "obs_kind",
    "project": "obs_project",
    "harness": "obs_harness",
    "sha":     "obs_sha",
}


def _sanitize(v: str) -> str:
    """Convert any string to a valid TL domain identifier."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(v)) or "x"


def _build_tl_source(observations: list[dict[str, Any]], rule: str) -> str:
    """
    Build a valid TL program source for querying observations.

    Categorical EAV fields become TL relations over (obs_id, val) domains.
    The caller-supplied rule must define result(x, y) in terms of those relations.
    """
    obs_ids = [str(obs["id"]) for obs in observations]
    id_domain = " ".join(obs_ids)

    # Collect unique sanitized values per categorical field
    field_vals: dict[str, set[str]] = {rel: set() for rel in _CATEGORICAL_FIELDS.values()}
    for obs in observations:
        for key, rel in _CATEGORICAL_FIELDS.items():
            if key in obs:
                field_vals[rel].add(_sanitize(obs[key]))

    lines = [f"domain obs_id {{ {id_domain} }}"]

    # Declare per-field value domains and relations
    declared_rels: set[str] = set()
    for rel, vals in field_vals.items():
        if not vals:
            continue
        val_domain = " ".join(sorted(vals))
        lines.append(f"domain {rel}_val {{ {val_domain} }}")
        lines.append(f"relation {rel}(obs_id, {rel}_val)")
        declared_rels.add(rel)

    # Infer result domain from the rule (which relation it references first)
    result_val_domain = None
    for rel in _CATEGORICAL_FIELDS.values():
        if rel in rule and rel in declared_rels:
            result_val_domain = f"{rel}_val"
            break

    if result_val_domain is None:
        # Fallback: use obs_kind_val if available, else obs_id for second arg
        result_val_domain = "obs_kind_val" if "obs_kind" in declared_rels else "obs_id"

    lines.append(f"relation result(obs_id, {result_val_domain})")
    lines.append("")

    # Add facts
    for obs in observations:
        obs_id = str(obs["id"])
        for key, rel in _CATEGORICAL_FIELDS.items():
            if key in obs and rel in declared_rels:
                lines.append(f"fact {rel}({obs_id}, {_sanitize(obs[key])})")

    lines.append("")
    lines.append(f"rule {rule}")
    return "\n".join(lines)


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
    Returns an evaluate(rule: str) -> EvalResult closure.

    The rule must be a TL rule string defining result(x, y):
        result(x, y) := obs_kind(x, y).step()

    Builds a full TL program, evaluates it, and scores by coverage
    (fraction of obs IDs for which result(id, _) > 0).
    """
    from tensor_logic.file_format import load_tl
    from tensor_logic.proofs import prove, fmt_proof_tree
    from tensor_logic.optimize import EvalResult

    all_ids = [obs["id"] for obs in observations]
    total = len(all_ids)

    def evaluate(rule: str) -> EvalResult:
        try:
            tl_source = _build_tl_source(observations, rule)
        except Exception as exc:
            return EvalResult(
                artifact=json.dumps({"obs_ids": [], "query": rule, "proofs": []}),
                score=0.0,
                asi=f"source build error: {exc}",
                asi_kind="engine_error",
            )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tl', delete=False) as f:
                f.write(tl_source)
                tmp_path = f.name
            loaded = load_tl(tmp_path)
            program = loaded.program
        except Exception as exc:
            return EvalResult(
                artifact=json.dumps({"obs_ids": [], "query": rule, "proofs": []}),
                score=0.0,
                asi=f"parse error: {exc}",
                asi_kind="engine_error",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Find which obs_ids have any result(id, y) > 0
        result_tensor = program.eval("result")  # shape: [n_ids, n_vals]
        matched_ids = []
        proof_texts = []
        result_domain = program.relations["result"].domains[1]
        for i, obs_id in enumerate(all_ids):
            if result_tensor[i].sum() > 0:
                matched_ids.append(obs_id)
                # Prove one witness value for proof text
                for val in result_domain.symbols:
                    pr = prove(program, "result", str(obs_id), val)
                    if pr is not None:
                        proof_texts.append(fmt_proof_tree(pr))
                        break
                else:
                    proof_texts.append("")

        score = len(matched_ids) / total if total > 0 else 0.0
        unmatched_count = total - len(matched_ids)
        asi = (
            f"matched {len(matched_ids)}/{total}" if len(matched_ids) > 0
            else f"no matches — {unmatched_count} observations unmatched by rule: {rule}"
        )

        return EvalResult(
            artifact=json.dumps({"obs_ids": matched_ids, "query": rule, "proofs": proof_texts}),
            score=score,
            secondary_score=score,
            asi=asi,
            asi_kind="proof" if matched_ids else "why_not",
        )

    return evaluate


def _make_reason_proposer(user_query: str, all_obs_ids: list[int],
                          model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Returns a propose(feedback: str) -> TL rule string closure."""
    cat_rels = list(_CATEGORICAL_FIELDS.values())

    def propose(feedback: str) -> str:
        # Fallback: keyword match on user_query → valid TL rule
        words = user_query.lower().split()
        for kind in ("decision", "fix", "edit", "commit", "discovery"):
            if kind in words:
                return f"result(x, y) := obs_kind(x, y).step()"
        for proj in words:
            if len(proj) > 3 and proj.isalpha():
                return f"result(x, y) := obs_project(x, y).step()"
        return "result(x, y) := obs_kind(x, y).step()"

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
