from __future__ import annotations

from typing import Any

from .proofs import NegativeProof, Proof, fmt_negative_proof_tree, fmt_proof_tree, prove, prove_negative


def format_proof_result(
    *,
    proof: Proof | None = None,
    negative_proof: NegativeProof | None = None,
    format_type: str = "tree",
) -> dict[str, Any]:
    if format_type not in {"tree", "json"}:
        raise ValueError("format_type must be 'tree' or 'json'")
    if negative_proof is not None:
        if format_type == "tree":
            return {"answer": False, "proof": fmt_negative_proof_tree(negative_proof)}
        return _negative_proof_to_json(negative_proof)
    if proof is None:
        return {"answer": False, "proof": None}
    if format_type == "tree":
        return {"answer": True, "proof": fmt_proof_tree(proof)}
    return {"answer": True, "proof": _proof_to_json(proof)}


def prove_binary_relation_result(
    program: Any,
    relation: str,
    arg0: str,
    arg1: str,
    *,
    recursive: bool = False,
    why_not: bool = False,
    format_type: str = "tree",
) -> dict[str, Any]:
    """Run and format proof results for binary proof/query adapter boundaries."""
    proof = prove(program, relation, arg0, arg1, recursive=recursive)
    if proof is not None:
        return format_proof_result(proof=proof, format_type=format_type)
    if not why_not:
        return {"answer": False, "proof": None}
    neg_proof = prove_negative(program, relation, arg0, arg1, recursive=recursive)
    if neg_proof is None:
        return {"answer": True}
    return format_proof_result(negative_proof=neg_proof, format_type=format_type)


def _proof_to_json(proof: Proof) -> dict[str, Any]:
    rel, src, dst = proof.head
    result = {
        "head": [rel, src, dst],
        "confidence": proof.confidence,
        "body": [_proof_to_json(child) for child in proof.body],
    }
    if proof.source is not None:
        result["source"] = {"file": proof.source.file, "lineno": proof.source.lineno}
    return result


def _negative_proof_to_json(neg_proof: NegativeProof) -> dict[str, Any]:
    return {"answer": False, "explanation": _negative_proof_explanation_to_json(neg_proof)}


def _negative_proof_explanation_to_json(neg_proof: NegativeProof) -> dict[str, Any]:
    rel, src, dst = neg_proof.head
    return {
        "head": [rel, src, dst],
        "reason": neg_proof.reason,
        "body": [_negative_proof_explanation_to_json(child) for child in neg_proof.body],
    }
