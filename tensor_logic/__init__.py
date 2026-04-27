"""Reusable tensor-logic substrate pieces."""

from .closure import bfs_query, bfs_per_source_closure, dense_closure
from .language import Domain, Relation, evaluate_expr, facts
from .program import Program
from .file_format import load_tl
from .proofs import fmt_proof_tree, fmt_negative_proof_tree, prove, prove_negative, Proof, NegativeProof
from .provenance import evaluate_with_provenance, fmt_proof, proof_score
from .rules import (
    Atom,
    Rule,
    evaluate_rule,
    parse_rule,
    query_relation,
)

__all__ = [
    "Atom",
    "Domain",
    "Relation",
    "Rule",
    "Program",
    "Proof",
    "NegativeProof",
    "bfs_query",
    "bfs_per_source_closure",
    "dense_closure",
    "evaluate_rule",
    "evaluate_expr",
    "evaluate_with_provenance",
    "facts",
    "fmt_proof_tree",
    "fmt_negative_proof_tree",
    "load_tl",
    "prove",
    "prove_negative",
    "fmt_proof",
    "parse_rule",
    "proof_score",
    "query_relation",
]