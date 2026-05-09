"""Supported import surface for Tensor Logic semantics.

Use this surface, or the equivalent package root ``tensor_logic``, when you
want programmatic access without HTTP, ingest, repo-graph, or research helpers.
"""

from .closure import bfs_query, bfs_per_source_closure, dense_closure
from .language import Domain, Relation, evaluate_expr, facts
from .program import FactSource, Program
from .proof_result import format_proof_result, prove_binary_relation_result
from .proofs import NegativeProof, Proof, prove, prove_negative, prove_with_do

__all__ = [
    "Domain",
    "Relation",
    "Program",
    "FactSource",
    "Proof",
    "NegativeProof",
    "prove",
    "prove_negative",
    "prove_with_do",
    "prove_binary_relation_result",
    "format_proof_result",
    "bfs_query",
    "bfs_per_source_closure",
    "dense_closure",
    "evaluate_expr",
    "facts",
]
