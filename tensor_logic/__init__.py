"""Reusable tensor-logic substrate pieces."""

from .closure import bfs_query, bfs_per_source_closure, dense_closure
from .language import Domain, Relation, evaluate_expr, facts
from .program import Program, FactSource
from .file_format import load_tl
from .http_api import ingest_python_source, prove_source, query_source, run_source, serve
from .ingest import PythonImportGraph, ingest_python, render_python_imports_tl
from .proofs import fmt_proof_tree, fmt_negative_proof_tree, prove, prove_negative, prove_with_do, Proof, NegativeProof
from .proof_tree_viewer import ProofTreeNode, build_proof_tree_view, render_proof_tree
from .provenance import evaluate_with_provenance, fmt_proof, proof_score
from .repo_graph_view import RepoGraphData, dependency_report, load_repo_graph
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
    "FactSource",
    "PythonImportGraph",
    "ProofTreeNode",
    "RepoGraphData",
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
    "ingest_python",
    "ingest_python_source",
    "render_python_imports_tl",
    "run_source",
    "query_source",
    "prove_source",
    "serve",
    "build_proof_tree_view",
    "render_proof_tree",
    "load_repo_graph",
    "dependency_report",
    "prove",
    "prove_negative",
    "prove_with_do",
    "fmt_proof",
    "parse_rule",
    "proof_score",
    "query_relation",
]
