"""Supported root import surface for Tensor Logic.

The package root intentionally mirrors ``tensor_logic.core``. Adapter,
research, repo-graph, and legacy graph-dict helpers remain importable from
their owning modules. A small lazy compatibility layer below keeps older
``from tensor_logic import ...`` callers working without treating those names
as supported public exports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .core import *  # noqa: F403
from .core import __all__ as _CORE_ALL

__all__ = list(_CORE_ALL)

_INTERNAL_COMPAT_EXPORTS = {
    "Atom": ("tensor_logic.program", "Atom"),
    "Rule": ("tensor_logic.program", "Rule"),
    "PythonImportGraph": ("tensor_logic.ingest", "PythonImportGraph"),
    "ProofTreeNode": ("tensor_logic.proof_tree_viewer", "ProofTreeNode"),
    "RepoGraphData": ("tensor_logic.repo_graph_view", "RepoGraphData"),
    "evaluate_rule": ("tensor_logic.rules", "evaluate_rule"),
    "evaluate_with_provenance": ("tensor_logic.provenance", "evaluate_with_provenance"),
    "execute_command": ("tensor_logic.execution", "execute_command"),
    "fmt_proof_tree": ("tensor_logic.proofs", "fmt_proof_tree"),
    "fmt_negative_proof_tree": ("tensor_logic.proofs", "fmt_negative_proof_tree"),
    "load_tl": ("tensor_logic.file_format", "load_tl"),
    "ingest_python": ("tensor_logic.ingest", "ingest_python"),
    "ingest_python_source": ("tensor_logic.http_api", "ingest_python_source"),
    "render_python_imports_tl": ("tensor_logic.ingest", "render_python_imports_tl"),
    "run_source": ("tensor_logic.http_api", "run_source"),
    "query_source": ("tensor_logic.http_api", "query_source"),
    "prove_source": ("tensor_logic.http_api", "prove_source"),
    "serve": ("tensor_logic.http_api", "serve"),
    "build_proof_tree_view": ("tensor_logic.proof_tree_viewer", "build_proof_tree_view"),
    "render_proof_tree": ("tensor_logic.proof_tree_viewer", "render_proof_tree"),
    "load_repo_graph": ("tensor_logic.repo_graph_view", "load_repo_graph"),
    "dependency_report": ("tensor_logic.repo_graph_view", "dependency_report"),
    "fmt_proof": ("tensor_logic.provenance", "fmt_proof"),
    "parse_rule": ("tensor_logic.rules", "parse_rule"),
    "proof_score": ("tensor_logic.provenance", "proof_score"),
    "query_relation": ("tensor_logic.rules", "query_relation"),
}


def __getattr__(name: str) -> Any:
    target = _INTERNAL_COMPAT_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
