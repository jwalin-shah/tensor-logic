# Tensor Logic Public API

`tensor_logic` now treats the package root as the supported import surface for
the reusable semantics layer. It mirrors `tensor_logic.core` and intentionally
excludes local adapters, repo-graph helpers, legacy graph-dict evaluators, and
research utilities.

## Supported root exports

These names are supported from both `tensor_logic` and `tensor_logic.core`:

- Data model: `Domain`, `Relation`, `Program`, `FactSource`.
- Proof model: `Proof`, `NegativeProof`.
- Proof functions: `prove`, `prove_negative`, `prove_with_do`,
  `prove_binary_relation_result`, `format_proof_result`.
- Tensor/relation helpers: `bfs_query`, `bfs_per_source_closure`,
  `dense_closure`, `evaluate_expr`, `facts`.

`tensor_logic.__all__` is the contract for wildcard imports and public root
exports.

## Module-scoped helpers

The following helpers are useful, but they are not part of the supported root
API. Import them from their owning modules when a caller really needs them:

- Rule representation helpers: `tensor_logic.program.Atom`,
  `tensor_logic.program.Rule`.
- File, CLI, and HTTP adapters: `tensor_logic.file_format.load_tl`,
  `tensor_logic.execution.execute_command`,
  `tensor_logic.http_api.ingest_python_source`,
  `tensor_logic.http_api.prove_source`, `tensor_logic.http_api.query_source`,
  `tensor_logic.http_api.run_source`, `tensor_logic.http_api.serve`.
- Python import-graph adapters: `tensor_logic.ingest.PythonImportGraph`,
  `tensor_logic.ingest.ingest_python`,
  `tensor_logic.ingest.render_python_imports_tl`,
  `tensor_logic.repo_graph_view.RepoGraphData`,
  `tensor_logic.repo_graph_view.dependency_report`,
  `tensor_logic.repo_graph_view.load_repo_graph`.
- Legacy graph-dict rule/provenance helpers:
  `tensor_logic.rules.parse_rule`, `tensor_logic.rules.evaluate_rule`,
  `tensor_logic.rules.query_relation`,
  `tensor_logic.provenance.evaluate_with_provenance`,
  `tensor_logic.provenance.fmt_proof`,
  `tensor_logic.provenance.proof_score`.
- Proof tree rendering helpers: `tensor_logic.proofs.fmt_proof_tree`,
  `tensor_logic.proofs.fmt_negative_proof_tree`,
  `tensor_logic.proof_tree_viewer.ProofTreeNode`,
  `tensor_logic.proof_tree_viewer.build_proof_tree_view`,
  `tensor_logic.proof_tree_viewer.render_proof_tree`.
- Research-only utilities: `tensor_logic.research`.

Some old root-level helper imports are still resolved lazily for compatibility,
but they are not listed in `__all__` and should not be expanded as public API.

## Rule and proof boundaries

`tensor_logic.program.Atom` is the canonical rule atom shape and stores
`args: tuple[str, ...]`, so the representation itself is variadic. Keep that
shape when adding future arity work.

The current proof and query boundaries are still binary:

- `prove`, `prove_negative`, `prove_with_do`, and
  `prove_binary_relation_result` operate on `relation(arg0, arg1)` goals.
- `Proof.head` and `NegativeProof.head` are triples:
  `(relation_name, arg0, arg1)`.
- CLI and HTTP query/proof adapters validate exactly two query/proof args.
- Legacy `<tl_rule ...>` parsing and graph-dict evaluation in
  `tensor_logic.rules` only support binary atom syntax and binary relation
  tensors.
- `Atom.left` and `Atom.right` are binary convenience properties. Use
  `Atom.args` at shared rule boundaries.

Expanding proofs, query adapters, or legacy graph-dict rule evaluation beyond
binary relations should be a dedicated arity slice with focused tests.
