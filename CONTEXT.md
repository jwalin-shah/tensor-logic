# Tensor Logic Context

This file is the first stop for agents trying to understand the repo. Keep it
short, factual, and current; longer audit reports belong under `docs/`.

## Repo Identity

`tensor-logic` has two roles:

- A research notebook for tensor-logic experiments, demos, and memos.
- A reusable Python package, `tensor_logic`, consumed by other workspace
  projects.

Prefer the package role when making maintainability decisions. New experiments
should import `tensor_logic` instead of copying logic from older `exp*.py`
files.

## Domain Terms

- **Tensor logic**: a rule language where joins, reductions, and recurrences are
  represented as tensor operations.
- **Program**: the mutable in-memory knowledge base in `tensor_logic.program`.
  It owns domains, relations, facts, rules, and query execution.
- **Domain**: a named set of entities used to shape relations.
- **Relation**: a named tensor-backed predicate over one or more domains.
- **Fact**: an asserted relation tuple, optionally with source metadata.
- **Rule**: the canonical internal rule shape, composed of a head atom and body
  atoms.
- **Atom**: a predicate plus arguments, with optional negation metadata.
- **Proof**: the explanation object produced by the proof engine for why a query
  is true or false.
- **TL file**: a `.tl` source file parsed by `tensor_logic.file_format`.
- **Repo graph**: a tensor-logic fact model of Python import dependencies.

## Package Map

- `tensor_logic/language.py`: tensor/equation substrate, domains, relations,
  expression nodes, and fixpoint evaluation.
- `tensor_logic/program.py`: primary `Program`, canonical `Atom`/`Rule`, and the
  string rule parser.
- `tensor_logic/file_format.py`: `.tl` file parsing, includes, facts, rules, and
  queued commands.
- `tensor_logic/execution.py`: shared query/prove execution used by adapters.
- `tensor_logic/proofs.py`: positive and negative proof search over `Program`
  rules and facts.
- `tensor_logic/proof_result.py` and `tensor_logic/proof_tree_viewer.py`: proof
  payload shaping and display formatting.
- `tensor_logic/ingest.py` and `tensor_logic/repo_graph_view.py`: Python import
  graph ingestion and dependency reports.
- `tensor_logic/http_api.py` and `tensor_logic/__main__.py`: outward adapters
  for HTTP and CLI usage.
- `tensor_logic/research/`: reusable research helpers that are not the core
  package interface.

## Experiment Map

- `demos/`: small runnable demonstrations of a single idea.
- `experiments/`: historical and active experiment scripts. Many scripts write
  result JSON by default, so use explicit temporary `--output` paths when
  validating unless the work order owns those artifacts.
- `phase_training/`: embodied-agent and phase-training experiments.
- `notes/`: long-form research notes, memos, and transcript material.
- `web_workbench/`: local web workbench that exercises package behavior through
  its server adapter.

## Validation Vocabulary

Use `docs/VALIDATION.md` as the source of truth for validation tiers.

- **Full suite**: `python3 -m pytest tests/ -v` locally, or the documented
  `python -m pytest tests/ -v` in CI environments where `python` exists.
- **Packaging/code-index preflight**:
  `python3 -m pytest tests/test_packaging_ci.py tests/test_code_index.py -v`.
- **Support/stability fast path**:
  `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v`.
- **CLI smoke without repo mutation**: pass explicit temp output paths to
  experiment scripts that otherwise write under tracked `experiments/*_data/`
  directories.

## Maintainer Notes For Agents

- Before planning code changes in `tensor_logic/`, run
  `python3 tools/code_index.py --lookup <RelevantSymbol>` as instructed by
  `CLAUDE.md`.
- Treat `Program`, `language.py`, and the canonical `Atom`/`Rule` types as the
  deepest package interfaces.
- Be cautious around binary-arity assumptions. The canonical `Atom.args` shape
  is variadic, while several current proof, execution, and rule-adapter paths
  still consume binary relations.
- Do not manually edit `AGENTS.md`; it is memjuice-managed.
- `tools/index.json`, `.cocoindex_code/`, and `.pytest_cache/` are local ignored
  state, not source artifacts.
