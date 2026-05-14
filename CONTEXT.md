# Tensor Logic Context

This repository is both a research workspace and a reusable `tensor_logic`
package. Treat checked-in experiments as evidence with provenance
requirements, not as license to make broader claims.

## Repo Identity

`tensor-logic` has two roles:

- A research notebook for tensor-logic experiments, demos, and memos.
- A reusable Python package, `tensor_logic`, consumed by other workspace
  projects.

Prefer the package role when making maintainability decisions. New experiments
should import `tensor_logic` instead of copying logic from older `exp*.py`
files.

## Domain Vocabulary

- **Tensor Logic (TL)**: the named-index tensor/einsum rule substrate in
  `tensor_logic/` and the experiment scripts that probe its limits.
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
- **Experiment**: a numbered `experiments/expN_*.py` script with an explicit
  hypothesis, local or remote run command, and recorded result row in
  `notes/EXPERIMENTS.md`.
- **Result artifact**: a committed JSON/plot/manifest under an
  `experiments/*_data/` directory. A result artifact is evidence only for the
  exact script, inputs, command, dependency set, and commit that produced it.
- **Claim**: any README, note, PR, issue, or comment statement that generalizes
  from experiment results.
- **No-overclaim rule**: every claim must state its evidence tier, scope, and
  caveats. Do not promote a toy, simulated, oracle, synthetic, internally
  validated, or remote-only result into a general capability claim.

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
- `notes/`: long-form research notes, memos, transcript material.
- `web_workbench/`: local web workbench that exercises package behavior through
  its server adapter.

## Current Provenance Rules

- Use `docs/EXPERIMENT_PROVENANCE.md` before changing README claims,
  experiment-status notes, result tables, or benchmark language.
- Use `docs/VALIDATION.md` before selecting a validation command.
- Use `docs/RUN_PROTOCOL.md` for remote/Kaggle/model-training runs and for any
  run that can create durable experiment artifacts.
- Keep `notes/EXPERIMENTS.md` append-only in spirit: failed, mixed, superseded,
  and invalid experiments are evidence and should remain visible.

## Claim Boundaries

- `exp87` through `exp95` are support/stability research slices. They establish
  specific toy/simulated object-table and pixel-facing interface behavior, not
  real-world physics, production perception, or end-to-end vision.
- `exp94` is an oracle/simulated upper-bound structural recovery result. Cite it
  as an upper bound unless the claim also references non-oracle follow-up.
- `exp95` is non-oracle and mixed: it improves false-positive ranking but still
  exposes missing-object and merge identity/cardinality gaps.
- `exp78`, `exp79`, and `exp83` have committed result artifacts, but their
  status and gates must be read from the result artifact plus
  `notes/EXPERIMENTS.md`, not inferred from old planning docs.
- `exp53` and `exp54` headline import-graph numbers depend on external package
  source downloads unless frozen manifests are added. Do not present them as
  locally reproducible from checked-in artifacts alone.

## Validation Vocabulary

Use `docs/VALIDATION.md` as the source of truth for validation tiers.

- **Local handoff gate**: `python3 tools/local_validation.py`, which runs the
  full suite and `git diff --check`.
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
- `AGENTS.md` is memjuice-managed per `CLAUDE.md`; do not hand-edit it when a
  repo-local context or validation rule can live in `CONTEXT.md`,
  `docs/VALIDATION.md`, or `docs/EXPERIMENT_PROVENANCE.md`.
- `tools/index.json`, `.cocoindex_code/`, and `.pytest_cache/` are local ignored
  state, not source artifacts.
