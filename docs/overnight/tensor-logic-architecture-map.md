# tensor-logic architecture-map audit

Queue item: `tensor-logic-architecture-map`
Branch: `codex/goal-tensor-logic-architecture-map`
Repo path: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-logic-architecture-map`
Audit time observed locally: `2026-05-06 23:53:02 PDT`
Initial HEAD: `c9e2cfce206bb279a4348189726908fcb436d3dc`

## Scope and decisions

- Wrote exactly one report file: `docs/overnight/tensor-logic-architecture-map.md`.
- Did not edit product code, generated data, secrets, external services, deploys, pushes, PRs, or trackers.
- Used the architecture review vocabulary for module/interface/seam/adapter/depth/locality, but did not enter the interactive refactor loop because this queue item requires a standalone read-only report.
- Used `python3` for code-index and test commands because `python tools/code_index.py --dump` failed locally with `python: command not found`. The repo docs still show `python -m ...` commands in places, so local runners should either provide a `python` shim or use `python3` explicitly.

## Commands run

- `pwd && (command -v llm-tldr >/dev/null 2>&1 && llm-tldr tree . || ...)`
  - Result: captured repo structure. Main directories are `tensor_logic/`, `experiments/`, `demos/`, `tests/`, `docs/`, `notes/`, `phase_training/`, `web_workbench/`, `tools/`, and `examples/`.
- `git status --short && git branch --show-current && git rev-parse HEAD`
  - Result: initial worktree was clean; branch was `codex/goal-tensor-logic-architecture-map`; HEAD was `c9e2cfce206bb279a4348189726908fcb436d3dc`.
- `rg --files -g 'AGENTS.md' -g 'CLAUDE.md' -g 'README*' -g 'package.json' -g 'pyproject.toml' -g 'Cargo.toml' -g 'go.mod' -g 'Makefile' -g 'justfile' -g 'docs/**'`
  - Result: found `CLAUDE.md`, `README.md`, `pyproject.toml`, domain docs, run protocols, and superpowers plans/specs.
- `python tools/code_index.py --dump`
  - Result: failed with `python: command not found`.
- `python3 tools/code_index.py --dump`
  - Result: indexed the reusable API, including `Program`, `RuleParser`, `prove`, `prove_negative`, `execute_command`, `load_tl`, HTTP helpers, repo graph helpers, and rule adapters.
- `python3 tools/code_index.py --lookup Program && ... --lookup RuleParser && ... --lookup prove && ... --lookup execute_command && ... --lookup parse_rule`
  - Result: confirmed current signatures before writing next-work suggestions against `tensor_logic/`.
- `rg -n "^(class|def|@dataclass|from |import )" tensor_logic tests/...`
  - Result: mapped public classes/functions and the tests exercising core seams.
- `wc -l tensor_logic/*.py tensor_logic/research/*.py tests/*.py experiments/exp8*.py experiments/exp9*.py web_workbench/server.py web_workbench/static/app.js`
  - Result: reusable package is about 2.9k lines; `tests/test_tensor_logic_core.py` is the broadest core test at 939 lines; late support experiments are larger and mostly outside the reusable package.
- `python3 -m tensor_logic ingest-python tensor_logic | sed -n '1,160p'`
  - Result: generated a local import graph for `tensor_logic/` with 22 module symbols and direct imports such as `__main__ -> execution`, `http_api -> execution`, `repo_graph_view -> proofs`, `rules -> program`, and `program -> language`.
- `python3 -m pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_program_string_rules tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_cli_and_http_share_negative_proof_json_semantics -q`
  - Result: `2 passed in 2.50s`.
- Queue validation command: `git status --short`
  - Result after writing this report: exit 0, output `?? docs/overnight/`.

## Dirty state

- Initial dirty state: clean.
- Final dirty state before any commit: one untracked report under `docs/overnight/`.
- No product code files were modified.

## Architecture map

The repository has two identities that now share one tree:

- Research notebook/history repo: `README.md:5` and `README.md:7` frame the repo as a learning/research project; `experiments/`, `demos/`, `notes/`, and `phase_training/` preserve the exploration arc.
- Reusable library dependency: `README.md:3` says this repo now serves as the shared `tensor_logic` library dependency for other workspace projects, and `README.md:73` says new experiments should import the package rather than copying older logic.

### Main entrypoints

- Python package import surface: `tensor_logic/__init__.py:3-20` imports most reusable modules, and `tensor_logic/__init__.py:23-66` exports a broad API. This includes core substrate classes, CLI/execution helpers, HTTP helpers, proof helpers, repo graph helpers, and legacy `tl_rule` parsing/evaluation.
- CLI: `tensor_logic/__main__.py:13-49` defines `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`. `tensor_logic/__main__.py:77-90` routes loaded `.tl` programs into shared command execution.
- HTTP API: `tensor_logic/http_api.py:20-57` exposes source-level helper functions; `tensor_logic/http_api.py:64-99` maps POST paths to `/ingest-python`, `/run`, `/query`, and `/prove`.
- Web workbench: `web_workbench/server.py:44-83` writes a temporary `.tl` file and invokes `sys.executable -m tensor_logic` as a subprocess instead of calling `tensor_logic.execution` directly.
- Memjuice reason subprocess: `tensor_logic/reason.py:1-8` documents a `python -m tensor_logic.reason` entrypoint; `tensor_logic/reason.py:239-293` runs the optimize loop and prints JSON.

### Core modules and ownership

- Tensor/equation substrate: `tensor_logic/language.py` owns `Domain`, `Relation`, expression dataclasses, relation data tensors, expression evaluation, boolean fixpoints, and arbitrary-arity relation storage. `tensor_logic/closure.py` owns graph closure helpers, and `tensor_logic/semirings.py` owns small matrix semiring helpers.
- Program and file format: `tensor_logic/program.py:43-102` owns mutable `Program` state for domains, relations, rules, and fact sources. `tensor_logic/program.py:85-90` parses a rule once into an eager tensor equation and again into canonical proof rules. `tensor_logic/file_format.py` owns `.tl` parsing, includes, facts, rules, and queued query/prove commands.
- Canonical rule shape: `docs/superpowers/specs/2026-05-04-rule-language-shape.md:11-27` defines `tensor_logic.program.Atom` and `Rule` as the canonical internal representation. `tensor_logic/program.py:12-34` matches that shape.
- Rule adapters and evaluators: `tensor_logic/rules.py:13-26` parses XML-like `<tl_rule>` tags into canonical `program.Atom`/`program.Rule`. `tensor_logic/rules.py:93-155` evaluates binary graph-dict rules with PyTorch tensors. This is an adapter/evaluator surface, not the owner of the durable rule language.
- Proof engine: `tensor_logic/proofs.py:44-76` handles positive proofs with tabling and an optional recursive fallback; `tensor_logic/proofs.py:78-117` handles negative proofs. Formatting and JSON payload shaping are split into `tensor_logic/proof_result.py` and `tensor_logic/proof_tree_viewer.py`.
- Repo import graph: `tensor_logic/ingest.py` turns Python imports into `.tl` source; `tensor_logic/repo_graph_view.py:17-32` loads repo graph facts and `tensor_logic/repo_graph_view.py:92-114` builds dependency reports with query/proof output.
- Search/reason loop: `tensor_logic/optimize.py` owns the general Pareto frontier loop; `tensor_logic/reason.py:49-89` maps observations into `.tl` domains/relations and `tensor_logic/reason.py:185-217` evaluates candidate TL rules.
- Research support modules: `tensor_logic/research/` contains slot-attention and rule-induction utilities used by later experiments. Larger late experiments such as `experiments/exp85_support_tl.py` are still experiment-local modules.

### Import graph evidence

`python3 -m tensor_logic ingest-python tensor_logic` found these important internal edges:

- CLI and adapters point inward: `__main__ -> execution`, `__main__ -> file_format`, `__main__ -> http_api`, `__main__ -> ingest`, `__main__ -> repo_graph_view`.
- HTTP points to execution, file format, and ingestion: `http_api -> execution`, `http_api -> file_format`, `http_api -> ingest`.
- Program points to language: `program -> language`.
- Rules and proofs consume canonical program shapes: `rules -> program`, `proofs -> program`.
- Proof result and repo graph consume proof engine: `proof_result -> proofs`, `repo_graph_view -> proofs`.
- Research constants depend on research utils: `research_constants -> research_utils`.

This is a mostly acyclic package with `program.py`/`language.py` as the deepest reusable substrate and outward adapters around it.

## Module depth and friction

### Strong seams

- `Program` is a useful deep module: callers get domain/relation/fact/rule/query/fixpoint behavior through a compact interface, while tensor data, sources, expressions, and proof rules stay local to `program.py` and `language.py`.
- `execution.py` is a useful adapter seam between CLI/HTTP and the proof/query implementation. `tensor_logic/execution.py:38-83` centralizes query/prove command behavior and keeps formatting in one place.
- `ingest.py` plus `repo_graph_view.py` forms a coherent import-graph slice: one module renders `.tl`; the other loads and explains dependency facts.

### Shallow or leaking seams

- `__init__.py` is broad enough to blur ownership. Importing `tensor_logic` imports HTTP server helpers, repo graph helpers, legacy graph-dict provenance, command execution, and core language classes all at once (`tensor_logic/__init__.py:3-20`). This is convenient but weakens locality for future dependency-sensitive users.
- `web_workbench/server.py` shells out to the CLI (`web_workbench/server.py:57-71`) while `http_api.py` and `execution.py` already provide in-process adapter surfaces. That choice may be intentional for parity with the CLI, but it means the workbench has a different error and performance path than the HTTP API.
- `provenance.py` is a parallel proof-tree implementation over graph dictionaries (`tensor_logic/provenance.py`) while `proofs.py` is the `Program`-based proof engine. This preserves older demos, but future proof UI work has two proof representations to remember.

## Stale assumptions and boundary risks

1. Binary arity is explicit but spread across several modules.
   - The canonical spec says `Atom.args` is variadic and warns not to bake binary-only fields into the shared language (`docs/superpowers/specs/2026-05-04-rule-language-shape.md:94-110`).
   - Current code still has binary consumer boundaries in `execution.py:38-52`, `execution.py:64-112`, `rules.py:60-63`, `rules.py:93-155`, and `proofs.py:249-305`.
   - `program.Atom.left/right` properties at `tensor_logic/program.py:22-28` assume at least two args without guardrails. That is compatible with current binary consumers but is a sharp edge if arbitrary arity expands.

2. The rule parser has a two-output, two-pass interface.
   - The approved spec says `Program.rule()` / `RuleParser` should parse one source rule into both the tensor expression and one or more canonical `Rule` values (`docs/superpowers/specs/2026-05-04-rule-language-shape.md:42-47`).
   - Current `Program.rule()` calls `RuleParser(self).parse_rule(text)` and then creates a fresh `RuleParser(self).parse_rule_ast_list(text)` (`tensor_logic/program.py:85-90`). This keeps correctness local today, but it makes expression and proof AST behavior easier to drift.
   - TL program syntax does not currently expose a negation spelling; `RuleParser._parse_factor_atoms` records positive atoms and skips expression methods (`tensor_logic/program.py:210-227`). Negation is only visible in the XML-like `<tl_rule>` adapter (`rules.py:43-51`) even though `program.Atom.negated` is shared.

3. Recursive relation proof still has a heuristic fallback.
   - `prove()` first tries tabled rule proofs, but if no proof is found and `recursive=True`, it calls `_prove_recursive_chain` (`tensor_logic/proofs.py:67-76`).
   - `_find_base_relation` picks the first other binary relation with matching domains (`tensor_logic/proofs.py:372-378`). This works for simple `imports -> depends_on` / `edge -> path` shapes, but it is not a durable interface if a program has multiple same-domain base relations.

4. Support/stability is called a TL engine but is currently experiment-local.
   - The plan calls for a "TL rule engine over primitive relations" (`docs/superpowers/plans/2026-05-05-support-stability.md:70-78`) and a "TL Stability Engine" (`docs/superpowers/plans/2026-05-05-support-stability.md:217-229`).
   - The implemented experiment starts as `experiments/exp85_support_tl.py` and defines its own `ProofTrace`, `PrimitiveRelations`, `StabilityResult`, and fixpoint logic (`experiments/exp85_support_tl.py:1-67`) rather than using `Program`, `Rule`, or `Proof`.
   - This may be the right local experiment interface, but it should be called out before promoting it as reusable `tensor_logic` substrate.

5. Historical plans are useful context but not always current source of truth.
   - Example: `docs/superpowers/plans/2026-04-28-optimize-loop.md` describes desired `tensor_logic.reason` work and mentions APIs such as `Program.from_source`; current code instead builds `.tl` source and loads it through temp files (`tensor_logic/reason.py:196-215` and `_program_from_tl_source` around `tensor_logic/reason.py:126-137`).
   - Morning agents should treat `docs/superpowers/specs/` decisions as higher-signal than older implementation plans unless the code contradicts them.

## Missing validation

- Required queue validation is only `git status --short`; it is runnable locally.
- Required queue validation ran successfully after this report was written: `git status --short` exited 0 and reported `?? docs/overnight/`.
- Full worker validation from `README.md:85-93` (`python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`) was not run because this queue item requested a read-only architecture report and validation command is `git status --short`.
- I ran two focused proof/CLI parity tests as audit evidence: `python3 -m pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_program_string_rules tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_cli_and_http_share_negative_proof_json_semantics -q` passed with `2 passed in 2.50s`.
- Local environment note: bare `python` is missing; `python3` works for code-index and focused pytest commands.

## Next safe work

1. Make the binary arity boundary mechanically explicit.
   - Files likely touched: `tensor_logic/program.py`, `tensor_logic/proofs.py`, `tensor_logic/rules.py`, `tensor_logic/execution.py`, `tests/test_tensor_logic_core.py`.
   - Acceptance criteria: binary consumers reject non-binary atoms/commands with consistent errors; `Atom.left/right` either guard arity or become adapter-local helpers; canonical `Atom.args` remains variadic.
   - Validation: `python3 -m pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_cli_invalid_arity_and_parse_errors_are_external_failures tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_tl_rule_evaluate_reports_unknown_relation -q`.

2. Decide whether support/stability stays experiment-local or becomes reusable package surface.
   - Files likely touched for a planning-only slice: `docs/superpowers/plans/2026-05-05-support-stability.md` or a new ADR/context doc. Files likely touched for implementation: `experiments/exp85_support_tl.py`, `tensor_logic/research/` or a new `tensor_logic/support.py`, and `tests/test_exp85_support_tl.py`.
   - Acceptance criteria: docs state whether `exp85` is an isolated experiment adapter or the intended reusable TL support module; if promoted, public interfaces and proof format align with `Program`/`Proof` or explicitly justify a separate interface.
   - Validation: planning-only `git diff --check`; implementation `python3 -m pytest tests/test_exp85_support_tl.py -q`.

3. Unify web workbench execution with the existing in-process execution seam, or document the subprocess choice.
   - Files likely touched: `web_workbench/server.py`, `tests/test_web_workbench.py`, possibly `web_workbench/README.md`.
   - Acceptance criteria: workbench either calls `tensor_logic.execution`/`http_api` directly, or README/tests explicitly assert CLI-subprocess parity as the intended contract.
   - Validation: `python3 -m pytest tests/test_web_workbench.py -q`.

## Handoff

- Changed files: `docs/overnight/tensor-logic-architecture-map.md`.
- Commit SHA at audit start: `c9e2cfce206bb279a4348189726908fcb436d3dc`.
- Validation result: `git status --short` exited 0 and reported `?? docs/overnight/`.
- PR URL: none; PR creation is out of scope for this queue item.
- Blockers: none for writing the report. Environment caveat: `python` command is absent, so commands documented with `python ...` need `python3 ...` or a shim in this local worker.
