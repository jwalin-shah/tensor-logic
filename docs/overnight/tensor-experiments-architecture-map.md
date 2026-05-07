# tensor-experiments architecture-map audit

Date: 2026-05-07
Queue item: `tensor-experiments-architecture-map`
Branch: `codex/goal-tensor-experiments-architecture-map`

## Scope and constraints

This audit is read-only with one allowed write: this report at
`docs/overnight/tensor-experiments-architecture-map.md`. I did not edit product
code, generated data, secrets, external services, deploys, pushes, PRs, or
external trackers.

Initial dirty state:

```text
$ git status --short
```

No output. The worktree started clean.

## Commands run

Architecture and repo mapping:

```bash
llm-tldr tree .
rtk read CLAUDE.md
rtk read README.md
rtk read pyproject.toml
rtk read docs/agents/domain.md
rg --files -g 'CONTEXT.md' -g 'CONTEXT-MAP.md' -g 'docs/adr/**'
llm-tldr search "if __name__|def main|argparse|uvicorn|FastAPI|Flask|click" .
llm-tldr search "from tensor_logic|import tensor_logic" .
python tools/code_index.py --dump
python3 tools/code_index.py --dump
rg --files tensor_logic tests demos web_workbench tools | sort
llm-tldr search "class |def " tensor_logic
python3 -m tensor_logic ingest-python tensor_logic
python3 -m tensor_logic ingest-python tensor_logic/research
python3 -m tensor_logic ingest-python web_workbench
```

Targeted source reads:

```bash
rtk read tensor_logic/__init__.py
rtk read tensor_logic/core.py
rtk read tensor_logic/program.py
rtk read tensor_logic/language.py
rtk read tensor_logic/file_format.py
rtk read tensor_logic/execution.py
rtk read tensor_logic/proofs.py
rtk read tensor_logic/provenance.py
rtk read tensor_logic/rules.py
rtk read tensor_logic/closure.py
rtk read tensor_logic/ingest.py
rtk read tensor_logic/repo_graph_view.py
rtk read tensor_logic/http_api.py
rtk read tensor_logic/optimize.py
rtk read tensor_logic/reason.py
rtk read tensor_logic/research/utils.py
rtk read tensor_logic/research/constants.py
rtk read tensor_logic/research/slot_attention.py
rtk read tools/code_index.py
rtk read web_workbench/server.py
rtk read docs/superpowers/specs/2026-05-04-rule-language-shape.md
rtk read .github/workflows/ci.yml
```

Test and usage evidence:

```bash
llm-tldr search "def test_" tests
rtk read tests/test_tensor_logic_core.py
rtk read tests/test_reason.py
rtk read tests/test_web_workbench.py
rtk read tests/test_code_index.py
rtk read tests/test_packaging_ci.py
rg "from experiments|import experiments|from phase_training|import phase_training" -n .
rg "from tensor_logic.research|import tensor_logic.research" experiments tests
rg "from tensor_logic import|from tensor_logic\\." experiments demos tests web_workbench
rg "^def |^class |if __name__ ==" experiments/exp79_lewm_tl.py experiments/exp80_fafsa_kb.py experiments/exp81_optimize_rule_induction.py experiments/exp83_slot_attention.py
rg "^def |^class |if __name__ ==" phase_training
rg "_execute_command|execute_command\\(" tensor_logic tests web_workbench experiments demos -n
rg "tensor_logic\\.core|from tensor_logic.core|import tensor_logic.core" .
rg "sys\\.path\\.insert|sys\\.path\\.append" experiments demos tests web_workbench phase_training tensor_logic -n
rg "from scipy|import scipy|transformers|mlx|sklearn|numpy|matplotlib" experiments tensor_logic tests pyproject.toml -n
rg "parse_rule\\(|RuleParser|<tl_rule|rule .*:=" -n tensor_logic demos experiments tests docs
```

Notes:

- `python tools/code_index.py --dump` failed because `python` is not available in this local shell.
- `python3 tools/code_index.py --dump` succeeded and produced the symbol dump used below.
- `rg --files docs/overnight` failed before this report was created because the directory did not exist.

## Domain docs and decisions found

`docs/agents/domain.md` says to read `CONTEXT.md`, `CONTEXT-MAP.md`, and
`docs/adr/` when present. The targeted `rg --files` command found none of those
files in this checkout, so this audit used `README.md`, `CLAUDE.md`,
`notes/EXPERIMENTS.md`, and `docs/superpowers/specs/*` as the available domain
and decision evidence.

The most directly relevant design doc is
`docs/superpowers/specs/2026-05-04-rule-language-shape.md`. It defines
`tensor_logic.program.Atom` and `tensor_logic.program.Rule` as the canonical rule
shape, while treating `<tl_rule ...>` parsing as an adapter surface.

## Architecture map

### Reusable package surface

The reusable library lives under `tensor_logic/` and is packaged by
`pyproject.toml` with `tool.setuptools.packages.find.include = ["tensor_logic*"]`.
Runtime dependency is only `torch>=2.0`; dev extras include `pytest`, `numpy`,
and `matplotlib`.

Current package root:

- `tensor_logic/__init__.py` re-exports core semantics, proof APIs, file loading,
  execution, HTTP helpers, ingest helpers, provenance helpers, proof-tree viewer,
  and repo-graph helpers.
- `tensor_logic/core.py` exists as a documented narrow import surface for
  "programs, proofs, closure" without HTTP, ingest, or repo-graph adapters.
- `rg "tensor_logic\\.core|from tensor_logic.core|import tensor_logic.core" .`
  returned no users, so the narrow surface is available but not adopted locally.

Important boundary: importing from `tensor_logic` pulls in adapter modules such
as `http_api`, `ingest`, and `repo_graph_view` through `__init__.py`. That is
convenient for demos, but it makes the package root a full-toolkit interface,
not a minimal reasoning-core interface.

### Core semantic layer

Core tensor-logic semantics are concentrated in:

- `tensor_logic/language.py`: finite `Domain`, tensor `Relation`, expression AST
  nodes, `evaluate_expr`, `Relation.eval`, and `Relation.fixpoint`.
- `tensor_logic/program.py`: mutable `Program`, canonical `Atom`, `Rule`,
  `FactSource`, and `RuleParser` for human-authored `.tl` rule syntax.
- `tensor_logic/closure.py`: dense and BFS reachability implementations.
- `tensor_logic/semirings.py`: boolean, GF(2), and reliability matmul helpers.

The core is small and readable. `Program.rule()` currently does two jobs: it
installs an executable tensor expression on the target relation and also stores
one or more canonical `Rule` values for proof search. That matches the
2026-05-04 rule-language design: expression operators remain evaluation syntax,
while proof search consumes logical atom conjunctions.

### Rule adapter and provenance layer

There are two rule syntax surfaces:

- `.tl` program syntax, parsed in `tensor_logic/program.py` and
  `tensor_logic/file_format.py`, for statements like
  `rule ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()`.
- LM/tool tag syntax, parsed in `tensor_logic/rules.py`, for
  `<tl_rule head="..." body="..."></tl_rule>`.

`tensor_logic/rules.py` now imports `Atom` and `Rule` from
`tensor_logic.program`, which is consistent with the canonical-rule design. Older
experiments still contain local copies of `<tl_rule>` parsing logic, including
`experiments/exp65_rule_chain_joins.py`, `experiments/exp66_datalog_negation.py`,
and `experiments/exp67_provenance.py`. Those are historical experiment records,
not necessarily active architecture, but they are a stale-assumption source when
future agents search examples.

`tensor_logic/provenance.py` is graph-dict based and consumes `Rule` values from
the adapter parser. `tensor_logic/proofs.py` is `Program` based and consumes
rules stored on `Program.rules`. These are related proof surfaces but not the
same execution path.

### File, execution, CLI, and HTTP adapters

Adapter layering is mostly clean:

- `tensor_logic/file_format.py` loads `.tl` files into `LoadedProgram` and
  `Command` values.
- `tensor_logic/execution.py` executes `Command` values against a `Program` and
  renders `CommandResult`.
- `tensor_logic/__main__.py` implements `python -m tensor_logic` subcommands:
  `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`.
- `tensor_logic/http_api.py` exposes in-process helpers plus a
  `ThreadingHTTPServer` handler.
- `web_workbench/server.py` serves static files and shells out to
  `python -m tensor_logic` for local workbench actions.

The cleanest ownership line is:

```text
file_format -> execution -> CLI/HTTP/workbench adapters
```

There is one small duplication: `tensor_logic/__main__.py` has a private
`_execute_command()` wrapper, and `tensor_logic/http_api.py` has another private
`_execute_command()` wrapper. `rg` found the HTTP version is not referenced by
tests or other modules. The public shared function is already
`tensor_logic.execution.execute_command()`.

### Repo-ingest and repo-graph view

`tensor_logic/ingest.py` converts local Python imports into a `.tl` dependency
program. `python3 -m tensor_logic ingest-python tensor_logic` produced a graph
with direct imports such as:

- `__main__` -> `execution`, `file_format`, `http_api`, `ingest`,
  `program`, `repo_graph_view`
- `execution` -> `file_format`, `program`, `proof_result`
- `http_api` -> `execution`, `file_format`, `ingest`
- `proof_result` -> `proofs`
- `proofs` -> `program`
- `rules` -> `program`
- `research_constants` -> `research_utils`

This import graph supports the observed layering: core modules point inward to
language/program/proofs; CLI and HTTP point outward to adapters.

`tensor_logic/repo_graph_view.py` is an adapter over `.tl` dependency programs,
not a core proof engine. It loads a file, builds adjacency, and delegates proofs
to `tensor_logic.proofs.prove()`.

### Optimization and reasoning loop

`tensor_logic/optimize.py` is a generic propose/evaluate/accept loop with an
`EvalResult` dataclass and Pareto frontier helpers. It has a tight test file
(`tests/test_optimize.py`) and no direct dependency on experiments.

`tensor_logic/reason.py` is an application of that optimize loop for EAV-style
observation facts. It also has its own subprocess entrypoint:
`python -m tensor_logic.reason --query ... --facts-file ...`. This module is
inside the reusable package, but its docstring names `memjuice reason`, which is
external to this repo. That is a real ownership crossing: the tensor package
contains an adapter intended for another workspace tool.

### Research helpers and experiments

`tensor_logic/research/` is the emerging shared research layer:

- `research/utils.py`: `Schema`, random world generation, rule enumeration,
  body application, F1, semantic equivalence, and induction helpers.
- `research/constants.py`: toy schemas and gold rules.
- `research/slot_attention.py`: `SlotAttention` and `VisualEncoder`.

Active experiments consume this layer:

- `experiments/exp78_rule_induction.py`
- `experiments/exp79_lewm_tl.py`
- `experiments/exp79_self_play_loop.py`
- `experiments/exp81_optimize_rule_induction.py`
- `experiments/exp83_slot_attention.py`
- `tests/test_exp81.py`

This is a useful extraction: repeated induction and schema utilities are no
longer only copied between scripts. The newer perception/TL experiments have not
been fully extracted, though. `experiments/exp79_lewm_tl.py` and
`experiments/exp83_slot_attention.py` duplicate scene constants, frame
generation, relation computation, TL state building, object removal, and
derived-relation evaluation. `exp83` imports `SlotAttention` and `VisualEncoder`
from `tensor_logic.research.slot_attention`, but keeps most scene/TL wiring
local.

### Script islands

The repo still has many standalone script islands:

- `experiments/`: 99 files from early one-off probes through later exp80+ work.
- `phase_training/`: embodied/object-permanence training scripts.
- `demos/`: runnable examples for the package and older learning demos.
- `web_workbench/`: local browser shell.
- `tools/code_index.py`: stdlib AST indexer for `tensor_logic/`.

The script island pattern is intentional for a learning/research repo, but it
means module ownership is not uniform. `tests/test_exp79.py`, `tests/test_exp80.py`,
and `tests/test_exp81.py` import selected experiment scripts directly, promoting
those scripts from disposable notebooks to semi-owned modules.

## Entrypoints

Primary local entrypoints:

- `python3 -m tensor_logic run <file.tl>`
- `python3 -m tensor_logic query <file.tl> <relation> <arg> <arg> [--recursive]`
- `python3 -m tensor_logic prove <file.tl> <relation> <arg> <arg> [--recursive] [--why-not] [--format tree|json]`
- `python3 -m tensor_logic ingest-python <path>`
- `python3 -m tensor_logic repl`
- `python3 -m tensor_logic repo-graph <file.tl>`
- `python3 -m tensor_logic serve --host 127.0.0.1 --port 8000`
- `python3 -m tensor_logic.reason --query ... --facts-file ...`
- `python3 web_workbench/server.py --host 127.0.0.1 --port 8080`
- `python3 tools/code_index.py --dump|--lookup|--status|--rebuild`
- Many `python3 experiments/exp*.py` and `python3 demos/*.py` scripts.

Local shell note: plain `python` was not available here. Commands that use
`sys.executable` or CI's configured Python are fine, but local docs that say
`python` may need a shell with that alias or a switch to `python3`.

## Validation surface observed

The repo-level test contract appears in three places:

- `CLAUDE.md`: `pytest tests/ -v`
- `README.md`: `python -m pip install -e ".[dev]"` and
  `python -m pytest tests/ -v`
- `.github/workflows/ci.yml`: installs `".[dev]"` and runs
  `python -m pytest tests/ -v`

Test ownership is broad:

- `tests/test_tensor_logic_core.py` covers closure, rule parsing, program rules,
  file loading, proofs, negative proofs, source-backed facts, REPL, includes,
  CLI/HTTP parity, repo ingest, repo-graph helpers, proof-tree rendering, and
  command execution.
- `tests/test_optimize.py` covers the generic optimize loop.
- `tests/test_reason.py` covers the EAV reason adapter.
- `tests/test_web_workbench.py` covers workbench subprocess behavior and parity
  with `tensor_logic.http_api`.
- `tests/test_code_index.py` covers the local AST indexer.
- `tests/test_exp79.py`, `tests/test_exp80.py`, and `tests/test_exp81.py` cover
  selected experiment modules.

The requested validation for this queue item is only:

```bash
git status --short
```

Full pytest was not run because this was a read-only architecture audit and the
Goal Pack supplied `git status --short` as the required validation.

## Stale assumptions and ownership risks

1. Package-root ownership is broad.

   Evidence: `tensor_logic/__init__.py` re-exports HTTP, ingest, repo-graph, and
   proof-tree helpers, while `tensor_logic/core.py` advertises a narrow surface.
   No local code imports `tensor_logic.core`. Future library consumers may
   import the full toolkit when they only need core semantics.

2. Rule language is mostly consolidated, but historical examples can mislead.

   Evidence: `docs/superpowers/specs/2026-05-04-rule-language-shape.md` says
   `program.Atom` and `program.Rule` are canonical. Current
   `tensor_logic/rules.py` imports those canonical dataclasses. Older
   experiments still contain local `<tl_rule>` parser/evaluator copies, so code
   search surfaces stale parser shapes beside the canonical adapter.

3. `tensor_logic/reason.py` crosses repo ownership.

   Evidence: its docstring calls it the `memjuice reason` subprocess entrypoint,
   but it lives in the `tensor_logic` package and is packaged with the reusable
   library. That may be correct as a dependency, but the owner and compatibility
   contract should be explicit.

4. Research helpers are only partially extracted.

   Evidence: exp78/79/81/83 use `tensor_logic.research.utils`, and exp83 uses
   `tensor_logic.research.slot_attention`, but exp79 and exp83 duplicate most of
   the synthetic-scene and TL-derived-relation wiring. If exp83 becomes active,
   this duplication is likely to drift.

5. Experiment dependency surface is wider than package metadata.

   Evidence: `pyproject.toml` declares runtime `torch` and dev
   `pytest/numpy/matplotlib`; experiments also import `scipy`, `transformers`,
   `mlx_lm`, `peft`, `datasets`, `accelerate`, and other optional packages. This
   is fine for historical scripts, but not for scripts promoted into tests or
   reusable workflows.

6. Script-mode path mutation remains common.

   Evidence: `rg "sys.path.insert|sys.path.append"` found path insertion in
   demos, tests, and several experiment scripts. This supports direct local
   script execution, but it blurs whether modules should be run after editable
   install, from repo root, or as standalone files.

7. Docs lag the current experiment shape.

   Evidence: README's repo layout says `experiments/ exp1..exp54`; the checkout
   has 99 files under `experiments/` and experiments through exp83. README also
   describes `demos/` as "5 headline runnable demos" while the directory has 8
   demo scripts. This is not a code defect, but it is stale navigation for future
   architecture work.

8. The workbench and HTTP API are sibling adapters, not one shared server.

   Evidence: `web_workbench/server.py` shells out to `python -m tensor_logic`,
   while `tensor_logic/http_api.py` provides direct in-process helpers and a JSON
   HTTP server. Tests assert parity for important proof behavior, which mitigates
   the split. The architectural decision is still implicit.

## Next safe work

1. Document and adopt the import-surface split.

   Acceptance criteria:

   - README or `docs/agents/domain.md` states when consumers should use
     `tensor_logic.core` versus `tensor_logic`.
   - At least one pure core demo or test imports from `tensor_logic.core`.
   - Package root remains backwards compatible.

   Validation:

   ```bash
   python3 -m pytest tests/test_tensor_logic_core.py -v
   python3 -m pytest tests/test_packaging_ci.py -v
   ```

2. Mark historical `<tl_rule>` parser copies as legacy or route active examples
   to `tensor_logic.rules.parse_rule`.

   Acceptance criteria:

   - Active docs point future agents to `tensor_logic.rules.parse_rule` and
     `tensor_logic.program.Rule`.
   - Historical experiment-local parsers are either explicitly labeled as
     archived experiment code or no longer used by active tests.
   - The 2026-05-04 rule-language spec remains the canonical contract.

   Validation:

   ```bash
   python3 -m pytest tests/test_tensor_logic_core.py -k "rule or provenance or proof" -v
   ```

3. Extract shared scene/TL wiring for exp79 and exp83 only if exp83 continues.

   Acceptance criteria:

   - Shared constants, `generate_sequence`, `compute_relations`,
     `generate_labeled_frames`, `build_tl_state`, `remove_object`, and
     `compute_gt_derived` live in a `tensor_logic.research` module.
   - Exp79 and exp83 keep experiment-specific model/training code local.
   - No generated data files are touched.

   Validation:

   ```bash
   python3 -m pytest tests/test_exp79.py -v
   python3 -m pytest tests/test_tensor_logic_core.py -v
   ```

4. Make the `tensor_logic.reason` ownership contract explicit.

   Acceptance criteria:

   - A short doc or module comment states whether `tensor_logic.reason` is a
     stable package feature or a memjuice adapter kept here for locality.
   - If stable, README includes the subprocess entrypoint.
   - If adapter-local, tests still cover it but package exports do not imply a
     broader public API.

   Validation:

   ```bash
   python3 -m pytest tests/test_reason.py tests/test_optimize.py -v
   ```

5. Normalize local Python command examples.

   Acceptance criteria:

   - Local docs consistently use `python3` or explain that `python` is provided
     by the activated environment.
   - CI can keep `python` because `actions/setup-python` provides it.

   Validation:

   ```bash
   python3 tools/code_index.py --status
   python3 -m pytest tests/test_code_index.py -v
   ```

## Handoff

Files changed by this queue item:

- `docs/overnight/tensor-experiments-architecture-map.md`

Commit/PR status:

- HEAD SHA at handoff: `5565791a8c888511bbbff1107d2d357164f88baa`
- No commit created.
- No PR created.

Required validation:

```bash
git status --short
```

Result: exited 0 with expected untracked report output:

```text
?? docs/overnight/
```

Scoped status confirms the only changed file:

```text
?? docs/overnight/tensor-experiments-architecture-map.md
```

Blockers:

- None for the requested audit.
- Full test validation was intentionally not run; the queue item requested
  `git status --short`.
