# tensor-logic dependency-surface audit

Queue item: `tensor-logic-dependency-surface`
Date: 2026-05-07
Focus area: dependency surface
Repo: `tensor-logic`
Branch observed: `codex/goal-tensor-logic-dependency-surface`

## Purpose and State

`tensor-logic` is a Python research repo that now serves two roles: a reusable
`tensor_logic` package for tensor-logic reasoning utilities, and a large
collection of demos, one-off experiments, notes, generated experiment data, and
local workbench tools. The reusable package is small; the dependency surface is
mostly created by experiment scripts and generated artifacts around it.

Initial dirty state was clean. `git status --short --branch` returned only:

```text
## codex/goal-tensor-logic-dependency-surface
```

Starting HEAD before this report was:

```text
c9e2cfce206bb279a4348189726908fcb436d3dc
```

No product code, generated experiment data, external services, pushes, or PRs
were touched. The only intended durable change from this worker is this report.

## Command Evidence

| Command | Observation |
|---|---|
| `llm-tldr tree .` | Repo has `tensor_logic/`, `experiments/`, `demos/`, `phase_training/`, `web_workbench/`, `tests/`, `docs/`, `notes/`, `tools/`, and committed experiment data folders. |
| `rg --files` | Tracked surface includes 167 Python files, 31 Markdown files, 22 JSON files, 7 JSONL files, 4 `.tl` examples, 2 PNGs, 1 workflow, and 1 `pyproject.toml`. |
| `du -sh . tensor_logic experiments tests docs web_workbench phase_training demos` | Repo is 11M total; `experiments/` is 10M; `tensor_logic/` is 148K; `tests/` is 160K. |
| `python3 --version` | Local interpreter is Python 3.12.8; CI pins Python 3.11. |
| Python AST import scan over `tensor_logic`, `experiments`, `demos`, `phase_training`, `tools`, `web_workbench`, `tests` | Scanned 167 Python files. Top import roots include `torch` in 104 files, `tensor_logic` in 17 files, `transformers` in 5 files, `numpy` in 4 files, `matplotlib` in 2 files, `mlx_lm` in 1 file, `peft` in 1 file, `datasets` in 1 file, and `scipy` in 1 file. |
| `python3 - <<'PY' ... importlib.util.find_spec(...)` | Local environment has `torch`, `numpy`, `matplotlib`, `scipy`, `tqdm`, and `pytest`; it is missing `transformers`, `mlx_lm`, `peft`, and `datasets`. |
| `python3 tools/code_index.py --status` | Exited 1 with `stale: index.json is missing or older than source files`; `tools/index.json` is intentionally gitignored and absent on a clean checkout. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/ -q` | Passed: `198 passed in 34.50s`. This test run generated ignored `tools/index.json`; the worker removed it afterward. |
| `git status --short --ignored=matching` after cleanup | Empty before writing this report, confirming no ignored cache/index state was left behind. |

## Local File Evidence

- `pyproject.toml` declares package metadata, Python `>=3.11`, one runtime dependency (`torch>=2.0`), and dev extras for `pytest`, `numpy`, and `matplotlib`.
- `.github/workflows/ci.yml` installs `.[dev]` on Python 3.11 and runs `python -m pytest tests/ -v`.
- `README.md` documents worker validation as `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`; it also documents demo execution through `uv run --with torch ...`.
- `CLAUDE.md` documents MLX and transformers support in `experiments/exp78_rule_induction.py`, but those packages are not represented in `pyproject.toml` extras.
- `.gitignore` excludes `.venv/`, `.uv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.cocoindex_code/`, `tools/index.json`, build outputs, `*.pt`, and `*.pth`.
- `tools/code_index.py` writes `tools/index.json` from `--rebuild`, `--lookup`, and `--dump` through `ensure_fresh`; the generated file is ignored.
- `tests/test_code_index.py` verifies `tools/index.json` is gitignored, but also runs CLI rebuild/status tests that can create the ignored index in a worker checkout.
- `tests/test_packaging_ci.py` asserts the current install/test contract and README/docs validation strings, so dependency contract changes should update this test.
- `tensor_logic/__main__.py` exposes CLI entrypoints: `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`.
- `tensor_logic/http_api.py` provides a local HTTP API using stdlib `ThreadingHTTPServer`; it accepts source and path payloads and routes to package execution helpers.
- `web_workbench/server.py` writes editor source to a temporary `.tl` file and calls `[sys.executable, "-m", "tensor_logic"]` through `subprocess.run` without a timeout.
- `tensor_logic/execution.py` uses temporary `.tl` files to load source strings, and removes those temporary files in `finally`.
- `tensor_logic/ingest.py` parses Python imports with stdlib `ast` and skips cache, VCS, virtualenv, and hidden directories.
- `experiments/exp53_real_imports.py` and `experiments/exp54_big_imports.py` call `pip download --no-deps --no-binary :all:` and extract archives into temp directories to evaluate third-party package import graphs.
- `experiments/exp60d_sft.py` documents `pip install torch transformers peft datasets accelerate`, imports `transformers`, `peft`, `datasets`, and optional `tqdm`, and writes a default LoRA output under `experiments/exp60_data/lora_adapter`.
- `experiments/exp78_rule_induction.py` lazily imports `transformers` and `mlx_lm`; it has a no-transformers fallback, but MLX/transformers behavior is not covered by declared extras.
- `experiments/exp83_slot_attention.py` imports `scipy.optimize.linear_sum_assignment`, `numpy`, `matplotlib`, and `torch`; it writes `experiments/exp83_slot_data/results.json` and `complexity_curve.png`.

## Dependency Surface Map

### Declared Python contract

The published package contract is intentionally small:

- Runtime: `torch>=2.0`
- Dev extra: `pytest>=8.0`, `numpy>=1.24`, `matplotlib>=3.7`
- Package discovery: `tensor_logic*`
- No lockfile, `requirements.txt`, `uv.lock`, `poetry.lock`, `tox.ini`, `noxfile.py`, `Makefile`, or pre-commit config was found. The broad lockfile scan only matched `experiments/exp27_blocks_world.py` because the filename contains `block`.

This is enough for the current test suite and the core library, but not enough
to reproduce the full experiment portfolio.

### Actual import surface

The core package is mostly stdlib plus `torch`. The broader repo includes these
additional stacks:

- Research/data stack: `numpy`, `matplotlib`, `scipy`
- LM stack: `transformers`, `mlx_lm`
- SFT stack: `peft`, `datasets`, `accelerate` documented but not imported directly in the scan, and `tqdm`
- Network/package acquisition: `pip download` through subprocess in `exp53` and `exp54`
- Local HTTP/subprocess stack: stdlib `http.server`, `tempfile`, `subprocess`

The missing local modules (`transformers`, `mlx_lm`, `peft`, `datasets`) mean a
clean `.[dev]` environment can pass CI while failing several documented
experiments.

### Scripts and entrypoints

The repo has no `[project.scripts]` entry in `pyproject.toml`. Users run tools
through module/script commands:

- `python -m tensor_logic ...` for CLI commands implemented in `tensor_logic/__main__.py`
- `python web_workbench/server.py --host 127.0.0.1 --port 8080`
- `python tools/code_index.py --status|--lookup|--dump|--rebuild`
- Many direct experiment scripts, several with `--quick` or output flags

This is workable for a research repo, but it makes the supported command set
depend on README/docs discipline rather than package metadata.

### Generated artifacts and local-only state

Tracked generated artifacts are already present:

- `experiments/exp60_data/*.jsonl`: 3 files, 655,897 bytes
- `experiments/exp76_data/*.jsonl`: 4 files, 2,753,555 bytes
- `experiments/exp78_data/results.json`: 4,744 bytes
- `experiments/exp79_data/results.json` plus `complexity_curve.png`: 54,312 bytes
- `experiments/exp83_slot_data/results.json` plus `complexity_curve.png`: 23,811 bytes
- `experiments/exp87_support_data` through `experiments/exp95_scored_object_hypotheses_data`: committed `results.json` and `results_quick.json` outputs, ranging from about 10K to 2.3M per folder

Ignored local-only state includes Python caches, virtualenvs, `.uv/`,
`.cocoindex_code/`, `tools/index.json`, build outputs, and PyTorch checkpoint
extensions. It does not explicitly ignore LoRA adapter folders, safetensors, or
Hugging Face model cache paths under experiment data folders.

### Environment variables and secrets

No active source under `tensor_logic`, `experiments`, `demos`,
`phase_training`, `web_workbench`, `tests`, or `tools` reads `os.environ` for
API keys or service tokens. The env-var search only found false positives in
identifier names and historical docs. There are no `.env*` files in the repo
scan. The main external risks are package downloads, model downloads from
`transformers`/MLX paths, and local subprocess execution, not checked-in
secrets.

## Risks and Stale Assumptions

1. Declared dependencies understate the experiment surface. `pyproject.toml`
   and CI make `.[dev]` look complete, while documented scripts need
   `transformers`, `mlx_lm`, `peft`, `datasets`, `accelerate`, `scipy`, and
   sometimes model downloads. Morning reviewers should not assume "CI green"
   means "all experiments reproducible."

2. The code-index workflow creates hidden local state. `CLAUDE.md` tells agents
   to use `tools/code_index.py`, but `--lookup` and `--dump` rebuild the ignored
   `tools/index.json` when missing or stale. Tests also rebuild it. This can
   leave worktrees looking clean under `git status --short` while carrying an
   ignored generated index.

3. Several scripts overwrite tracked experiment outputs. `exp83`, `exp79`, and
   support-eval experiments write `results.json`, `results_quick.json`, or PNGs
   under committed data folders. Running a script for validation can mutate
   review artifacts unless workers know which commands are safe.

4. External package/model acquisition is not gated in metadata. `exp53` and
   `exp54` use `pip download` against live package indexes; LM experiments can
   trigger Hugging Face or MLX model loads. Those are appropriate for explicit
   experiment runs, but they should not be treated as cheap local validation.

5. Local web execution has no timeout boundary. `web_workbench/server.py` shells
   out to `python -m tensor_logic` with user-provided source and no timeout.
   This is local-only, but a recursive or expensive query can tie up a worker or
   browser QA session.

6. Python version coverage is narrower than the package range. The package says
   Python `>=3.11`, CI tests 3.11, and this local audit ran under 3.12.8. There
   is no matrix or lockfile proving behavior across future Python releases or
   Torch wheels.

## Validation Candidates

| Command | Expected status | Observed status in this audit |
|---|---|---|
| `git status --short` | Pass. Required queue validation; should show only the report before commit, or no output after a local commit. | Passed after writing this report with `?? docs/overnight/`. |
| `python3 -m pytest tests/test_packaging_ci.py -q` | Pass; verifies current packaging/CI/docs contract. | Passed: `4 passed in 0.02s`. |
| `python3 -m pytest tests/test_tensor_logic_core.py -q` | Pass; cheap core behavior check, no optional LM deps. | Passed: `54 passed in 10.90s`. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/ -q` | Pass in current local environment; may create ignored `tools/index.json` through `tests/test_code_index.py`. | Passed: `198 passed in 34.50s`; generated index was removed. |
| `python3 tools/code_index.py --status` | Fail on a clean checkout unless `tools/index.json` has been rebuilt. This is expected under current design. | Failed with exit 1: `stale: index.json is missing or older than source files`. |
| `python3 experiments/exp86_support_baselines.py --quick` | Candidate support/stability smoke from README; expected pass if core deps are installed. | Not run in this audit; product experiment execution was outside the dependency-surface report scope. |
| `python3 experiments/exp60d_sft.py --skip-train --n-eval 1` | Expected fail locally unless `transformers`, `peft`, and `datasets` are installed and model access is available. | Not run; import availability scan already showed missing LM/SFT deps. |
| `python3 experiments/exp83_slot_attention.py --help` | Expected pass locally because `scipy`, `numpy`, `matplotlib`, and `torch` are installed. Full run is expensive and mutates tracked outputs. | Not run to avoid accidental artifact mutation. |

## Next Safe Work

1. Add optional dependency extras for experiment families.
   Acceptance: `pyproject.toml` exposes clear extras such as `.[slot]`,
   `.[lm]`, `.[sft]`, or `.[experiments]`; README and `tests/test_packaging_ci.py`
   document which commands each extra supports; core `.[dev]` remains small.
   Validation: `python3 -m pytest tests/test_packaging_ci.py -q` and an import
   availability smoke for each new extra in an environment where those extras
   are installed.

2. Make code-index commands side-effect explicit.
   Acceptance: `tools/code_index.py --status` remains read-only; `--dump` and
   `--lookup` either support a documented `--no-rebuild` mode or print a clear
   "will rebuild ignored tools/index.json" message before writing; docs tell
   agents how to clean generated index state.
   Validation: `python3 -m pytest tests/test_code_index.py -q` and
   `git status --short --ignored=matching` after the test run.

3. Add an experiment artifact manifest.
   Acceptance: each committed `experiments/*_data/` folder has a documented
   producing script, whether files are canonical or regenerable, whether script
   runs overwrite tracked files, and the cheapest non-mutating validation.
   Validation: a small test or script checks every tracked `*_data` directory is
   listed in the manifest.

4. Gate external acquisition scripts.
   Acceptance: `exp53` and `exp54` require an explicit flag such as
   `--allow-network` for `pip download`, document package-index dependency, and
   use safe archive extraction behavior where the supported Python version
   allows it.
   Validation: `python3 experiments/exp53_real_imports.py --help` and a unit
   test for the no-network default path.

5. Add a timeout to local web subprocess execution.
   Acceptance: `web_workbench/server.py` bounds `subprocess.run` with a timeout
   and returns a structured timeout error; tests cover a timeout path without
   sleeping for a long time.
   Validation: `python3 -m pytest tests/test_web_workbench.py -q`.

## Non-Goals

- No product code edits.
- No dependency installation or lockfile generation.
- No experiment reruns that write result JSON, PNG, checkpoints, adapters, or
  model outputs.
- No network access, package downloads, model downloads, deploys, pushes, or PR
  creation.
- No changes to GitHub/Linear/external tracker state.

## Unknowns

- Whether the intended public install contract is only the reusable
  `tensor_logic` package, or the full experiment portfolio.
- Whether optional LM/SFT dependencies should be grouped by experiment number,
  by capability, or left as script-local docstrings.
- Whether committed result files under `experiments/*_data/` are canonical
  audit artifacts or disposable generated outputs.
- Whether downstream sibling repos (`fafsa-engine`, future `officeqa-*`) depend
  only on `tensor_logic`, or also assume experiment modules are importable.
- Whether `tools/index.json` should stay ignored and regenerated locally, or be
  replaced by a pure read-only AST query path for agent planning.

## Handoff

The dependency surface is manageable if reviewers distinguish three layers:
core package (`torch` plus stdlib), tested developer contract (`.[dev]`), and
optional experiment stacks. The highest-leverage cleanup is to encode those
layers in package metadata/docs and to make mutating/generated state explicit
before more workers use this repo as a dependency or validation target.

Local commit/PR handoff was not completed. `git add
docs/overnight/tensor-logic-dependency-surface.md` failed because the worktree
index lives under `/Users/jwalinshah/projects/tensor/tensor-logic/.git/worktrees/tensor-logic-dependency-surface/index.lock`,
which is outside this sandbox's writable roots. No PR was created, per queue
scope.
