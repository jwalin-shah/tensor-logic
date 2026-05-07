# tensor-experiments risk-register audit

Queue item: `tensor-experiments-risk-register`  
Repo path: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-experiments-risk-register`  
Branch: `codex/goal-tensor-experiments-risk-register`  
Starting HEAD: `5565791a8c888511bbbff1107d2d357164f88baa`  
Focus area: security, credentials, data, deployment, destructive-command, and external-service risks.

## Scope and decisions

This was a read-only audit plus this single report file. I did not change product code, generated experiment data, tests, CI, external trackers, deploys, Kaggle artifacts, model artifacts, or PR state.

The repo is a research codebase plus a reusable `tensor_logic` Python package. The main live surfaces are:

- `tensor_logic/`: reusable library, CLI, parser, HTTP API, proof/reasoning helpers.
- `experiments/`: numbered research scripts, some pure local, some remote-download/model-training oriented.
- `web_workbench/`: local browser workbench that shells out to `python -m tensor_logic`.
- `docs/` and `notes/`: run protocols, specs, research memos, and long transcripts.
- `tests/`: pytest coverage for core tensor logic, packaging/CI, web workbench, exp79/80/81, optimize, reason, and code index.

## Command evidence

Commands run during the audit:

- `git status --short --branch`: observed clean branch `codex/goal-tensor-experiments-risk-register` before this report.
- `git rev-parse --show-toplevel && git branch --show-current && git rev-parse HEAD`: confirmed repo root, branch, and starting HEAD `5565791a8c888511bbbff1107d2d357164f88baa`.
- `llm-tldr tree .`: mapped package, experiments, docs, tests, web workbench, examples, and phase training files.
- `rtk read CLAUDE.md`, `rtk read pyproject.toml`, `rtk read README.md`, `rtk read docs/RUN_PROTOCOL.md`, `rtk read docs/SYMPHONY_RUN_PROTOCOL.md`: gathered local conventions, dependencies, and run protocol.
- `rtk grep "os\\.environ|os\\.getenv|OPENAI|ANTHROPIC|HF_|HUGGING|KAGGLE|TOKEN|API_KEY|SECRET|password" .`: found API-key examples only in docs/plans, plus non-secret token constants in an experiment.
- `rtk grep "subprocess|shell=True|os\\.system|shutil\\.rmtree|rm -rf|unlink\\(|Path\\([^)]*\\)\\.write|write_text|open\\([^)]*, ['\\\"]w|NamedTemporaryFile|mkdtemp" .`: found subprocess/temp-file/write surfaces in `web_workbench`, `tensor_logic`, `tools`, tests, and remote experiment scripts.
- `rtk grep "requests|urllib|httpx|FastAPI|uvicorn|HTTPServer|BaseHTTPRequestHandler|webbrowser|fetch\\(|localhost|0\\.0\\.0\\.0" .`: found local HTTP servers in `tensor_logic/http_api.py` and `web_workbench/server.py`.
- `rtk grep "torch\\.load|pickle|safetensors|from_pretrained|AutoModel|mlx_lm|huggingface|kaggle|dataset|download|url" .`: found remote model/download surfaces and Kaggle artifact references.
- `rg --files -g '.*' -g '!/.git'`: found `.gitignore`.
- `rg --files .github . 2>/dev/null | rg '(^|/)\\.github/|(^|/)requirements|uv.lock|poetry.lock|Pipfile|environment\\.yml|conda|Dockerfile|docker-compose|Makefile|justfile|noxfile|tox.ini|setup.cfg|ruff.toml|mypy.ini|pre-commit'`: found only `.github/workflows/ci.yml` among workflow/lock/build config candidates.
- `git ls-files | wc -l`: observed 197 tracked files.
- `git ls-files -o --exclude-standard`: no untracked non-ignored files before this report.
- `git ls-files -o -i --exclude-standard`: no ignored files before code-index lookup.
- `du -sh . experiments notes docs tensor_logic tests web_workbench`: repo size was about 5.5M, with `experiments/` about 4.3M.
- `find experiments -maxdepth 2 -type f (...)`: found committed JSON/JSONL/PNG experiment artifacts, no committed `.pt` or `.safetensors` files.
- `wc -l experiments/exp60_data/*.jsonl experiments/exp76_data/*.jsonl experiments/exp80_spot_check_cases.json`: observed 6,221 total lines across those data/check files.
- `python tools/code_index.py --lookup load_tl`: failed because `python` is not present in this shell.
- `python3 tools/code_index.py --lookup load_tl`: succeeded; signature `tensor_logic.file_format.load_tl(path: str) -> LoadedProgram`.
- `python3 tools/code_index.py --lookup ingest_python`: succeeded; signature `tensor_logic.ingest.ingest_python(root: str | Path) -> PythonImportGraph`.
- `python3 tools/code_index.py --lookup execute_query`: succeeded; signature `tensor_logic.execution.execute_query(program: Program, relation: str, args: list[str] | tuple[str, ...], recursive: bool) -> dict[str, Any]`.
- `python3 --version`: Python 3.12.8.
- `python3 -m pytest --version`: pytest 9.0.3.
- Python import availability check: `torch`, `numpy`, `matplotlib`, `pytest`, `scipy`, and `tqdm` are importable; `transformers`, `peft`, `datasets`, and `mlx_lm` are not importable in this environment.

Note: the `python3 tools/code_index.py --lookup ...` commands created ignored `tools/index.json` as designed by `tools/code_index.py`; I removed that ignored generated file after the lookup so this worktree keeps exactly one intentional output file. Normal `git status --short` remains scoped to this report.

## Local evidence map

1. `README.md` documents this as a learning/research project and a shared `tensor_logic` library dependency for sibling projects. It also advertises worker validation as `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`.
2. `CLAUDE.md` repeats the full-test command and documents LM backend support in `experiments/exp78_rule_induction.py`, including MLX/transformers fallback and a known VLM incompatibility.
3. `pyproject.toml` declares only `torch` as a runtime dependency and `matplotlib`, `numpy`, and `pytest` as dev dependencies.
4. `.github/workflows/ci.yml` installs `.[dev]` and runs `python -m pytest tests/ -v` on push to `main` and pull requests.
5. `.gitignore` excludes caches, virtualenvs, `*.pt`, `*.pth`, `.cocoindex_code/`, `tools/index.json`, `dist/`, `build/`, and `*.egg-info/`.
6. `tensor_logic/__main__.py` exposes CLI commands `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`; `serve` defaults to `127.0.0.1:8000`.
7. `tensor_logic/http_api.py` exposes POST endpoints `/ingest-python`, `/run`, `/query`, and `/prove` using `ThreadingHTTPServer`; `ingest-python` accepts a caller-provided path and generic exceptions are returned as error strings.
8. `web_workbench/server.py` exposes local POST `/api/*`, writes editor source to a temp `.tl` file, and invokes `[sys.executable, "-m", "tensor_logic", ...]` with `subprocess.run(..., check=False)` and no timeout.
9. `tensor_logic/file_format.py` implements `include "path"` by joining the include with the current file's base dir and recursively loading it; there is cycle detection but no sandbox/root restriction and absolute paths are not rejected.
10. `tensor_logic/execution.py` writes caller-supplied source to a temp `.tl` file, loads it, and unlinks it.
11. `tensor_logic/ingest.py` recursively walks `*.py` files under an arbitrary root, skips common cache/venv directories, parses Python via `ast`, and renders a TL import graph.
12. `experiments/exp53_real_imports.py` and `experiments/exp54_big_imports.py` use `pip download --no-deps --no-binary :all:` for fixed public packages, then `tarfile.extractall()` or `zipfile.extractall()`.
13. `experiments/exp60d_sft.py` imports transformers/PEFT/datasets lazily, loads model IDs via `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`, writes failure dumps, saves LoRA adapters/tokenizers, and writes a manifest in the output directory.
14. `experiments/exp78_rule_induction.py` optionally uses transformers or `mlx_lm` to load a model for LM pruning; if transformers is missing it falls back to all relations instead of blocking.
15. `experiments/exp79_lewm_tl.py` loads `encoder.pt` and `probe.pt` via `torch.load` when `--skip-train` is passed.
16. `experiments/exp83_slot_attention.py` imports `scipy.optimize.linear_sum_assignment`, `matplotlib`, `numpy`, and `torch`, while those non-core requirements are not all declared as runtime dependencies.
17. `docs/RUN_PROTOCOL.md` documents a previous lost Kaggle adapter and requires future runs to write `adapter/`, `manifest.json`, and `failures.jsonl`, then create a Kaggle dataset pointer instead of committing weights.
18. `experiments/exp80_fafsa_kb.py` implements a 2024-25 FAFSA SAI Formula A estimator with cited computation steps and real financial-aid claims.
19. `experiments/exp80_fafsa_wizard.py` is an interactive estimator that tells users they likely qualify for maximum/partial Pell Grant based on computed SAI.
20. `tests/test_exp80.py` covers invariants, citations, counterfactual monotonicity, named synthetic cases, and seeded synthetic families, but not external ED worked-example parity.

## Risk register

### R1. Local HTTP APIs can become unsafe if bound beyond localhost

Severity: high if exposed beyond the developer machine; medium for localhost-only use.

Evidence:

- `tensor_logic/__main__.py` exposes `serve --host --port`, defaulting to `127.0.0.1:8000`.
- `tensor_logic/http_api.py` uses `ThreadingHTTPServer` and accepts caller input for `/ingest-python`, `/run`, `/query`, and `/prove`.
- `web_workbench/server.py` accepts `--host`, defaulting to `127.0.0.1`, but does not prevent `0.0.0.0`.

Why it matters:

The APIs are designed for local development. If a user binds them to a LAN/public interface, there is no authentication, no CSRF protection, no request-size limit, and no explicit compute timeout. The `/ingest-python` path and TL execution paths can read local source structure or run expensive recursive/proof computations.

Current mitigating evidence:

- Defaults are localhost.
- Tests cover helper parity and workbench smoke behavior, not network exposure.

Next safe work:

- Add host-binding warnings or explicit `--unsafe-public` opt-in for non-loopback hosts.
- Add request-size limits and timeout/step limits for API execution.
- Add tests for oversized JSON bodies and non-loopback host guard behavior.

### R2. TL `include` can read outside an intended workspace

Severity: high if TL input comes from an exposed API or untrusted user.

Evidence:

- `tensor_logic/file_format.py` parses `include "path"`.
- `_load_into()` resolves includes with `os.path.realpath(os.path.join(base_dir, include_path))`.
- It detects include cycles but does not reject absolute include paths or `..` traversal.
- `tensor_logic/execution.py` writes caller-provided source to a temp file and calls `load_tl(path)`.

Why it matters:

A TL file can include paths outside the source directory. If `run_source()` or `web_workbench` ever accepts untrusted TL text, the include mechanism can read arbitrary readable local files, at least far enough to leak parse-error paths/content shape or load any file that happens to parse as TL. Even in local-only use, accidental includes can couple experiments to machine-local paths.

Current mitigating evidence:

- The workbench stores submitted source in a temp `.tl` file and unlinks it.
- Cycle detection prevents infinite include recursion.
- Existing tests cover normal includes and include cycles.

Next safe work:

- Add a `base_dir` or `allow_includes` policy to `load_tl`.
- Reject absolute includes and traversal outside the root by default.
- Add tests for absolute path rejection, `../` rejection, and opt-in trusted includes.

### R3. Web workbench subprocess execution has no timeout or size limit

Severity: medium for local development; high if exposed.

Evidence:

- `web_workbench/server.py` writes request source to a temp file, builds a command, and runs `subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT.parent), check=False)` with no timeout.
- `web_workbench/static/app.js` posts arbitrary editor content to `/api/<action>`.
- `web_workbench/README.md` explicitly states the server writes editor contents to a temp `.tl` file and invokes `python -m tensor_logic`.

Why it matters:

Malformed or intentionally expensive TL programs can consume CPU/memory or hang the browser flow. `capture_output=True` also buffers stdout/stderr in memory. A local-only tool can tolerate more risk, but the server interface makes it easy to accidentally expose a denial-of-service surface.

Current mitigating evidence:

- The command list is fixed and uses `shell=False`.
- Tests cover normal query and why-not flows.

Next safe work:

- Add `timeout=` to `subprocess.run`.
- Bound request source size and stdout/stderr returned to the browser.
- Add tests for timeout mapping to HTTP 408/400 and bounded output.

### R4. Remote package download experiments extract archives without path filtering

Severity: medium because package names are fixed; high if generalized or run in a shared workspace.

Evidence:

- `experiments/exp53_real_imports.py` downloads source archives for fixed packages via `pip download`.
- `experiments/exp54_big_imports.py` does the same for larger packages.
- Both scripts call `tarfile.extractall()` and `zipfile.extractall()` without checking archive member paths.
- Both scripts remove/recreate package directories under a temp destination with `shutil.rmtree()`.

Why it matters:

Archive extraction without member filtering is a known path traversal footgun. The current packages are hardcoded and fetched from PyPI via pip, but the script structure invites reuse with other package names. If a package archive is malicious or the source is compromised, extraction could write outside the intended temp directory.

Current mitigating evidence:

- The package lists are constants, not user CLI args.
- Extraction happens under a temp directory during normal script execution.

Next safe work:

- Add safe tar/zip extraction helpers that reject absolute paths and paths escaping the destination.
- Add unit tests using synthetic malicious tar/zip entries.
- Add an explicit `--allow-network-downloads` flag if these scripts gain CLI package selection.

### R5. Optional model/download dependencies are not declared or locked

Severity: medium for reproducibility; high for overnight/remote workers that assume docs are complete.

Evidence:

- `pyproject.toml` declares `torch` runtime and only `matplotlib`, `numpy`, `pytest` dev extras.
- `experiments/exp60d_sft.py` requires transformers, PEFT, datasets, accelerate, and often tqdm, but these are only documented in the file docstring.
- `experiments/exp78_rule_induction.py` optionally imports transformers and `mlx_lm`.
- Local import check found `transformers`, `peft`, `datasets`, and `mlx_lm` unavailable here.
- There is no lockfile, requirements file, Dockerfile, tox/nox config, or pre-commit config.

Why it matters:

Core CI can pass while important experiment entrypoints fail on a clean worker. Remote model loading can download large artifacts and may vary over time if model revisions are not pinned.

Current mitigating evidence:

- LM functionality in `exp78` falls back when transformers is unavailable.
- CI covers the reusable package tests rather than every research experiment.

Next safe work:

- Add optional extras such as `[project.optional-dependencies].lm`, `.remote`, and `.vision`.
- Pin or record model revisions in manifests for experiment runs.
- Add a dependency smoke test that imports modules for each documented extra without launching downloads.

### R6. Kaggle/adapter persistence is documented but not enforced

Severity: medium to high for expensive/long-running training work.

Evidence:

- `docs/RUN_PROTOCOL.md` says an exp76c adapter was lost because Kaggle `/kaggle/working/` was wiped.
- The protocol requires each run to write `adapter/`, `manifest.json`, and `failures.jsonl`.
- It instructs users not to commit weights and instead commit `results.json` plus `ADAPTER.md`.
- Current committed experiment artifacts include JSON/JSONL/PNG files, but no `ADAPTER.md` files were found under `experiments/*_data/`.
- `.gitignore` excludes `*.pt` and `*.pth`, but not every possible model artifact extension or adapter directory shape.

Why it matters:

The repo already had a lost-artifact incident. Without enforcement, a future worker can produce metrics without a durable adapter pointer, or preserve adapter files only in a remote notebook session.

Current mitigating evidence:

- `exp60d_sft.py` writes a `manifest.json` into its output dir.
- `docs/RUN_PROTOCOL.md` clearly states the intended manual workflow.

Next safe work:

- Add a manifest/pointer validator for committed `experiments/exp*_data/results.json` and `ADAPTER.md` where applicable.
- Add a checklist to CI or a docs-only test that fails if a run result references an adapter without a pointer.
- Extend `.gitignore` for common adapter/checkpoint directories while keeping pointer files trackable.

### R7. FAFSA estimator is a real-world financial-aid risk surface

Severity: high if presented outside research/demo context.

Evidence:

- `experiments/exp80_fafsa_kb.py` says it implements the 2024-25 FAFSA SAI guide, Formula A dependent student coverage.
- `experiments/exp80_fafsa_wizard.py` asks plain-English financial questions and tells users they likely qualify for maximum or partial Pell Grant based on SAI.
- `tests/test_exp80.py` validates invariants and synthetic cases, but does not prove parity with official ED worked examples.
- `experiments/exp80_validate_synthetic.py` says spot-check cases should be manually verified against `studentaid.gov/aid-estimator/`.

Why it matters:

Financial-aid calculations are policy and year specific. The code uses 2024-25 constants and explicitly omits independent-student Formula B/C coverage. The wizard wording can sound user-facing and advisory despite validation being synthetic and incomplete.

Current mitigating evidence:

- The module docstring states Formula A only and cites the guide version.
- Tests check internal invariants and traces carry citations/formulas.

Next safe work:

- Add a visible disclaimer in the wizard and README that this is research, 2024-25 only, dependent Formula A only, not financial advice.
- Add official worked-example parity tests or a tracked validation matrix.
- Rename/guard demo outputs that say "likely qualify" until official parity is proven.

### R8. Documentation commands assume `python`, but this local shell only has `python3`

Severity: low to medium, but it blocks exact worker instructions.

Evidence:

- `README.md`, `CLAUDE.md`, `.github/workflows/ci.yml`, and tests expect commands starting with `python`.
- Running `python tools/code_index.py --lookup load_tl` failed with `python: command not found`.
- Running `python3 tools/code_index.py --lookup load_tl` succeeded.
- Local Python is 3.12.8, while CI declares Python 3.11.

Why it matters:

The repo's documented validation and planning commands are not portable to this local environment as written. A worker following docs literally will fail before reaching tests or code-index lookup.

Current mitigating evidence:

- CI uses GitHub's Python setup, where `python` is usually available.
- `python3` works locally.

Next safe work:

- Document `python3` as the local fallback or use `python -m` only where the environment guarantees it.
- Add a tiny `make`/`just`/script wrapper if this repo is expected to be run across macOS shells.
- Update `CLAUDE.md` code-index instruction to mention `python3` fallback.

### R9. Experiment scripts overwrite committed result artifacts by default

Severity: medium for worktree hygiene and reproducibility.

Evidence:

- `experiments/exp78_rule_induction.py` defaults `--out` to `experiments/exp78_data/results.json` and writes it.
- `experiments/exp79_self_play_loop.py` writes JSON to its selected output path.
- `experiments/exp60d_sft.py` defaults `--out` to `experiments/exp60_data/lora_adapter` and writes model/manifest artifacts there.
- `experiments/exp80_validate_synthetic.py` writes `experiments/exp80_spot_check_cases.json`.
- `.gitignore` excludes some generated artifacts but not every JSON result path.

Why it matters:

Running experiments from the repo root can mutate committed results or create large ignored artifacts. That is fine for intentional experiment work, but unsafe for audit workers or validation scripts unless output paths are redirected to `/tmp`.

Current mitigating evidence:

- Many scripts expose `--out` or path arguments.
- `docs/RUN_PROTOCOL.md` explains run output structure and artifact persistence.

Next safe work:

- Standardize a `--out` default outside tracked paths for exploratory runs, or require explicit `--overwrite-committed-results`.
- Add README guidance for smoke tests using `/tmp`.
- Add tests that result-writing helpers create parent dirs and do not overwrite without opt-in.

### R10. Long transcripts and notes can accumulate sensitive operational history

Severity: low in the current scan; can become medium over time.

Evidence:

- `notes/SESSION_TRANSCRIPT.md` includes a local working directory and commands involving `alphaxiv auth set-api-key`.
- Grep found API-key examples in docs/plans, but they use placeholder `sk-...` values.
- No untracked non-ignored files were present before this report.

Why it matters:

Research repos with copied transcripts often accumulate local paths, tool names, account workflows, or pasted credentials. The current scan did not find live secrets, but the note structure invites future leakage unless review discipline exists.

Current mitigating evidence:

- No actual secret-looking values were found by the keyword scan.
- `.gitignore` excludes common local files and logs.

Next safe work:

- Add a lightweight secret scan command to ship-check/workflow docs.
- Add note hygiene guidance for transcripts before committing.
- Consider excluding raw transcripts or storing redacted summaries when the transcript is not needed for reproducibility.

## Stale assumptions and unsupported claims

- The docs and CI assume `python` exists. This local worker shell only has `python3`.
- `pyproject.toml` does not represent the full experiment dependency surface. Several experiment scripts require packages outside declared runtime/dev extras.
- `docs/RUN_PROTOCOL.md` says adapter pointers should be committed, but no `ADAPTER.md` files were found under current experiment data dirs.
- `experiments/exp80_fafsa_kb.py` comments mention asset-exempt concepts in sample families, but the visible Formula A implementation does not expose a full asset-exemption branch in the inspected section.
- The FAFSA wizard's user-facing wording is stronger than the committed validation evidence.
- Remote download/model-loading scripts are research scripts, not production entrypoints, but the README headline claims rely on outputs from some of those scripts.

## Validation command candidates

Required queue validation:

- `git status --short`
  - Expected: exits 0. Before this report it printed nothing. After this report and before any commit it should show only `?? docs/overnight/tensor-experiments-risk-register.md`.
  - Purpose: proves scope of filesystem mutation.

Cheap local validation candidates:

- `python3 -m pytest tests/test_packaging_ci.py -q`
  - Expected: pass in this environment.
  - Purpose: checks pyproject, README worker validation text, and GitHub Actions validation text.

- `python3 -m pytest tests/test_web_workbench.py -q`
  - Expected: pass in this environment.
  - Purpose: covers workbench subprocess path and HTTP/API helper parity without starting a server.

- `python3 -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_include_directive tests/test_tensor_logic_core.py::TestTensorLogicCore::test_include_cycle_raises -q`
  - Expected: pass in this environment.
  - Purpose: baseline for include behavior before sandbox hardening.

- `python3 -m pytest tests/test_exp80.py -q`
  - Expected: pass in this environment.
  - Purpose: covers current FAFSA invariants and synthetic cases; does not prove official parity.

Full validation candidate:

- `python3 -m pytest tests/ -v`
  - Expected: likely pass if the local package is installed/editable and dependencies are importable. This audit did not run it because the queue validation is `git status --short` and the task is report-only.
  - Risk: repo docs say `python -m pytest`, but this shell lacks `python`.

External or potentially mutating validation candidates:

- `python3 experiments/exp53_real_imports.py`
  - Expected: blocked/unsafe for this audit because it performs network downloads and archive extraction.

- `python3 experiments/exp54_big_imports.py`
  - Expected: blocked/unsafe for this audit for the same reason, with larger packages.

- `python3 experiments/exp60d_sft.py`
  - Expected: fail/block in this environment without optional `transformers`, `peft`, and `datasets`; also writes model/adapter artifacts.

- `python3 experiments/exp78_rule_induction.py --out /tmp/exp78-results.json`
  - Expected: likely pass locally without LM; safe if output is redirected to `/tmp`.

- `python3 experiments/exp78_rule_induction.py --lm --out /tmp/exp78-lm-results.json`
  - Expected: should fall back when transformers is unavailable, but does not validate real LM-pruner behavior.

## Independently grabbable next tasks

### Task 1: Sandbox TL include loading

Problem:

`load_tl()` allows includes outside the caller's intended root. This is acceptable for trusted local files but risky for HTTP/workbench execution.

Acceptance criteria:

- `include "local.tl"` in the same directory still works.
- Absolute include paths are rejected by default.
- Includes that resolve outside the configured root via `..` are rejected by default.
- A trusted opt-in path exists if old local workflows need cross-directory includes.
- Error messages do not leak unnecessary host path details in HTTP responses.

Validation:

- `python3 -m pytest tests/test_tensor_logic_core.py -k "include" -q`
- Add focused tests for absolute-path and traversal rejection.

Owned files:

- `tensor_logic/file_format.py`
- `tensor_logic/execution.py`
- `tensor_logic/http_api.py`
- `tests/test_tensor_logic_core.py`

### Task 2: Add API/workbench guardrails

Problem:

The local HTTP APIs and workbench subprocess runner have no auth, size limits, or timeouts. Defaults are localhost, but accidental public binding is too easy.

Acceptance criteria:

- Non-loopback host binding requires an explicit `--unsafe-public` flag or prints a high-visibility warning.
- POST body/source size is bounded.
- Workbench subprocess calls have a timeout.
- stdout/stderr response payloads are bounded.
- Timeout and oversized-body cases return deterministic HTTP errors.

Validation:

- `python3 -m pytest tests/test_web_workbench.py -q`
- Add handler/unit tests for timeout and body-size behavior without starting an external server.

Owned files:

- `web_workbench/server.py`
- `web_workbench/README.md`
- `tensor_logic/http_api.py`
- `tests/test_web_workbench.py`

### Task 3: Declare optional experiment dependency extras

Problem:

CI and `.[dev]` cover the reusable package, but several experiments require undeclared optional dependencies.

Acceptance criteria:

- Add documented extras for LM/SFT, vision/slot-attention, and remote/download experiments as appropriate.
- README documents which commands need which extra.
- A packaging test asserts the extras include expected package names.
- No heavy downloads occur in tests.

Validation:

- `python3 -m pytest tests/test_packaging_ci.py -q`

Owned files:

- `pyproject.toml`
- `README.md`
- `tests/test_packaging_ci.py`

### Task 4: Make remote package extraction safe

Problem:

`exp53` and `exp54` use archive `extractall()` after remote downloads.

Acceptance criteria:

- Tar and zip extraction reject absolute paths and paths escaping the target directory.
- Existing hardcoded package flows still work.
- Tests cover malicious archive member names without network.

Validation:

- Add a focused test file, then run `python3 -m pytest <new-test-file> -q`.

Owned files:

- `experiments/exp53_real_imports.py`
- `experiments/exp54_big_imports.py`
- New or existing tests under `tests/`

### Task 5: Add FAFSA safety and parity gate

Problem:

The FAFSA wizard is user-facing enough to need stronger disclaimers and official parity evidence.

Acceptance criteria:

- Wizard clearly states research/demo status, 2024-25 only, Formula A dependent-only, not financial advice.
- README or docs record the validation boundary.
- A tracked validation matrix distinguishes synthetic invariants from official worked examples.
- User-facing "likely qualify" wording is softened or gated behind parity status.

Validation:

- `python3 -m pytest tests/test_exp80.py -q`
- Add a text/behavior test for the disclaimer if the wizard remains interactive.

Owned files:

- `experiments/exp80_fafsa_wizard.py`
- `experiments/exp80_fafsa_kb.py`
- `tests/test_exp80.py`
- Optional docs file under `docs/`

### Task 6: Enforce run manifest and adapter pointer hygiene

Problem:

Run artifact persistence is documented after a lost adapter incident but not enforced.

Acceptance criteria:

- Add a checker for `experiments/exp*_data/results.json` files that require `git_sha` or explicit "legacy/no-manifest" annotation.
- If a result references an adapter/checkpoint, require a committed pointer file such as `ADAPTER.md`.
- CI can run the checker without downloading models or touching Kaggle.

Validation:

- `python3 -m pytest tests/test_packaging_ci.py -q` or a new `tests/test_run_protocol.py -q`.

Owned files:

- `docs/RUN_PROTOCOL.md`
- `tests/`
- Optional `tools/validate_run_protocol.py`

## Non-goals for this queue item

- No product-code edits.
- No generated result rewrites.
- No test-suite execution beyond environment/version/import probes.
- No remote downloads.
- No Kaggle, Hugging Face, Modal, Netlify, GitHub Actions, or other external service calls.
- No secret scanning beyond local grep patterns.
- No PR creation, push, merge, or external tracker updates.
- No attempt to verify scientific claims against external papers or package releases.
- No sibling-repo comparison; this repo was large enough for a focused local audit.

## Unknowns and blockers

- Whether maintainers intend the HTTP API/workbench to remain localhost-only forever or eventually support shared/team use.
- Whether `include` is intentionally a trusted local-file feature or should be sandboxed for all entrypoints.
- Whether committed experiment `results.json` files are canonical or merely snapshots from exploratory runs.
- Whether older Kaggle adapter datasets exist but are undocumented in this checkout.
- Whether FAFSA work is still research-only or intended to graduate to a user-facing tool.
- Whether CI currently passes on GitHub; this audit did not query GitHub or run the full suite.
- Whether ignored local files outside normal `git status --short` should be part of overnight review.

## Handoff

Files intentionally changed by this audit:

- `docs/overnight/tensor-experiments-risk-register.md`

Validation to run for the queue item:

- `git status --short`

Expected validation status:

- Before commit: one untracked report file under `docs/overnight/`.
- After an optional local report commit: clean tracked status.

Commit status:

- A local docs-only commit was attempted but blocked by sandbox permissions: `git add docs/overnight/tensor-experiments-risk-register.md` could not create the shared git metadata lock at `/Users/jwalinshah/projects/tensor/experiments/.git/worktrees/tensor-experiments-risk-register/index.lock`.
- Current commit remains `5565791a8c888511bbbff1107d2d357164f88baa`; the report is present as an untracked working-tree file.
