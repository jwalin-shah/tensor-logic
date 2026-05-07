# tensor-experiments workflow-handoff audit

Queue item: `tensor-experiments-workflow-handoff`
Date: 2026-05-07
Repo path: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-experiments-workflow-handoff`
Focus area: workflow handoff

## Scope and outcome

This audit is read-only against product code. The only intended repository change is this report.

The repo is a research project plus reusable `tensor_logic` package. `README.md` frames it as "a learning project, not a product", while also noting that `tensor_logic/` is now a shared library dependency for sibling projects such as `fafsa-engine` and future `officeqa-*` work. The practical workflow problem is that the repo has moved from one-off demos into a mixed surface: package code, tests, web workbench, local code-index tooling, heavy LM/GPU experiments, generated experiment artifacts, and external run protocols.

Morning-review conclusion: do not hand future workers a broad "continue experiments" task. The next safe work should be split into narrow handoff/validation slices that make experiment ownership, output paths, dependencies, and proof commands explicit before any more experiment code or remote runs happen.

## Local state

- Branch: `codex/goal-tensor-experiments-workflow-handoff`
- Pre-audit HEAD: `5565791a8c888511bbbff1107d2d357164f88baa`
- Remote: `origin https://github.com/jwalin-shah/tensor-logic.git`
- Initial `git status --short`: clean.
- After local proof commands and before this report, `git status --short`: clean.
- After local proof commands and before this report, `git status --short --ignored`: ignored `.pytest_cache/` and ignored `tools/index.json`.
- `docs/overnight/` did not exist before this queue item.
- Local shell has `python3` (`Python 3.12.8`) but not `python`; `python tools/code_index.py --status` failed with `python: command not found`.

## Evidence ledger

1. `llm-tldr tree .` showed the main surfaces: `tensor_logic/`, `experiments/`, `demos/`, `phase_training/`, `tests/`, `web_workbench/`, `tools/`, `docs/`, and `notes/`.
2. `README.md` documents purpose, headline experiment claims, worker validation, runnable demos, and the reusable `tensor_logic/` extraction.
3. `pyproject.toml` declares package name `tensor-logic`, Python `>=3.11`, core dependency `torch>=2.0`, and dev deps `matplotlib`, `numpy`, `pytest`.
4. `.github/workflows/ci.yml` installs with `python -m pip install -e ".[dev]"` and runs `python -m pytest tests/ -v` on pull requests and pushes to main.
5. `tests/test_packaging_ci.py` asserts the dependency contract, CI command, and README worker validation command, so validation docs are intentionally under test.
6. `CLAUDE.md` instructs agents to run `python tools/code_index.py --lookup <RelevantSymbol>` before planning changes under `tensor_logic/`, and records a known MLX/VLM limitation for `experiments/exp78_rule_induction.py`.
7. `tools/code_index.py` writes `tools/index.json` through `ensure_fresh()` during lookup/dump/rebuild; `.gitignore` excludes `tools/index.json`, so this is intentional local-only state.
8. `python3 tools/code_index.py --status` returned exit 1 with `stale: index.json is missing or older than source files` before any code-index lookup.
9. `docs/RUN_PROTOCOL.md` exists because an exp76c Kaggle adapter was lost when `/kaggle/working/` was wiped; it mandates per-run output dirs, manifests, failure dumps, notebook commit, and local pointer files.
10. `docs/SYMPHONY_RUN_PROTOCOL.md` defines a worker handoff contract: one issue, one branch, one PR, smallest validation, and recorded PR URL/commit SHA/validation/blockers.
11. `experiments/exp79_self_play_loop.py` defaults `--out` to `experiments/exp79_data/results.json`.
12. `experiments/exp79_lewm_tl.py` also uses `DATA_DIR = experiments/exp79_data` and writes `results.json`, `complexity_curve.png`, and ignored `.pt` weights there, creating an artifact namespace collision.
13. `experiments/exp79_data/results.json` currently contains self-play loop output modes (`easy`, `medium`, `hard`, `very_hard`, `adversarial`), not the LeWM+TL result schema from `experiments/exp79_lewm_tl.py`.
14. `experiments/exp83_slot_attention.py` imports `scipy.optimize.linear_sum_assignment`, but `pyproject.toml` does not declare `scipy` in core or dev dependencies.
15. `experiments/exp83_slot_data/results.json` records `probe=false`, `tl_only=true`, `e2e=true`; `notes/RESEARCH_NOTES.md` says exp82/exp83 should remain exploratory until they have clean specs and falsification gates.
16. `experiments/exp80_fafsa_kb.py` states Formula B/C for independent students is TODO; tests cover Formula A synthetic cases only.
17. `rg -n "os.environ|getenv|OPENAI|ANTHROPIC|HF_|KAGGLE|TOKEN|API_KEY|MODEL|mlx|transformers" .` found external/provider-shaped docs and optional LM paths, but no checked-in secret values.
18. `rtk pytest tests/test_packaging_ci.py -q`, `rtk pytest tests/test_code_index.py -q`, and `rtk pytest tests/test_web_workbench.py -q` all returned `Pytest: No tests collected`; raw `python3 -m pytest ...` did collect and pass the same files.

## Workflow map

### Package and CLI

`tensor_logic/` is the reusable library. `tensor_logic/__main__.py` provides CLI subcommands for `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`. `tensor_logic/http_api.py` and `web_workbench/server.py` expose browser/API execution paths that eventually call the same local CLI/runtime.

Safe worker rule: any task that touches `tensor_logic/` should start with a code-index lookup using `python3 tools/code_index.py --lookup <symbol>` in this local environment, because `python` is not available here. Expect that lookup to create or refresh ignored `tools/index.json`.

### Tests and cheap validation

The canonical documented full validation is:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

In this local shell, the equivalent command must use `python3` unless the worker creates a `python` alias or virtualenv. The cheap observed proof commands were:

```bash
python3 -m pytest tests/test_packaging_ci.py -q
python3 -m pytest tests/test_code_index.py -q
python3 -m pytest tests/test_web_workbench.py -q
```

### Experiment lanes

- `experiments/exp78_rule_induction.py`: TL-only rule induction, optionally LM-pruned. The no-LM path is local and cheap; the `--lm` path may load MLX/transformers models.
- `experiments/exp79_self_play_loop.py`: validated rule-factory/self-play loop. Writes `experiments/exp79_data/results.json` by default.
- `experiments/exp79_lewm_tl.py`: pixel/JEPA/probe/TL experiment. Also writes `experiments/exp79_data/`, which currently conflicts with self-play output.
- `experiments/exp80_fafsa_kb.py`: local Formula A FAFSA SAI proof trace. High-stakes domain; must stay framed as a research prototype and not a complete FAFSA engine.
- `experiments/exp81_optimize_rule_induction.py`: optimize-loop plus LM-guided proposer. Unit tests cover parse/evaluator pieces, but `test_make_proposer_returns_valid_json` can call `lm_prune()`, which may try model loading unless transformers is unavailable.
- `experiments/exp83_slot_attention.py`: exploratory slot-attention follow-up. Current tracked result fails the probe gate, and the script requires undeclared `scipy`.
- `experiments/exp60d_sft.py` and exp60/76 data: heavier SFT/tool-harness lane with optional `transformers`, `peft`, `datasets`, and `accelerate`; use `docs/RUN_PROTOCOL.md` before any remote run.

## Risks and stale assumptions

1. The validated worker command uses `python`, but this local environment only has `python3`. A worker following `README.md`, `.github/workflows/ci.yml`, or `CLAUDE.md` literally will fail before installing or testing.
2. The global preferred wrapper `rtk pytest` did not collect explicit test files as used here, while raw `python3 -m pytest` did. Future handoffs should name raw pytest commands when exact local proof matters.
3. `tools/index.json` is gitignored and local-only. `--status` fails on a clean/missing index, but `--lookup` can write ignored state. That is useful for agents, but it means code-index freshness is not represented by git state.
4. Two different exp79 scripts share the same output path. Running `experiments/exp79_lewm_tl.py` can overwrite the self-play `experiments/exp79_data/results.json` that is currently tracked and cited in `notes/EXPERIMENTS.md`.
5. Optional/heavy dependencies are not represented in `pyproject.toml`: `scipy` for exp83, `transformers`/`mlx_lm` for exp78/81 LM paths, and `peft`/`datasets`/`accelerate` for exp60d SFT.
6. Several experiment scripts write tracked result locations by default. A worker can accidentally mutate committed data by running an experiment without `--out /tmp/...` or a namespaced output dir.
7. README/doc claims have stale edges: repo layout says `experiments/ exp1..exp54` while the repo now contains exp83; the demo count says five headline runnable demos while `demos/` has eight files.
8. `experiments/exp80_fafsa_kb.py` cites a 2024-25 source and explicitly omits Formula B/C. Any next FAFSA task needs human product/legal framing before it becomes user-facing or claims coverage beyond Formula A.
9. External run protocol risk is proven by history: `docs/RUN_PROTOCOL.md` exists because a Kaggle adapter was lost. Remote jobs must not be launched without manifest, output path, notebook/dataset persistence, and explicit approval.
10. `docs/agents/domain.md` says to read `CONTEXT.md`, `CONTEXT-MAP.md`, and `docs/adr/` if present; none were found. Future design decisions are therefore scattered across `notes/` and `docs/superpowers/`, not a central context/ADR layer.

## Next safe Work Pack / issue tasks

### Task 1: Create an experiment workflow manifest

Purpose: give future agents a single, local, non-chat contract for what each active experiment is allowed to run or write.

Owned files:
- Add `docs/EXPERIMENT_WORKFLOW.md` or `docs/experiment-manifest.md`.
- Optionally update `README.md` to link to it.
- Add a small packaging/docs test if the README link becomes a maintained contract.

Acceptance criteria:
- Manifest lists at least exp78, exp79_self_play_loop, exp79_lewm_tl, exp80, exp81, exp83, exp60d, and exp76c.
- Each row names entrypoint, default output paths, tracked vs ignored artifacts, local cheap validation, heavy/remote validation, required optional deps, and stop conditions.
- The manifest tells workers to prefer `/tmp/...` outputs unless the issue explicitly asks to update committed result artifacts.
- No product code or experiment logic changes.

Validation:
- `python3 -m pytest tests/test_packaging_ci.py -q`
- `git status --short`

Stop conditions:
- If maintainers need to choose a canonical experiment naming scheme first, stop and ask. Do not rename files in this task.

### Task 2: Fix the exp79 artifact namespace collision

Purpose: prevent one experiment run from overwriting another experiment's committed result artifact.

Owned files:
- `experiments/exp79_lewm_tl.py`
- `experiments/exp79_self_play_loop.py` only if maintainers choose to rename both sides.
- `tests/` for a lightweight output-path regression test.
- `docs/superpowers/plans/2026-04-28-exp79-lewm-tl.md` or the new workflow manifest if it exists.

Acceptance criteria:
- `exp79_self_play_loop.py` and `exp79_lewm_tl.py` no longer write the same default `results.json`.
- Existing tracked self-play results remain intact or are migrated with an explicit manifest note.
- The LeWM `.pt` weights stay ignored and are not committed.
- A test asserts the two default output directories/files are distinct without running training.

Validation:
- `python3 -m pytest tests/test_exp79.py tests/test_packaging_ci.py -q`
- If a new handoff test is added, include it directly in the pytest command.
- `git status --short --ignored` should show no new tracked generated weights.

Stop conditions:
- If historical result path compatibility matters for notes/blog posts, ask before migrating committed artifacts.

### Task 3: Split optional dependency extras by workflow lane

Purpose: make local validation predictable and prevent "import surprise" failures in exploratory scripts.

Owned files:
- `pyproject.toml`
- `tests/test_packaging_ci.py`
- New or updated docs in the experiment workflow manifest.

Acceptance criteria:
- Core package remains minimal.
- Add explicit extras such as `vision` for exp83 (`scipy`), `lm` for exp78/81 LM proposer paths (`transformers`, maybe `mlx_lm` documented as platform-specific), and `sft` for exp60d (`transformers`, `peft`, `datasets`, `accelerate`).
- Tests assert the extras exist and that README/manifest names the install command for each lane.
- Unit tests must not require downloading model weights.

Validation:
- `python3 -m pytest tests/test_packaging_ci.py -q`
- `python3 -m pytest tests/test_exp80.py -q`
- `python3 -m pytest tests/test_web_workbench.py -q`

Stop conditions:
- If exact LM/MLX version pins require current upstream compatibility research, stop and create a separate dependency-refresh issue.

### Task 4: Normalize worker validation docs for local and CI Python

Purpose: remove the `python` vs `python3` handoff footgun without changing CI behavior unnecessarily.

Owned files:
- `README.md`
- `CLAUDE.md`
- `.github/workflows/ci.yml` only if maintainers want CI to mirror local commands.
- `tests/test_packaging_ci.py`

Acceptance criteria:
- Docs explain that CI uses `python` from `actions/setup-python`, while local macOS workers may need `python3`.
- The worker validation section includes a local-safe command block using `python3`.
- `tests/test_packaging_ci.py` is updated to assert the intended docs contract instead of only the old string.

Validation:
- `python3 -m pytest tests/test_packaging_ci.py -q`
- `python3 -m pytest tests/test_code_index.py -q`

Stop conditions:
- If the repo standard is to require virtualenvs that provide `python`, document that instead of changing all examples.

### Task 5: Add a no-write smoke lane for active experiments

Purpose: give future agents a safe command that proves import/CLI shape without mutating tracked result files or starting training.

Owned files:
- `tests/test_experiment_entrypoints.py` or similar.
- Experiment scripts only for adding `--help`, `--dry-run`, or `--out` support where missing.

Acceptance criteria:
- Smoke tests cover `--help` or dry-run for exp78, exp79_self_play_loop, exp79_lewm_tl, exp80, exp81, and exp83.
- Tests do not train models, download model weights, call external services, or write tracked result paths.
- Scripts that currently lack a no-write mode get the smallest CLI addition needed.

Validation:
- `python3 -m pytest tests/test_experiment_entrypoints.py -q`
- `python3 -m pytest tests/test_packaging_ci.py -q`

Stop conditions:
- If adding no-write flags to several scripts becomes broad, split by experiment family.

## Validation command candidates

| Command | Observed / expected status | Notes |
|---|---:|---|
| `git status --short` | Required queue validation; exit 0 | Initially clean. After this report, should show only the report until committed. |
| `python3 -m pytest tests/test_packaging_ci.py -q` | Observed pass: 3 passed | Proves package/README/CI worker validation contract tests. |
| `python3 -m pytest tests/test_code_index.py -q` | Observed pass: 15 passed | Creates ignored `tools/index.json` during CLI status freshness test. |
| `python3 -m pytest tests/test_web_workbench.py -q` | Observed pass: 2 passed in 6.18s | Proves workbench/CLI/API parity smoke. |
| `python -m pytest tests/ -v` | Expected local fail unless `python` exists | Documented CI command, but local shell has no `python`. |
| `python3 -m pytest tests/ -v` | Expected full-suite candidate, not run in this audit | Safe next validation after dependencies are installed; may take longer than narrow proofs. |
| `python3 tools/code_index.py --status` | Observed fail before index refresh | Exit 1 because ignored index is missing/stale. Use `--lookup <symbol>` for agent planning. |
| `rtk pytest tests/test_packaging_ci.py -q` | Observed misleading no-collection output | Prefer raw `python3 -m pytest` in handoffs for this repo. |
| `python3 experiments/exp78_rule_induction.py --out /tmp/exp78-results.json --n-pos 5 --n-neg 5` | Expected cheap no-LM experiment candidate, not run | Use `/tmp` output to avoid tracked artifact mutation. |
| `python3 experiments/exp79_self_play_loop.py --mode easy --out /tmp/exp79-self-play.json` | Expected local experiment candidate, not run | Avoid default tracked `experiments/exp79_data/results.json`. |

## Non-goals for this queue item

- No product code edits.
- No experiment reruns that write committed data.
- No remote jobs, Kaggle notebooks, datasets, deploys, pushes, PRs, or external tracker updates.
- No dependency upgrades.
- No validation of scientific or FAFSA claims against external sources.
- No sibling-repo comparison; this repo is large enough that the workflow-handoff audit had sufficient local depth.

## Unknowns and blockers

- Unknown whether maintainers want exp79 naming preserved for historical continuity or split into `exp79_self_play_data` and `exp79_lewm_data`.
- Unknown whether the preferred local Python contract is "use `python3`" or "always activate a venv that provides `python`".
- Unknown whether optional heavy dependencies should be version-pinned now or only documented per lane.
- Unknown whether `CONTEXT.md`/ADR docs should be introduced before more architectural changes.
- No blockers to writing this report. Creating a local commit from this worker is blocked by sandbox permissions because this worktree's Git metadata resolves outside the writable root.

## Final handoff notes

- Changed file: `docs/overnight/tensor-experiments-workflow-handoff.md`
- Product code changed: no
- External services used: no
- PR created: no
- Commit created: no. Current HEAD remains `5565791a8c888511bbbff1107d2d357164f88baa`.
- Required validation command: `git status --short`
- Required validation result after report write: exit 0, output `?? docs/overnight/`
- Commit blocker: `git add docs/overnight/tensor-experiments-workflow-handoff.md` failed with `fatal: Unable to create .../index.lock: Operation not permitted`.
- Main workflow blocker found: workflow ambiguity, not local execution. The highest-risk ambiguity is exp79 artifact ownership.
