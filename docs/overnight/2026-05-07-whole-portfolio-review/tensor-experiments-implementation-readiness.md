# tensor-experiments implementation-readiness review

Date: 2026-05-07
Queue item: `tensor-experiments-implementation-readiness`
Branch: `codex/goal-tensor-experiments-implementation-readiness`
Review HEAD: `c9e2cfce206bb279a4348189726908fcb436d3dc`

## Scope and method

This pass reviewed the local `tensor-experiments` worktree only. It did not edit product code, contact external services, update trackers, push branches, or create a PR.

Repo-local evidence checked:

```bash
llm-tldr tree .
git status --short --branch
git rev-parse HEAD
git branch --show-current
git ls-files | rg '^(\.github/|docs/overnight/|runs/|items/|AGENTS\.md$|CLAUDE\.md$|README\.md$|pyproject\.toml$)'
llm-tldr search "pytest|test|lint|validation|overnight|CI|workflow|ruff|black|mypy" .
python3 tools/code_index.py --dump
rtk grep "TODO|FIXME|Known limitation|NotImplemented|skip|xfail|argparse|__main__|TODO" .
rtk grep "exp84|exp85|exp86|exp87|exp88|exp89|exp90|exp91|exp92|exp93|exp94|exp95" notes/EXPERIMENTS.md
rg --files experiments | rg 'results(_quick)?\.json$'
jq -r 'input_filename + "\t" + ((.experiment // .name // "unknown")|tostring) + "\t" + ((.passed // .pass // .gate_passed // .success // .summary.pass // "")|tostring) + "\t" + ((.metrics // .summary // .results // {})|keys|join(","))' experiments/exp8*_data/results*.json experiments/exp9*_data/results*.json
```

No tracked prior `docs/overnight/`, `runs/`, or `items/` artifacts were present in this worktree before this report. The tracked control/config files found were `.github/workflows/ci.yml`, `CLAUDE.md`, `README.md`, and `pyproject.toml`.

## Implementation-readiness observations

1. `README.md` frames this as both a learning repo and the shared `tensor_logic` library dependency for sibling projects, so implementation work can affect downstream users even when it looks like an experiment-only change.
2. `README.md` documents the worker validation contract: install with `python -m pip install -e ".[dev]"`, then run `python -m pytest tests/ -v`; it also gives the support fast path and keeps `exp86` neural baseline smoke out of CI unless it stays quick and deterministic.
3. `pyproject.toml` packages only `tensor_logic*`, requires Python `>=3.11`, has a single runtime dependency on `torch>=2.0`, and puts `pytest`, `numpy`, and `matplotlib` in the `dev` extra. This makes new experiment scripts importable from tests but not package exports unless moved under `tensor_logic`.
4. `.github/workflows/ci.yml` runs on PRs and `main` pushes, sets up Python 3.11, installs `.[dev]`, and runs `python -m pytest tests/ -v`. There is CI coverage, but no separate result-manifest audit or quick-script smoke job.
5. `CLAUDE.md` requires `python tools/code_index.py --lookup <RelevantSymbol>` before implementation planning that touches `tensor_logic/`. In this environment `python` was not found, while `python3 tools/code_index.py --dump` worked. Future tickets should use the repo's documented command but may need to normalize interpreter expectations.
6. `CLAUDE.md` records a concrete exp78 backend limitation: VLM-style MLX model names can be incompatible with `mlx_lm.load()` and should use text-only models or transformers fallback. That is a ready blocker note for any rule-induction LM backend ticket.
7. `docs/RUN_PROTOCOL.md` is strong on remote experiment provenance: output dir per run, `manifest.json`, `failures.jsonl`, adapter persistence through committed Kaggle notebook versions, and local pointer files under `experiments/expN_data/`. This should be reused for any future remote or adapter-producing work.
8. `docs/SYMPHONY_RUN_PROTOCOL.md` is implementation-ready for normal workers: one issue, one branch, one PR, smallest validation, PR URL plus commit SHA in the handoff. This queue item intentionally stayed read-only except for this report.
9. `docs/superpowers/plans/2026-05-05-support-stability.md` is the clearest current productized research plan. It names the V1 support/stability claim, non-goals, rule sketch, required neural baselines, falsification gate, test contract, and vertical slices.
10. `tests/test_packaging_ci.py` already guards the packaging/CI/docs contract by asserting `pyproject.toml` dependencies, `.github/workflows/ci.yml` commands, README validation commands, and the Symphony PR handoff language.
11. `tests/test_exp84_support_data.py` has implementation-ready generator invariants: deterministic generation, ID/OOD object-count ranges, non-overlap, intervention retraction, deep-copy safety, and unknown removal target rejection.
12. `tests/test_exp85_support_tl.py` has strong TL support-engine tests: hand-built near misses, stable/falls proofs, branching full-width union, generator parity, removal retraction, and tolerant contact/horizontal extraction.
13. `experiments/exp87_support_eval.py` is a complete V1 gate harness. It builds ID, larger OOD, deeper OOD, branching OOD, and counterfactual splits, evaluates TL plus MLP/DeepSets, raises on TL deterministic or counterfactual accuracy below 100%, and records the 10pp OOD margin gate.
14. `tests/test_exp87_support_eval.py` verifies the required split coverage and result schema, including TL gates and neural metric shapes. This is a good pattern for future experiments that write `results.json`.
15. `experiments/exp88_support_noisy_relations.py` through `experiments/exp95_scored_object_hypotheses.py` all expose `--quick` modes and write `results_quick.json` or `results.json` under their experiment data directories. Their tests call the underlying run functions with temporary outputs, so schema regressions are covered without committing regenerated data.
16. `notes/EXPERIMENTS.md` has the freshest scientific evidence. Exp87 confirms the perfect-state support substrate, exp88-93 map brittleness and calibrated uncertainty, exp94 shows an oracle object-hypothesis upper bound, and exp95 shows non-oracle false-positive ranking is useful while missing/merge identity and cardinality remain unsafe.
17. `experiments/exp95_scored_object_hypotheses_data/results.json` is the current full-run artifact for the latest lane. The notes record the key gap: false-positive stress improves materially, but missing-object and merge recovery still trail the oracle and can introduce accepted wrong labels.
18. `tools/code_index.py` and `tests/test_code_index.py` provide a local AST signature index for `tensor_logic/`. `tests/test_code_index.py` also asserts `tools/index.json` is gitignored, which is correct for generated local planning state.
19. `web_workbench/server.py`, `web_workbench/static/app.js`, and `tests/test_web_workbench.py` give a small interactive surface over the package. Tests verify query API smoke behavior, CLI/HTTP why-not JSON parity, and bundled sample TL validity.
20. Current git history shows this branch at `origin/main` with recent merged support-lane PRs for exp92-95. There was no dirty product-code state before this report.

## Risks and blockers

- Interpreter mismatch: repo docs and CI use `python`, but this local shell only resolved `python3` during review. Any local worker issue should either run inside the intended venv or make the validation command explicit.
- The full validation command depends on `torch`, `numpy`, `matplotlib`, and `pytest` from `.[dev]`. This review did not install dependencies or run the full suite because the queue validation is `git status --short`.
- The latest support lane is implementation-ready, but exp95 is not claim-ready for missing/merge structural recovery. It needs an abstention policy and more identity/cardinality evidence before it can safely become a broader pixel-facing result.
- CI proves tests, but not that committed `experiments/*_data/results.json` files match their corresponding scripts and notes. Result drift is an easy way for claims to go stale.
- Pixel-facing experiments still use synthetic segmentation/detector stubs. Follow-up work should avoid implying real-world perception robustness until a real detector or a more faithful detector-error model is introduced.
- `docs/RUN_PROTOCOL.md` is strongest for Kaggle adapter work, but exp87-95 local CPU result files do not have a uniform manifest checker. A small audit tool would make future generated artifacts safer to update.
- `CLAUDE.md` contains a VLM backend limitation for exp78. Any ticket involving LM/VLM backend selection should carry that blocker forward instead of rediscovering it.
- No previous overnight report or runner handoff existed locally for this queue item. Morning review should treat this file as the first implementation-readiness artifact for `tensor-experiments`.

## Exact validation commands

Required validation for this queue item:

```bash
git status --short
```

Documented full repo validation:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

Support/stability fast path from README and Symphony protocol:

```bash
python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v
python experiments/exp86_support_baselines.py --quick
```

Current exp87-95 quick-script checks that would be useful before touching generated result artifacts:

```bash
python experiments/exp87_support_eval.py --quick
python experiments/exp88_support_noisy_relations.py --quick
python experiments/exp89_support_primitive_confidence.py --quick
python experiments/exp90_support_repair_sweep.py --quick
python experiments/exp91_interval_support_uncertainty.py --quick
python experiments/exp92_pixel_abstain_recover.py --quick
python experiments/exp93_detector_calibration_stress.py --quick
python experiments/exp94_object_hypothesis_layer.py --quick
python experiments/exp95_scored_object_hypotheses.py --quick
```

Use `python3` in this local shell if `python` is unavailable.

## Implementation-ready follow-up tasks

### 1. Add confidence-gated acceptance for exp95 non-oracle hypotheses

Owned files:

- `experiments/exp95_scored_object_hypotheses.py`
- `tests/test_exp95_scored_object_hypotheses.py`
- `notes/EXPERIMENTS.md` only if a full run updates the claim

Acceptance criteria:

- Scored false-positive drops can still be accepted when ranked confidently.
- Missing-support and merge-split hypotheses abstain instead of accepting when rank margin or identity evidence is weak.
- Result summaries include policy counters: accepted, abstained, accepted wrong, false-stable accepted, and recovery gap vs oracle.
- Unit tests cover one missing, one merge, and one false-positive scene without relying on oracle IDs.

Smallest useful validation:

```bash
pytest tests/test_exp95_scored_object_hypotheses.py -v
python experiments/exp95_scored_object_hypotheses.py --quick
```

### 2. Add identity/cardinality evidence features for missing and merge hypotheses

Owned files:

- `experiments/exp92_pixel_abstain_recover.py`
- `experiments/exp93_detector_calibration_stress.py`
- `experiments/exp95_scored_object_hypotheses.py`
- `tests/test_exp92_pixel_abstain_recover.py`
- `tests/test_exp93_detector_calibration_stress.py`
- `tests/test_exp95_scored_object_hypotheses.py`

Acceptance criteria:

- The detector/stress table exposes non-oracle evidence useful for object count anomalies, such as unmatched pixel area, compound-box aspect/height signals, and per-candidate support consistency.
- Exp95 ranking can consume those features without reading `affected_ids`, `source_ids`, or `false_positive_ids`.
- Tests prove missing/merge candidates can be ranked or abstained from evidence, not true source identity.

Smallest useful validation:

```bash
pytest tests/test_exp92_pixel_abstain_recover.py tests/test_exp93_detector_calibration_stress.py tests/test_exp95_scored_object_hypotheses.py -v
```

### 3. Add a result-artifact audit tool for exp87-95

Owned files:

- `tools/results_audit.py`
- `tests/test_results_audit.py`
- `README.md`

Acceptance criteria:

- The tool validates all committed `experiments/exp8*_data/results*.json` and `experiments/exp9*_data/results*.json` files for `experiment`, `quick`, `results_path`, `config`, and required top-level metric sections.
- The tool catches a mismatched `results_path` or missing `experiment` field.
- README documents the audit command next to worker validation.

Smallest useful validation:

```bash
pytest tests/test_results_audit.py -v
python tools/results_audit.py
```

### 4. Add CI coverage for support-lane quick-script smoke runs

Owned files:

- `.github/workflows/ci.yml`
- `tests/test_packaging_ci.py`
- `README.md`

Acceptance criteria:

- CI still runs the full test suite.
- CI or a separate optional job runs at least the deterministic quick scripts for exp87 and exp95 with temporary output paths.
- `tests/test_packaging_ci.py` asserts the new CI commands so validation docs and CI do not drift.
- README explains which quick scripts are CI-safe and which full runs are manual.

Smallest useful validation:

```bash
pytest tests/test_packaging_ci.py -v
```

### 5. Normalize local interpreter and code-index validation docs

Owned files:

- `CLAUDE.md`
- `README.md`
- `docs/SYMPHONY_RUN_PROTOCOL.md`
- `tests/test_packaging_ci.py`
- `tests/test_code_index.py`

Acceptance criteria:

- Worker docs specify whether `python`, `python3`, or a venv-provided interpreter is required.
- Code-index planning examples remain executable in a clean local shell or explicitly state the venv prerequisite.
- Tests continue to prove `tools/code_index.py --lookup Program` works through `sys.executable`.

Smallest useful validation:

```bash
pytest tests/test_packaging_ci.py tests/test_code_index.py -v
python3 tools/code_index.py --lookup Program
```

## Handoff

Changed files:

- `docs/overnight/2026-05-07-whole-portfolio-review/tensor-experiments-implementation-readiness.md`

Commit SHA reviewed:

- `c9e2cfce206bb279a4348189726908fcb436d3dc`

Validation result:

- `git status --short` exited 0 and reported the expected untracked report directory: `?? docs/overnight/`.

PR URL:

- Not created. External pushes and PR creation are out of scope for this queue item.

Blockers:

- None for the report.
- Full tests were not required by the queue item and were not run.
