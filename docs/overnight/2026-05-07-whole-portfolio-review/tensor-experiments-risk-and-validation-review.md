# tensor-experiments risk-and-validation review

Queue item: `tensor-experiments-risk-and-validation-review`  
Branch: `codex/goal-tensor-experiments-risk-and-validation-review`  
Evidence HEAD before report: `c9e2cfce206bb279a4348189726908fcb436d3dc`  
Review date: 2026-05-07

## Scope

This is a read-only risk and validation review of the local `tensor-experiments`
worktree, plus this report file. I did not edit product code, create or merge a
PR, push, deploy, run external services, or update trackers.

No previous local overnight reports, `runs/` handoffs, or queue `items/` were
present in this checkout when searched with `rg --hidden --files`.

## Validation Evidence

Commands run:

```bash
llm-tldr tree .
git branch --show-current
git rev-parse HEAD
git status --short
rg --hidden --files -g '.github/**' -g 'docs/overnight/**' -g 'runs/**' -g 'items/**'
python -m pytest tests/test_packaging_ci.py -q
python3 -m pytest tests/test_packaging_ci.py -q
python3 -m pytest --collect-only -q
python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -q
python3 -m pytest tests/test_exp87_support_eval.py -q
python3 -m pytest tests/test_exp88_support_noisy_relations.py tests/test_exp89_support_primitive_confidence.py tests/test_exp90_support_repair_sweep.py tests/test_exp91_interval_support_uncertainty.py tests/test_exp92_pixel_abstain_recover.py tests/test_exp93_detector_calibration_stress.py tests/test_exp94_object_hypothesis_layer.py tests/test_exp95_scored_object_hypotheses.py -q
```

Observed results:

- `git status --short` was clean before writing this report.
- `python -m pytest tests/test_packaging_ci.py -q` could not run locally because
  `python` is not on PATH in this worktree shell.
- `python3 -m pytest tests/test_packaging_ci.py -q` passed: `4 passed in 0.02s`.
- `python3 -m pytest --collect-only -q` collected 198 tests in 19.64s.
- `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -q`
  passed: `16 passed in 0.05s`.
- `python3 -m pytest tests/test_exp87_support_eval.py -q` passed:
  `2 passed in 2.28s`.
- The exp88-exp95 focused batch passed: `34 passed in 2.69s`.

The queue validation command to rerun after this report is:

```bash
git status --short
```

## Concrete Observations

1. `README.md` documents the worker validation contract as
   `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`, plus a
   support/stability fast path for `tests/test_exp84_support_data.py`,
   `tests/test_exp85_support_tl.py`, and `python experiments/exp86_support_baselines.py --quick`.
2. `CLAUDE.md` repeats the repo-local test command `pytest tests/ -v` and
   records a known exp78 limitation: VLM-style MLX model names can be
   incompatible with `mlx_lm.load()`.
3. `pyproject.toml` declares only `torch` as a runtime dependency, with
   `matplotlib`, `numpy`, and `pytest` in the `dev` extra; LLM experiment imports
   such as `transformers`, `mlx_lm`, `peft`, and `datasets` are not represented
   as installable extras.
4. `.github/workflows/ci.yml` exists and runs on pull requests and pushes to
   `main`, sets up Python 3.11, installs `".[dev]"`, and runs
   `python -m pytest tests/ -v`.
5. `tests/test_packaging_ci.py` explicitly checks the packaging and CI contract:
   pyproject dependencies, workflow triggers, editable dev install, full pytest
   command, README validation text, and `docs/SYMPHONY_RUN_PROTOCOL.md`.
6. `docs/RUN_PROTOCOL.md` requires experiment manifests with config, metrics,
   git SHA, timestamp, failure dumps, and result pointers, and says no experiment
   should run without a pre-written spec containing a falsification criterion.
7. `docs/superpowers/plans/2026-05-05-support-stability.md` defines the V1 gate:
   TL must hit 100% deterministic and counterfactual accuracy and beat the best
   neural baseline by at least 10pp on larger/deeper OOD stacks.
8. `experiments/exp87_support_eval.py` computes the 10pp OOD neural-margin gate
   and writes `thesis: "passed"` or `"falsified"`, but only raises hard failures
   for TL deterministic and counterfactual accuracy, not for a failed OOD margin.
9. `experiments/exp87_support_data/results.json` records the clean V1 pass:
   100% TL deterministic label accuracy on 3,438 labels, 100% counterfactual
   accuracy on 1,709 labels, and OOD margins of +16.3pp on larger OOD and
   +30.8pp on deeper OOD versus DeepSets.
10. `experiments/exp88_support_noisy_relations.py` and
    `experiments/exp88_support_noisy_relations_data/results.json` show the clean
    support engine is not perception-robust: first non-zero XY jitter
    `delta=0.0001` drops aggregate accuracy to about 52.7%; a 0.001 tolerance
    improves microscopic jitter but first drops by `delta=0.001`.
11. `experiments/exp92_pixel_abstain_recover.py` and
    `experiments/exp92_pixel_abstain_recover_data/results.json` show a useful
    abstain contract but also a zero-noise pixel quantization caveat: hard TL is
    90.7% accurate at `delta=0`, while interval feasibility gives 100% accepted
    accuracy at 90.7% coverage.
12. `experiments/exp93_detector_calibration_stress.py` explicitly models
    coordinate, missing, merge, and false-positive detector failures; its result
    artifact shows structural detector failures need a detector-health/object
    hypothesis signal rather than just coordinate bands.
13. `experiments/exp94_object_hypothesis_layer.py` is an oracle-style upper
    bound using simulated structural anomaly metadata; `notes/EXPERIMENTS.md`
    states this caveat directly and points to non-oracle ranking as the next
    gate.
14. `experiments/exp95_scored_object_hypotheses.py` removes several oracle
    identity assumptions, but `experiments/exp95_scored_object_hypotheses_data/results.json`
    and `notes/EXPERIMENTS.md` show large residual risk: missing-object scoring
    can be worse than observed-naive, merge recovery is limited, and false-stable
    accepted errors still occur.
15. `tests/test_exp88_support_noisy_relations.py` through
    `tests/test_exp95_scored_object_hypotheses.py` cover schema, zero-noise, and
    targeted unit cases, but they do not enforce full-run result freshness,
    manifest provenance, or acceptance thresholds for the exp95 non-oracle gate.
16. `notes/EXPERIMENTS.md` is well maintained through exp95 and contains the
    strongest current risk language: exp87 is a clean perfect-state substrate
    result, exp88-exp93 are robustness/perception boundary maps, exp94 is an
    oracle upper bound, and exp95 is partial with identity/cardinality gaps.
17. `.gitignore` excludes generated caches, model weights, `.cocoindex_code/`,
    and `tools/index.json`, matching `CLAUDE.md`'s warning that the local code
    index is generated and should not be hand-edited.
18. `tools/code_index.py` can rebuild `tools/index.json` on lookup/status, so
    planning agents touching `tensor_logic/` have a local API signature guard,
    but that guard is not part of the queue validation command.

## Risks And Blockers

- The required queue validation command, `git status --short`, only proves
  worktree state. It does not prove the report quality or the repo's executable
  validation contracts.
- Local shell portability is weak: the documented `python -m ...` commands fail
  here because only `python3` was available. CI's setup-python environment
  should provide `python`, but local worker docs should state the precondition or
  provide a `python3`/`uv` path.
- Result artifacts for exp87-exp95 can go stale silently. The scripts write
  `results.json`/`results_quick.json` by default, but the committed artifacts do
  not consistently include git SHA, run timestamp, command, or a freshness check
  against current code.
- The exp87 V1 gate is not fully hard-failing: if TL still hits 100% but the
  neural margin falls below 10pp, the script records `thesis: "falsified"` but
  does not raise.
- The support/perception story is easy to overclaim. Evidence currently supports
  a perfect-state substrate claim and an abstain/recover interface direction,
  not a production pixel model, real-world physics engine, or learned detector.
- The exp95 non-oracle object-hypothesis layer is not yet ready to become a
  positive claim. It has a real false-positive foothold, but missing/merge
  identity gaps and false-stable accepted errors remain.
- LLM-heavy older experiments are not covered by installable optional extras,
  so fresh workers can install `".[dev]"` and still fail when trying exp36,
  exp60d, exp77, or exp78 LM paths.
- No previous local overnight handoff/report existed for this queue item, so
  there was no prior generated review to reconcile.

## Implementation-Ready Follow-Up Tasks

### 1. Make the exp87 OOD margin a hard gate

Owned files:

- `experiments/exp87_support_eval.py`
- `tests/test_exp87_support_eval.py`

Acceptance criteria:

- `run_evaluation()` exits nonzero or raises when
  `tl_ood_margin_vs_best_neural_at_least_10pp.passed` is false.
- Tests cover the falsified-margin path without requiring a long neural run
  (monkeypatch or inject baseline metrics).
- Result JSON still records the failed gate details before the error path when
  practical.

Smallest useful validation:

```bash
python3 -m pytest tests/test_exp87_support_eval.py -q
```

### 2. Add CLI smoke tests that write to temp outputs

Owned files:

- `tests/test_support_cli_smokes.py`
- `experiments/exp87_support_eval.py` through `experiments/exp95_scored_object_hypotheses.py` only if a CLI bug is exposed

Acceptance criteria:

- A parametrized test runs each support/stability script with
  `--quick --output <tmp_path>/results.json`.
- The test asserts exit code 0, valid JSON on stdout, output file exists, and
  no default committed `experiments/exp*_data/results_quick.json` file is
  modified.
- Exp86 CLI quick smoke is included or covered by a separate focused test.

Smallest useful validation:

```bash
python3 -m pytest tests/test_support_cli_smokes.py -q
```

### 3. Validate result manifests and freshness

Owned files:

- `tools/validate_experiment_results.py`
- `tests/test_experiment_results_manifest.py`
- `docs/RUN_PROTOCOL.md` if the required schema is refined

Acceptance criteria:

- The validator checks committed `experiments/exp*_data/results*.json` for
  required fields: experiment name, quick/full mode, config, command or mode,
  result path, git SHA or explicit `unversioned` marker, and generated timestamp.
- The validator reports missing metadata for legacy artifacts without modifying
  them unless explicitly run with a repair flag.
- Tests cover both a valid manifest and a missing-field failure.

Smallest useful validation:

```bash
python3 -m pytest tests/test_experiment_results_manifest.py -q
python3 tools/validate_experiment_results.py
```

### 4. Add falsification specs for exp87-exp95

Owned files:

- `docs/exp87_support_eval_spec.md`
- `docs/exp88_support_noisy_relations_spec.md`
- `docs/exp89_support_primitive_confidence_spec.md`
- `docs/exp90_support_repair_sweep_spec.md`
- `docs/exp91_interval_support_uncertainty_spec.md`
- `docs/exp92_pixel_abstain_recover_spec.md`
- `docs/exp93_detector_calibration_stress_spec.md`
- `docs/exp94_object_hypothesis_layer_spec.md`
- `docs/exp95_scored_object_hypotheses_spec.md`
- Optional: `tests/test_experiment_specs.py`

Acceptance criteria:

- Every exp87-exp95 spec states the hypothesis, null baseline, falsification
  threshold, runtime tier, output artifact, and smallest validation command.
- The specs preserve the caveats already documented in `notes/EXPERIMENTS.md`,
  especially exp94's oracle upper-bound status and exp95's partial result.
- A lightweight test or script verifies every `experiments/exp8*.py`/`exp9*.py`
  support script has a matching spec with the words `Falsification` and
  `Validation`.

Smallest useful validation:

```bash
python3 -m pytest tests/test_experiment_specs.py -q
```

### 5. Add an install contract for optional LLM experiments

Owned files:

- `pyproject.toml`
- `tests/test_packaging_ci.py`
- `README.md`
- `CLAUDE.md`

Acceptance criteria:

- Add one or more optional extras, for example `lm`, that document/install the
  dependencies used by exp36, exp60d, exp77, and exp78 (`transformers`, `peft`,
  `datasets`, and MLX packages where appropriate).
- README and CLAUDE distinguish the core/dev install from optional LM/Kaggle
  experiment installs.
- Packaging tests assert the extras exist without requiring network access.

Smallest useful validation:

```bash
python3 -m pytest tests/test_packaging_ci.py -q
```

## Handoff

Changed files:

- `docs/overnight/2026-05-07-whole-portfolio-review/tensor-experiments-risk-and-validation-review.md`

PR URL: none; PR creation is out of scope for this queue item.

Blockers:

- No external credentials or approvals were needed.
- Local `python` is missing; use `python3` locally or a CI/setup-python
  environment that provides `python`.

Final required validation:

```bash
git status --short
```
