# Tensor Logic Baseline Sweep Remote Job

Purpose: produce repeatable evidence for support/stability baseline claims
without broadening the claim beyond the current toy object-table benchmark.

## Scope

Run the committed support/stability evaluation with multiple seeds and both
quick and full settings. The target claim is limited to:

> Given perfect object-state input, the deterministic TL support/stability module
> reaches exact label/proof/counterfactual behavior on the generated toy
> benchmark and beats the tested neural baselines on larger/deeper OOD stacks.

This job does not test real images, learned perception, learned rules, continuous
physics, collision, friction, deformation, or motion prediction.

## Inputs

Repo:

```bash
git clone <repo-url> tensor-logic
cd tensor-logic
python -m pip install -e ".[dev]"
```

Reference files:

- `CLAIMS.md`
- `docs/superpowers/plans/2026-05-05-support-stability.md`
- `experiments/exp84_support_data.py`
- `experiments/exp85_support_tl.py`
- `experiments/exp86_support_baselines.py`
- `experiments/exp87_support_eval.py`
- `tests/test_exp84_support_data.py`
- `tests/test_exp85_support_tl.py`
- `tests/test_exp86_support_baselines.py`
- `tests/test_exp87_support_eval.py`

## Preflight

Run deterministic tests first:

```bash
python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py tests/test_exp86_support_baselines.py tests/test_exp87_support_eval.py -v
```

Expected:

- Generator invariants pass.
- TL exact support/counterfactual tests pass.
- Baseline tensorization/masking tests pass.
- Evaluation schema and gate tests pass.

Stop if any preflight test fails.

## Baseline Sweep

Minimum local smoke run:

```bash
python experiments/exp87_support_eval.py --quick --output experiments/exp87_support_data/results_quick.json
```

Full reference run:

```bash
python experiments/exp87_support_eval.py --output experiments/exp87_support_data/results.json
```

Seed sweep, if the evaluator exposes a seed/config flag in the current branch:

```bash
for seed in 8700 8701 8702 8703 8704; do
  python experiments/exp87_support_eval.py \
    --seed "$seed" \
    --output "experiments/exp87_support_data/results_seed_${seed}.json"
done
```

If the current CLI does not expose seed selection, do not patch product code in a
remote job. Record that limitation and run the committed quick/full commands
only.

## Required Metrics

Record these fields from each result JSON:

- `config.quick`
- `config.seed`
- `splits.train.scenes`
- `splits.id.objects`
- `splits.larger_ood.objects`
- `splits.deeper_ood.objects`
- `splits.counterfactual.objects`
- `tl.id.accuracy`
- `tl.larger_ood.accuracy`
- `tl.deeper_ood.accuracy`
- `tl.counterfactual.accuracy`
- `mlp.larger_ood.accuracy`
- `mlp.deeper_ood.accuracy`
- `deepsets.larger_ood.accuracy`
- `deepsets.deeper_ood.accuracy`
- `gates.tl_deterministic_label_accuracy_100.passed`
- `gates.tl_counterfactual_retraction_accuracy_100.passed`
- `gates.tl_ood_margin_vs_best_neural_at_least_10pp.passed`
- `gates.v1_passed`

The V1 support/stability claim is reportable only if all gates pass.

## Expected Current Baseline

The committed full result in `experiments/exp87_support_data/results.json`
reports:

- TL deterministic label accuracy: 100.0% on 3,438 labels.
- TL counterfactual retraction accuracy: 100.0% on 1,709 labels.
- Best larger OOD neural baseline: DeepSets at 83.7%; TL margin +16.3pp.
- Best deeper OOD neural baseline: DeepSets at 69.2%; TL margin +30.8pp.
- `gates.v1_passed`: true.

Treat these numbers as the current reference, not as a guarantee for modified
branches.

## Artifact Layout

Do not commit model weights or large generated outputs. Store remote artifacts
under the remote runner's output directory, then mirror only compact result JSONs
or a pointer file if needed.

Suggested output layout:

```text
remote-output/tensor-logic-baseline-sweep/
  manifest.json
  results_quick.json
  results.json
  results_seed_8700.json
  results_seed_8701.json
  summary.md
```

`manifest.json` should include:

```json
{
  "job": "tensor-logic-baseline-sweep",
  "git_sha": "<sha>",
  "started_at": "<iso8601>",
  "finished_at": "<iso8601>",
  "python": "<version>",
  "torch": "<version>",
  "commands": [],
  "results": []
}
```

## Summary Template

```markdown
# Tensor Logic Baseline Sweep Summary

Git SHA:
Runner:
Started:
Finished:

## Commands

- `python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py tests/test_exp86_support_baselines.py tests/test_exp87_support_eval.py -v`
- `python experiments/exp87_support_eval.py --quick --output ...`
- `python experiments/exp87_support_eval.py --output ...`

## Result

| run | TL ID | TL larger OOD | best neural larger OOD | margin | TL deeper OOD | best neural deeper OOD | margin | TL counterfactual | V1 gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full | | | | | | | | | |

## Claim Status

- Supported:
- Unsupported:
- Notes:
```

## Stop Conditions

Stop and report instead of continuing if:

- Any preflight test fails.
- TL deterministic label accuracy is below 100%.
- TL counterfactual retraction accuracy is below 100%.
- Either larger/deeper OOD margin over the best neural baseline is below 10pp.
- The result JSON omits either `mlp` or `deepsets`.
- The run modifies product code, tests, lockfiles, generated assets, or
  `web_workbench`.

## Boundary For Reporting

Acceptable final sentence:

> The perfect-state support/stability V1 gate passed for this SHA: TL was exact
> on generated labels and counterfactuals and cleared the planned OOD margin over
> MLP/DeepSets baselines.

Unacceptable final sentence:

> Tensor Logic solves visual physical reasoning.
