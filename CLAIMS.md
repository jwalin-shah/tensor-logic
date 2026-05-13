# Tensor Logic Claim/Evidence Boundary

Last updated: 2026-05-06

This file separates claims that are supported by committed experiments from
claims that remain hypotheses, extrapolations, or non-goals. It is intentionally
conservative: if a result depends on toy data, perfect object state, oracle
metadata, or a synthetic detector, the claim must say so.

## Supported Claims

### 1. Tensor Logic is strong when the operator matches the task structure.

Supported by:

- `README.md`
- `notes/EXPERIMENTS.md`
- `experiments/exp44_import_closure.py`
- `experiments/exp47_size_generalization.py`
- `experiments/exp53_real_imports.py`
- `experiments/exp54_big_imports.py`

Evidence boundary:

- The strongest validated closure claim is about transitive closure/reachability
  and import-graph closure.
- The 3-scalar recurrence generalizes across graph sizes because the recurrence
  matches Boolean closure over matrix powers.
- The README reports zero-shot real-package import-graph F1 results and a larger
  package sweep. It also names Django as a weak case, so the claim is not
  "perfect on all import graphs."

Allowed phrasing:

- "TL is highly parameter-efficient on closure-shaped tasks."
- "For transitive closure and import reachability, the structural recurrence
  generalizes across graph size better than flat MLP baselines in these
  experiments."

Do not claim:

- "TL beats neural nets generally."
- "TL solves code understanding."
- "TL handles arbitrary program analysis."

### 2. Perfect-state support/stability V1 passed its planned gate.

Supported by:

- `docs/superpowers/plans/2026-05-05-support-stability.md`
- `experiments/exp84_support_data.py`
- `experiments/exp85_support_tl.py`
- `experiments/exp86_support_baselines.py`
- `experiments/exp87_support_eval.py`
- `experiments/exp87_support_data/results.json`
- `tests/test_exp84_support_data.py`
- `tests/test_exp85_support_tl.py`
- `tests/test_exp86_support_baselines.py`
- `tests/test_exp87_support_eval.py`

Evidence boundary:

- Inputs are perfect object tables with primitive geometry computed from known
  object boxes.
- The full-run exp87 result reports TL at 100.0% on 3,438 deterministic labels
  and 100.0% on 1,709 counterfactual labels.
- The planned OOD gates passed against both padded MLP and DeepSets baselines:
  larger OOD margin was +16.3 percentage points over the best neural baseline,
  and deeper OOD margin was +30.8 percentage points.
- TL receives primitive geometry facts, not pre-derived `supports` labels.

Allowed phrasing:

- "Given perfect object-state input, the deterministic TL support module gives
  exact labels, proofs, and counterfactual retraction on the toy support/stability
  benchmark, and beats the tested MLP/DeepSets baselines on larger/deeper OOD
  stacks."

Do not claim:

- "TL solves physical reasoning."
- "TL works from real images."
- "TL has learned support rules."
- "The result covers collision, friction, deformation, motion transfer, or
  continuous trajectories."

### 3. The clean support/stability result is brittle to noisy geometry, but the
failure boundary is mapped.

Supported by:

- `experiments/exp88_support_noisy_relations.py`
- `experiments/exp88_support_noisy_relations_data/results.json`
- `tests/test_exp88_support_noisy_relations.py`

Evidence boundary:

- Zero-noise accuracy remains 100.0%.
- Tiny coordinate jitter can break exact support predicates, especially through
  contact and full-width coverage boundaries.
- A small extraction tolerance recovers microscopic jitter but does not make the
  system perception-robust.

Allowed phrasing:

- "The perfect-state support engine has a sharp brittleness boundary under
  geometry noise."
- "Tolerance helps at the tolerance scale only."

Do not claim:

- "The hard TL engine is robust to perception noise."

### 4. Confidence, interval, and repair wrappers provide useful abstain/recover
contracts under synthetic object-table and pixel-stub noise.

Supported by:

- `experiments/exp89_support_primitive_confidence.py`
- `experiments/exp90_support_repair_sweep.py`
- `experiments/exp91_interval_support_uncertainty.py`
- `experiments/exp92_pixel_abstain_recover.py`
- `experiments/exp89_support_primitive_confidence_data/results.json`
- `experiments/exp90_support_repair_sweep_data/results.json`
- `experiments/exp91_interval_support_uncertainty_data/results.json`
- `experiments/exp92_pixel_abstain_recover_data/results.json`
- `tests/test_exp89_support_primitive_confidence.py`
- `tests/test_exp90_support_repair_sweep.py`
- `tests/test_exp91_interval_support_uncertainty.py`
- `tests/test_exp92_pixel_abstain_recover.py`

Evidence boundary:

- These are wrappers around the same hard TL engine, not learned perception.
- The pixel-facing experiment uses rendered synthetic segmentation boxes and
  detector-style localization perturbations.
- The supported claim is about interface shape: calibrated uncertainty lets TL
  abstain or recover instead of forcing brittle hard primitive facts.

Allowed phrasing:

- "Calibrated coordinate bands are a useful interface between object detection
  and hard TL support reasoning."
- "Interval/repair routing can preserve 100% accepted accuracy at reduced
  coverage when uncertainty is calibrated in these synthetic settings."

Do not claim:

- "A real vision model has been integrated."
- "The pixel pipeline handles occlusion, missing detections, merged detections,
  or false positives without additional detector-health signals."

### 5. Detector structural failures require object identity/cardinality signals.

Supported by:

- `experiments/exp93_detector_calibration_stress.py`
- `experiments/exp94_object_hypothesis_layer.py`
- `experiments/exp95_scored_object_hypotheses.py`
- `experiments/exp93_detector_calibration_stress_data/results.json`
- `experiments/exp94_object_hypothesis_layer_data/results.json`
- `experiments/exp95_scored_object_hypotheses_data/results.json`
- `tests/test_exp93_detector_calibration_stress.py`
- `tests/test_exp94_object_hypothesis_layer.py`
- `tests/test_exp95_scored_object_hypotheses.py`

Evidence boundary:

- exp93 shows coordinate calibration is insufficient for missing, merged, or
  false-positive object failures.
- exp94 is an oracle/upper-bound object-hypothesis layer that uses simulated
  anomaly metadata.
- exp95 removes oracle source IDs and shows a partial foothold for false-positive
  ranking, but missing and merge repairs remain unsafe or incomplete.

Allowed phrasing:

- "Pixel-facing TL needs coordinate uncertainty and detector-health/object
  hypothesis signals."
- "The current non-oracle object-hypothesis layer is partial; false positives are
  the strongest recovered case, while missing/merge identity gaps remain open."

Do not claim:

- "Object hypotheses solve detector structural failures."
- "The non-oracle hypothesis ranker closes the gap to the oracle upper bound."

### 6. TL-as-tool works on the tested kinship closure protocol.

Supported by:

- `notes/EXPERIMENTS.md` rows for exp60d and related exp60 files.

Evidence boundary:

- The supported result is a small-LM delegation protocol for kinship closure,
  not general language reasoning.
- The language model learns to emit tool calls; TL performs the closure.

Allowed phrasing:

- "A small SFT'd LM can learn to delegate specific kinship closure questions to a
  TL tool in the tested synthetic protocol."

Do not claim:

- "The LM learned symbolic reasoning."
- "This proves general agentic reasoning."

## Unsupported Or Only Partially Supported Claims

- Real-world physical reasoning from natural images.
- Learned rule induction for support/stability.
- Learned perception for support/stability.
- Continuous physics, motion prediction, collision response, friction,
  deformation, or dynamics.
- General code understanding beyond import/reachability-style closure.
- General theorem proving.
- General superiority over neural models.
- Robustness to arbitrary detector failures.
- Non-oracle identity/cardinality recovery for missing or merged objects.
- TL inside transformers as a proven performance win.

## Explicit Non-Claims

- This repository is a research/learning project, not a production product.
- A toy benchmark pass is not a real-world deployment claim.
- A deterministic proof trace is not evidence that the input facts are true.
- Exact inference over wrong detected facts can still produce wrong conclusions.
- Baseline wins are scoped to the specific baseline implementations, training
  distributions, seeds, and gates in the committed experiments.

## Baselines Required Before Stronger Claims

For support/stability:

- Perfect-state TL rule engine.
- Padded-object MLP.
- DeepSets or GNN-style variable-object baseline.
- ID, larger OOD, deeper OOD, branching OOD, and counterfactual splits.
- Explicit gates for TL deterministic accuracy, TL counterfactual accuracy, and
  TL margin over the best neural baseline.

For noisy/pixel-facing claims:

- Hard point-estimate TL.
- Primitive confidence or calibrated uncertainty route.
- Interval feasibility route.
- Repair route, if repairs are part of the claim.
- Guarded abstention when detector structural failures are present.
- Oracle and non-oracle object-hypothesis variants must be reported separately.

For closure/code-graph claims:

- TL recurrence at fixed parameter count.
- Neural baseline with a stated size/input limitation.
- Native-size or explicitly cropped evaluation, with the crop caveat stated.
- Real-package results plus the known weak cases.

## Validation Commands

Clean checkout validation:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

Support/stability fast path:

```bash
python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v
python experiments/exp86_support_baselines.py --quick
```

Support/stability V1 gate:

```bash
python -m pytest tests/test_exp86_support_baselines.py tests/test_exp87_support_eval.py -v
python experiments/exp87_support_eval.py --quick
python experiments/exp87_support_eval.py
```

Noisy and pixel-facing boundary checks:

```bash
python -m pytest tests/test_exp88_support_noisy_relations.py tests/test_exp89_support_primitive_confidence.py tests/test_exp90_support_repair_sweep.py tests/test_exp91_interval_support_uncertainty.py -v
python -m pytest tests/test_exp92_pixel_abstain_recover.py tests/test_exp93_detector_calibration_stress.py tests/test_exp94_object_hypothesis_layer.py tests/test_exp95_scored_object_hypotheses.py -v
```

Quick experiment entrypoints:

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

## Reporting Rule

When writing a README, memo, PR description, paper note, or Linear issue from
this repo, include three fields:

- Claim:
- Evidence:
- Boundary:

If the boundary cannot be stated in one or two sentences, the claim is probably
too broad.
