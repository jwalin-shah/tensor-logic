# Experiment Provenance And No-Overclaim Contract

This contract keeps experiment claims tied to the evidence that produced them.
Use it for README edits, issue descriptions, PR summaries, research notes, and
future experiment handoffs.

## Evidence Tiers

| Tier | Meaning | Claim wording allowed |
| --- | --- | --- |
| T0 docs-only | Planning note, spec, or audit with no fresh run | "planned", "hypothesized", "audit found" |
| T1 local deterministic | Runs on CPU from repo code with default/dev deps and no network | "locally validated by `<command>`" |
| T2 local optional | Requires optional extras such as `lm`, `sft`, `science`, MLX, or longer training | "validated with optional deps", plus dependency/runtime notes |
| T3 external data/model | Downloads packages, uses official/private datasets, or loads model/provider artifacts | "validated against `<dataset/model>`", plus version/pointer |
| T4 remote/accelerated | Kaggle/Colab/Modal/HF/GPU/MPS run or saved adapter/checkpoint | "remote run produced", plus provider and durable artifact pointer |

Claims may not be worded above their evidence tier. A T1 toy result cannot imply
production readiness, real-world validity, or robustness to inputs it did not
test.

## Required Provenance For New Results

Every new or refreshed result claim must record:

- Experiment script path.
- Exact command, including flags and whether `--quick` was used.
- Git commit SHA at run time.
- Python version and dependency set (`default`, `dev`, `lm`, `sft`, `science`,
  or external environment).
- Input dataset/model/package references, including versions or immutable
  pointers when available.
- Output artifact paths committed to the repo or durable external artifact URLs.
- Falsification criterion or pass/fail gate.
- Caveats that limit the claim.

If an experiment writes weights, adapters, large outputs, or remote artifacts,
do not commit those binaries. Commit `results.json` and a pointer file such as
`ADAPTER.md` or `PROVENANCE.md` instead.

## Current Result Artifact Index

This is a repo-local pointer index, not a full metric table. Read the linked
artifact and `notes/EXPERIMENTS.md` before quoting numbers.

| Experiment range | Committed evidence | Claim boundary |
| --- | --- | --- |
| `exp53`, `exp54` import graphs | No checked-in `exp53_data/results.json` or `exp54_data/results.json` | Headline numbers depend on external package downloads; do not call them fully locally reproducible without frozen manifests. |
| `exp78` | `experiments/exp78_data/results.json` | Rule-induction result for the recorded targets/settings; LM-pruner claims require model/dependency context. |
| `exp79` | `experiments/exp79_data/results.json` and tests | Use current implementation/results, not stale "pending" planning text. |
| `exp83` | `experiments/exp83_slot_data/results.json` | Slot-attention probe gate is mixed; do not imply the probe solved the exp79 limitation unless the artifact gate supports it. |
| `exp87` | `experiments/exp87_support_data/results.json` plus `results_quick.json` | Perfect object-state support/stability substrate, not perception or real physics. |
| `exp88` | `experiments/exp88_support_noisy_relations_data/results.json` plus `results_quick.json` | Noise boundary mapping; exact TL is brittle under geometry jitter. |
| `exp89` | `experiments/exp89_support_primitive_confidence_data/results.json` plus `results_quick.json` | Primitive confidence is a triage/abstention signal, not a perception fix. |
| `exp90` | `experiments/exp90_support_repair_sweep_data/results.json` plus `results_quick.json` | Repair helps boundary noise but is not a full pixel/object uncertainty solution. |
| `exp91` | `experiments/exp91_interval_support_uncertainty_data/results.json` plus `results_quick.json` | Calibrated coordinate intervals support abstain/recover; under-calibration leaks errors. |
| `exp92` | `experiments/exp92_pixel_abstain_recover_data/results.json` plus `results_quick.json` | Pixel-facing detector stub/interface proof, not production vision. |
| `exp93` | `experiments/exp93_detector_calibration_stress_data/results.json` plus `results_quick.json` | Structural detector failures require health/identity/cardinality signals. |
| `exp94` | `experiments/exp94_object_hypothesis_layer_data/results.json` plus `results_quick.json` | Oracle/simulated object-hypothesis upper bound. |
| `exp95` | `experiments/exp95_scored_object_hypotheses_data/results.json` plus `results_quick.json` | Non-oracle mixed result: useful false-positive ranking, unresolved missing/merge gaps. |

## No-Overclaim Rules

- State the tested input regime: toy, synthetic, simulated, oracle,
  internally generated, external package corpus, official dataset, or remote
  model run.
- Preserve negative and mixed results. They are not cleanup targets.
- Distinguish substrate claims from application claims. Exact TL behavior over
  perfect object tables is not evidence that perception is solved.
- Distinguish interface proofs from production systems. A detector stub or
  simulated anomaly signal is not an end-to-end vision model.
- Distinguish reproducible local artifacts from historical remote results.
  Remote claims need durable provider/artifact pointers.
- Quote metrics from artifacts or `notes/EXPERIMENTS.md`; do not round them into
  stronger language such as "solves", "robust", "general", or "production" unless
  the gate actually tested that.

## Validation For Claim Edits

For docs-only claim/provenance edits, run:

```bash
python3 -m pytest tests/test_packaging_ci.py -v
```

If a claim edit touches code-index or validation-contract wording, also run:

```bash
python3 -m pytest tests/test_code_index.py -v
```

If a claim edit changes experiment numbers, run the exact experiment command or
state why the artifact is historical and not being refreshed.
