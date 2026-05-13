# Experiment Status Snapshot

Last reviewed: 2026-05-13

This snapshot is a small status artifact for agents and reviewers. It does not
upgrade any experiment's evidence tier. Treat `docs/EXPERIMENT_PROVENANCE.md`,
`docs/VALIDATION.md`, and `notes/EXPERIMENTS.md` as the authority before
quoting numbers or changing claim language.

## Current Contract Status

| Area | Status | Evidence boundary |
| --- | --- | --- |
| Provenance contract | Active | New or refreshed claims must record command, commit, dependency set, inputs, outputs, gate, and caveats. |
| No-overclaim rule | Active | Toy, synthetic, simulated, oracle, remote-only, or historical results must stay scoped to the exact tested regime. |
| Validation matrix | Active | Docs-only contract edits use the narrow parse/package checks; metric changes need the exact experiment command or a historical-artifact note. |
| Result artifact index | Partial | `exp78`, `exp79`, `exp83`, and `exp87`-`exp95` have local pointers; earlier headline import-graph claims still need frozen manifests before calling them fully locally reproducible. |

## Claim Status Summary

| Claim family | Current status | Allowed wording |
| --- | --- | --- |
| Reusable `tensor_logic` package | Locally testable package surface | May claim local package/import/test validation when backed by the CI command in `docs/VALIDATION.md`. |
| Transitive-closure import graph headline | Historical experiment result with external package-download dependency | May describe as historical evidence; do not claim fully reproducible from checked-in artifacts alone until manifests/results are committed. |
| Support/stability experiments `exp87`-`exp95` | Locally indexed artifact-backed research slices | May cite exact artifact-backed behavior and caveats; do not imply production perception, real-world physics, or end-to-end vision. |
| Oracle/simulated object hypotheses `exp94` | Upper-bound simulated result | Must call it oracle/simulated or upper-bound when cited. |
| Non-oracle scored hypotheses `exp95` | Mixed non-oracle result | May cite false-positive ranking improvement; must retain missing-object and merge/cardinality caveats. |

## Required Next Review Triggers

- README headline metrics or benchmark language changes.
- Any new `experiments/exp*_data/results.json` or refreshed plot/result file.
- Any PR/issue claim that broadens from a local artifact to a general capability.
- Any remote, GPU, model-download, external dataset, or package-download run.

Run the command selected from `docs/VALIDATION.md` after updating this status.
