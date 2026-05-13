# Tensor Experiments Context

This repository is a research workspace and reusable `tensor_logic` package, not
a product benchmark suite. Treat checked-in experiments as evidence with
provenance requirements, not as license to make broader claims.

## Domain Vocabulary

- **Tensor Logic (TL)**: the named-index tensor/einsum rule substrate in
  `tensor_logic/` and the experiment scripts that probe its limits.
- **Experiment**: a numbered `experiments/expN_*.py` script with an explicit
  hypothesis, local or remote run command, and recorded result row in
  `notes/EXPERIMENTS.md`.
- **Result artifact**: a committed JSON/plot/manifest under an
  `experiments/*_data/` directory. A result artifact is evidence only for the
  exact script, inputs, command, dependency set, and commit that produced it.
- **Claim**: any README, note, PR, issue, or comment statement that generalizes
  from experiment results.
- **No-overclaim rule**: every claim must state its evidence tier, scope, and
  caveats. Do not promote a toy, simulated, oracle, synthetic, internally
  validated, or remote-only result into a general capability claim.

## Current Provenance Rules

- Use `docs/EXPERIMENT_PROVENANCE.md` before changing README claims,
  experiment-status notes, result tables, or benchmark language.
- Use `docs/VALIDATION.md` before selecting a validation command.
- Use `docs/RUN_PROTOCOL.md` for remote/Kaggle/model-training runs and for any
  run that can create durable experiment artifacts.
- Keep `notes/EXPERIMENTS.md` append-only in spirit: failed, mixed, superseded,
  and invalid experiments are evidence and should remain visible.

## Claim Boundaries

- `exp87` through `exp95` are support/stability research slices. They establish
  specific toy/simulated object-table and pixel-facing interface behavior, not
  real-world physics, production perception, or end-to-end vision.
- `exp94` is an oracle/simulated upper-bound structural recovery result. Cite it
  as an upper bound unless the claim also references non-oracle follow-up.
- `exp95` is non-oracle and mixed: it improves false-positive ranking but still
  exposes missing-object and merge identity/cardinality gaps.
- `exp78`, `exp79`, and `exp83` have committed result artifacts, but their
  status and gates must be read from the result artifact plus
  `notes/EXPERIMENTS.md`, not inferred from old planning docs.
- `exp53` and `exp54` headline import-graph numbers depend on external package
  source downloads unless frozen manifests are added. Do not present them as
  locally reproducible from checked-in artifacts alone.

## Agent File Note

`AGENTS.md` is memjuice-managed per `CLAUDE.md`. Do not hand-edit it when a
repo-local context or validation rule can live in `CONTEXT.md`,
`docs/VALIDATION.md`, or `docs/EXPERIMENT_PROVENANCE.md`.
