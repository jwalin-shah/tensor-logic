# Support/Stability TL Research Plan

Date: 2026-05-05

## Goal

Build the first clean physics-grounded Tensor Logic result:

> Given perfect object-state input, a tensor-logic support/stability module generalizes to larger and deeper physical stacks better than neural-only baselines, while preserving exact proof and counterfactual behavior.

This is deliberately not a full physics engine, pixel pipeline, learned rule induction, collision simulator, or motion model. The first proof isolates the relational support structure where TL should be load-bearing.

## Research Claim

TL provides a deterministic relational substrate for physical support reasoning. Neural models can map observations into that substrate later, but the first experiment must prove the substrate helps when object state is already known.

## Non-Goals For V1

- No pixel input.
- No learned perception.
- No learned rules.
- No continuous trajectory prediction.
- No collision response, friction, deformation, or motion transfer.
- No partial physics priors. Encode the full toy support rule or do not claim support reasoning.

## Input And Output

Input object table:

```text
objects = [
  {id: A, x, y, w, h},
  {id: B, x, y, w, h},
  ...
]
intervention = none | remove(object_id)
```

Primitive relations computed from geometry:

```text
touching(X, Y)
above(X, Y)
horiz_overlap(X, Y)
on_ground(X)
removed(X)
```

Output:

```text
stable(X)
falls(X)
proof_tree(X)
```

## Rule Sketch

Exact rules may shift during implementation, but the rule family should stay closed and deterministic:

```text
stable(X) :- on_ground(X), not removed(X)
supports(Y, X) :- touching(X, Y), above(X, Y), horiz_overlap(X, Y), stable(Y), not removed(Y)
stable(X) :- supports(Y, X), not removed(X)
falls(X) :- not stable(X), not on_ground(X), not removed(X)
```

The key modeling choice is that TL receives primitive geometry facts, not pre-derived `supports` labels. If `supports` is handed in directly, the experiment becomes trivial.

## Baselines

Run at least:

1. TL rule engine over primitive relations.
2. MLP over padded object-table features.
3. DeepSets or GNN baseline for variable object counts.

The GNN/DeepSets baseline is required before making a serious generalization claim. Beating only a weak MLP is not enough.

## Evaluation

Train baselines on:

```text
2-3 objects
simple vertical stacks
short support depth
```

Evaluate on:

```text
ID held-out scenes
4-8 object larger-stack OOD scenes
deeper support chains
branching support structures
counterfactual removals
unseen widths/heights
```

Metrics:

```text
ID object-label accuracy
OOD object-label accuracy
counterfactual accuracy
depth-k support-chain accuracy
proof correctness for TL
```

## Falsification Gate

V1 passes only if:

- TL reaches 100% deterministic label accuracy on generated scenes.
- TL reaches 100% counterfactual retraction accuracy on removal cases.
- TL beats the best neural baseline by at least 10 percentage points on larger/deeper OOD stacks.

If TL cannot beat the best baseline with perfect object state, stop before adding pixels or learned rule induction.

## Testing Contract

The tests should catch contract bugs, not just exercise functions. Every layer needs an oracle or invariant that is independent of the code path under test.

### Generator Invariants

The generator tests must prove scenes are valid before any model consumes them:

- Same seed produces the same scene and labels.
- Different seeds produce varied scenes.
- Rectangles stay in bounds.
- Rectangles do not accidentally overlap.
- Contact boundaries exist only where intended.
- Interventions remove exactly one object and recompute labels.
- Labels match an independent oracle that is not the same function used by the generator internals.

### Relation Extractor Tests

Use hand-built object tables with obvious geometry, and assert primitive facts directly:

- `touching(X, Y)`.
- `above(X, Y)`.
- `horiz_overlap(X, Y)`.
- `on_ground(X)`.
- `removed(X)`.

These tests should include near-miss cases: almost touching, vertically aligned without horizontal overlap, horizontal overlap without contact, and removed objects.

### TL Rule Tests

Use fixed scenes with exact expected outcomes:

- Single block on ground is stable.
- Block on a stable block is stable.
- Floating block falls.
- Deeper support chain remains stable.
- Removing the bottom support causes dependent blocks to fall.
- Branching support retracts only affected facts.
- Object order and object id renaming do not change structural labels.

### Property Tests

Generated/property-style tests should cover:

- Removing an unrelated object does not change an independent tower.
- Increasing support depth does not break TL.
- Permuting object order does not change labels.
- Renaming object IDs does not change labels.
- TL reaches 100% against generator labels on deterministic generated samples.

### Baseline Sanity Tests

Neural baseline unit tests should not require a high research accuracy, but they must verify:

- Tensorization shape.
- Padding/masking correctness.
- Variable object-count batching.
- Metrics ignore padded objects.
- `--quick` training reduces loss on a tiny overfit set or at least runs deterministically and emits metrics.

### Evaluation Gate Tests

The final evaluation script should fail loudly if:

- TL deterministic accuracy is below 100%.
- TL counterfactual accuracy is below 100%.
- Results JSON is missing required fields.
- ID and OOD splits accidentally use the same distribution.
- The best neural baseline is absent.
- The 10 percentage-point OOD gate is not computed.

Fast CI should run deterministic tests only:

```bash
pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v
```

Keep full neural runs out of CI. Use `--quick` modes for smoke checks.

## Vertical Slices

### 1. Stack Generator

Create deterministic scene generation for object-table stacks with exact `stable/falls` labels.

Expected artifact:

- `experiments/exp84_support_data.py`
- `tests/test_exp84_support_data.py`

Validation:

```bash
pytest tests/test_exp84_support_data.py -v
```

### 2. TL Stability Engine

Convert object tables to primitive relations, run TL fixpoint support rules, and return proof trees.

Expected artifact:

- `experiments/exp85_support_tl.py`
- `tests/test_exp85_support_tl.py`

Validation:

```bash
pytest tests/test_exp85_support_tl.py -v
```

### 3. Neural Baselines

Train/evaluate an MLP and a DeepSets or GNN baseline on the same generated dataset.

Expected artifact:

- `experiments/exp86_support_baselines.py`
- `tests/test_exp86_support_baselines.py`

Validation:

```bash
pytest tests/test_exp86_support_baselines.py -v
python experiments/exp86_support_baselines.py --quick
```

### 4. OOD And Counterfactual Evaluation

Run the full ID/OOD/counterfactual comparison and write a compact results table.

Expected artifact:

- `experiments/exp87_support_eval.py`
- runtime output under `experiments/exp87_support_data/`

Validation:

```bash
python experiments/exp87_support_eval.py --quick
```

Full run:

```bash
python experiments/exp87_support_eval.py
```

### 5. Noisy Relation Robustness

Perturb positions and primitive relations to quantify where deterministic TL becomes brittle and where neural smoothing helps.

This is intentionally after the clean perfect-state result.

### 6. Pixels Or Rule Induction

Choose one stretch direction after V1:

- Pixel frontend: render stacks and train object/relation extraction.
- Rule induction: induce/select support rules from labeled examples.

Do not do both in the first month.

## Month Execution Plan

Week 1:

- Stack generator.
- TL engine.
- Deterministic and counterfactual unit tests.

Week 2:

- MLP baseline.
- DeepSets/GNN baseline.
- Initial ID/OOD eval.

Week 3:

- Counterfactual eval at scale.
- Proof tree checks.
- Noisy relation robustness.

Week 4:

- Results cleanup.
- Add `notes/EXPERIMENTS.md` row.
- Write final memo with claim, failure modes, and next vertical slice.

## Worker Contract

Each Linear issue should include:

- Repo: `jwalin-shah/tensor` or local path `tensor-logic`.
- Branch rule: `codex/<ISSUE-KEY>-<short-title>`.
- Required plan: this document.
- Files owned by the issue.
- Acceptance criteria.
- Validation command.
- No edits outside owned files unless the worker records the reason in `CODEX_WORKPAD.md`.
