# Support/Stability V1 Findings

Date: 2026-05-09

## Summary

The V1 support/stability thesis passed on the clean, perfect-object-state benchmark.
Given object tables and primitive geometry facts, the deterministic TL module matched
the generated labels exactly, preserved counterfactual retraction, and beat the best
neural baseline by more than the planned 10 percentage-point margin on the larger and
deeper OOD gates.

This is a substrate result only. It does not show pixel-level perception, learned rule
induction, noisy-object robustness, collision dynamics, friction, deformation, or real
physics.

## Evidence Used

Full result:

```bash
python experiments/exp87_support_eval.py
```

Result JSON:

```text
experiments/exp87_support_data/results.json
```

Reproduced quick validation during this memo pass:

```bash
python3 experiments/exp87_support_eval.py --quick
```

Quick result JSON:

```text
experiments/exp87_support_data/results_quick.json
```

The full run was already present from SYM-54, so this memo did not rerun the longer
full experiment.

## What Happened

The benchmark trained neural baselines on 2-3 object ID stacks, then evaluated ID
held-out scenes, 4-8 object larger-stack OOD scenes, deeper support chains, branching
support structures, and removal counterfactuals.

TL received primitive geometry facts, not pre-derived `supports` labels. Its proof
coverage was complete across the full deterministic and counterfactual eval sets:
the `missing` proof count was 0 on every split in `results.json`.

## V1 Gates

| Gate | Result | Verdict |
|---|---:|---|
| TL deterministic label accuracy is 100% | 3,438 / 3,438 labels, 100.0% | PASS |
| TL counterfactual retraction accuracy is 100% | 1,709 / 1,709 labels, 100.0% | PASS |
| TL beats best neural baseline by at least 10pp on larger OOD | TL 100.0% vs DeepSets 83.7%, +16.3pp | PASS |
| TL beats best neural baseline by at least 10pp on deeper OOD | TL 100.0% vs DeepSets 69.2%, +30.8pp | PASS |

Overall V1 verdict: PASS.

## Baseline Readout

DeepSets was the best neural baseline on both planned OOD margin gates:

| Split | TL | MLP | DeepSets | Best neural gap |
|---|---:|---:|---:|---:|
| ID | 100.0% | 100.0% | 100.0% | 0.0pp |
| Larger OOD | 100.0% | 71.5% | 83.7% | +16.3pp |
| Deeper OOD | 100.0% | 68.1% | 69.2% | +30.8pp |
| Branching OOD | 100.0% | 62.6% | 74.4% | +25.6pp |
| Counterfactual | 100.0% | 62.8% | 90.3% | +9.7pp |

The planned OOD gate only named larger and deeper OOD stacks. Counterfactual was a
separate 100% TL retraction gate, not a 10pp neural-margin gate.

## Interpretation

The original clean-state claim holds: when the object table is correct and the full
toy support rule is encoded over primitive geometry relations, TL gives exact labels,
complete proofs, and exact removal retraction. The OOD gap is not just against a weak
padded MLP; DeepSets is the stronger baseline and still loses on the larger and deeper
generalization gates.

The result does not justify claiming general physical reasoning. It says that this
particular support/stability relation is a good fit for a deterministic tensor-logic
substrate when the symbolic object-state interface is already solved.

## Limitations And Failure Modes

- Perfect object state is assumed. The benchmark starts from object tables and clean
  primitive geometry facts, so perception and detector errors are out of scope.
- The encoded rule is the full toy support rule. V1 does not show that TL can learn
  the rule from data.
- The world model is narrow: static axis-aligned stacks with support, falling labels,
  and removals. It excludes friction, collisions, deformation, momentum, and continuous
  trajectories.
- Exact threshold predicates can be brittle near contact and overlap boundaries. This
  is why noisy relation robustness should be measured before any pixel or real-physics
  claim.
- The counterfactual neural margin was smaller than the planned larger/deeper OOD
  margins: DeepSets reached 90.3% on counterfactual labels while TL stayed at 100.0%.
  That does not fail a V1 gate, but it does show where the baseline was closest.

## Next Slice

Do noisy relation robustness next.

Reason: the clean V1 result is strong enough to continue, but the first likely failure
mode is not rule expressivity; it is whether exact geometry predicates survive noisy
object tables. The next slice should perturb positions and primitive relations, measure
where hard TL breaks, and decide whether the right interface is tolerance, confidence,
abstention, repair, or a detector uncertainty contract.

Do not move to pixels or learned rule induction until that boundary is mapped.

## Deliberately Omitted Changes

- `notes/EXPERIMENTS.md` already has the exp87 row from SYM-54, so this memo does not
  duplicate it.
- `README.md` currently does not make a support/stability result claim stronger than
  the produced data supports, so this memo leaves README unchanged.
