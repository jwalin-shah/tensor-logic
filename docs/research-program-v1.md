# Research Program v1 - Typed Sparse-Tensor Cognitive Architecture

Source basis: the September 20, 2026 skeptical research report supplied by the
user, plus the existing Personal Physics / Tensor Logic implementation.

This file is an execution manifest, not a claim that the experiments have
already succeeded. GitHub CI is only a source-level regression witness. The
substantive runs belong on the Mac and OCI once LifeOps repo materialization and
dispatch are repaired.

## Runtime policy

- Mac: local/Apple data adapters, light CPU experiments, MPS-capable model runs.
- OCI: persistent LifeOps world state, tensor/materialized-view runtime, governed
  workers, reproducible CPU/GPU experiments, durable receipts.
- GitHub: source/version control, issues, independent regression tests.
- No experiment is promoted from synthetic sanity check to architectural evidence
  until it runs on its intended real benchmark and records raw manifests.

## Experiment queue

| ID | Issue | Question | Source-ready code | Substantive runtime |
|---|---:|---|---|---|
| A1 | #93 | CPG+TEG+tensors vs raw-log RAG for repair | software_schema.py, software_graph.py, temporal_execution_graph.py, exp105 | OCI |
| A2 | #94 | provenance semiring vs witness DAG | provenance_semiring.py, provenance_datalog.py, exp103 | Mac/OCI |
| A3 | #95 | ACT-R activation vs generic top-k | actr_memory.py, exp101 | Mac/OCI |
| A4 | #96 | VOC/resource-rational meta-control vs thresholds | metacognition.py, exp102 | Mac/OCI |
| A5 | #97 | latent -> rule extraction under shortcut/drift | rule_hypothesis_eval.py + existing rule-induction modules | OCI/GPU |
| A6 | #98 | persistent materialized state vs stateless RAG | reasoning_persistence.py + existing materialized views | OCI |
| A7 | #99 | restricted trusted kernel vs arbitrary execution | kernel_contract.py, exp104 | Mac/OCI |
| A8 | #100 | differentiable Datalog / Scallop over world state | local provenance_datalog.py; external Scallop run pending | Mac/OCI |
| A9 | #101 | JEPA-style predictive representation vs token prediction | predictive_world.py | OCI/GPU |
| A10 | #92 | Laya/MiniJev/classifier/LLM/Jev | branch research-os/system1-benchmark-v1 | Mac/OCI |

## Integrated hybrid-world proof

exp105_hybrid_software_world.py demonstrates the intended three-view substrate:

append-only evidence log
-> Temporal Execution Graph
-> typed sparse tensor materialization
-> tensor relation composition

The synthetic derivation maps a runtime error to the symbol executing in its
failing frame while preserving the original span evidence.

This is a mechanism proof only; it is not evidence that the representation
improves software repair.

## A1 - software world representations

Substantive design:
- fixed bug set from SWE-Bench Lite and/or Defects4J
- same model, prompts, tool budget, and stopping condition
- raw-log/RAG baseline
- CPG+filtered TEG query baseline
- CPG+TEG plus tensor materialization
- persist patch hypotheses, tests, outcomes, and receipts

Primary gate:
Do not make CPG/TEG mandatory if it fails to improve PASS@1, evidence quality,
or long-horizon efficiency enough to justify construction/update cost.

## A2 - provenance

Local baseline:
- exact proof-set semiring
- recursive provenanced closure
- explicit witness sidecar from existing materialized-view machinery

External comparison:
- install/run Scallop without changing the common dataset/evaluator
- compare top-k proof extraction, gradient-bearing provenance, memory, update
  cost, and explanation fidelity

Primary gate:
Use semiring provenance when algebra/learning needs it; keep structural/temporal
witness graphs for rich human/debugging queries unless evidence favors one view.

## A3 - working memory

Implemented:
- base-level activation from frequency/recency
- additive task-context association/mismatch
- retrieval probability
- retrieval latency
- vectorized activation

Substantive benchmark:
synthetic narratives and commitment/project memories with interference and
distractors; compare ACT-R-inspired retrieval with recency, frequency, cosine,
and learned relevance.

Primary gate:
Do not keep ACT-R-specific mechanics if a simpler learned relevance model
matches robustness and cost.

## A4 - metacognition

Implemented actions:
- ACT
- THINK
- ESCALATE

State includes confidence, contradiction, source staleness, error cost and
compute budget. Utility explicitly includes expected error loss and reasoning
cost.

Substantive benchmark:
learn/estimate action accuracies under distribution shift instead of using
environment-known values.

Primary gate:
If fixed thresholds match learned/VOC control on held-out task families, keep the
controller simple.

## A5 - latent to rule

Implemented admission evaluator:
- >=2 independent splits
- minimum held-out accuracy
- split variance/stability
- counterexample count
- explicit admission-candidate result

Substantive benchmark:
CLUTRR/SALSA-CLRS and existing anonymous-representation experiments, with rules
remaining candidate hypotheses until adversarial verification.

Primary gate:
Never promote a rule because it is interpretable; promote only when it survives
independent generalization and counterexample tests.

## A6 - persistent state

Implemented source-revision-aware materialized view baseline recording:
- source reads
- derivations
- cache hits
- invalidations

Substantive benchmark:
same long-horizon tasks under stateless RAG, retrieval cache, and admitted
proof-carrying materialized state.

Primary gate:
Persist derived state only where it measurably reduces repeated work/errors.

## A7 - trusted kernel

Implemented:
- finite monotone transitive closure
- explicit finite relation bound
- arbitrary host-language transition baseline requiring external fuel/watchdog

Primary gate:
Keep the trusted kernel restricted unless wider expressivity yields enough value
to outweigh termination/proof/incrementality costs.

## A8 - differentiable Datalog

Local baseline now supports recursive Datalog-style closure with provenance
propagated through algebra.

Still required:
- actual Scallop dependency
- differentiable neural predicate
- shared benchmark vs current Tensor Logic and pure GNN/neural baseline

## A9 - predictive world representations

Implemented:
- state encoder
- action-conditioned latent predictor
- stop-gradient target representation
- cosine predictive loss
- latent transition error

Still required:
- next-token/log baseline
- reconstruction/autoencoder baseline
- real trajectory dataset
- planning downstream task
- representation reuse across >=2 tasks

## A10 - System-1 decision models

Branch research-os/system1-benchmark-v1 contains:
- common Choice/Score/Noul result schema
- accuracy, Brier, NLL, ECE
- selective coverage/accuracy/risk curve
- option-order robustness
- Laya question translation/response normalization
- MiniJev adapter

Required backends:
- public Laya checkpoints
- MiniJev
- conventional encoder classifier
- small constrained-output LLM
- frontier constrained-output LLM
- Jev when direct API access exists

Important claim boundary:
Published third-party Jev numbers are external reference only. No head-to-head
claim is valid until the same cases run through both APIs.

## Execution order after LifeOps #44/#38

1. Materialize tensor-logic on OCI and bind exact commit.
2. Re-run exp99 performance on OCI as runtime baseline.
3. Run exp101-105 sanity experiments and write raw JSON manifests.
4. Run A10 on Laya + MiniJev first; add classifier and LLM backends.
5. Run A3/A4 on real/synthetic held-out datasets.
6. Install Scallop in an isolated environment and run A2/A8.
7. Run A6 persistent-state benchmark.
8. Build A1 CPG/TEG benchmark on a bounded software-repair subset.
9. Train A9 predictive representation.
10. Run A5 rule-extraction/adversarial generalization.
11. Recompute architectural decisions from evidence; do not preserve components
    merely because they were part of the original design.

## Current blockers

External execution blocker:
- LifeOps #44: governed admission/dispatch can strand proposals IN_FLIGHT before
  lease consumption/worker execution.
- LifeOps #38: tensor-logic repo materialization on OCI depends on functioning
  governed execution/bootstrap.

These blockers prevent substantive OCI experiments, but not source-native
implementation and regression testing.
