# exp78: LM-Guided TL Rule Induction

## Thesis

Tensor Logic induces rules from examples. The LM only constrains the search and
explains the result. The LM never authors a rule.

This is the differentiated version of "small LM + symbolic reasoner": the LM is
a search-space narrower, not a program writer. exp77 (LM writes rules) is
strictly weaker — skip it.

## Prior art in repo

`experiments/exp21_rule_induction.py` already implements brute-force two-hop
induction over a fixed family schema (5×5 = 25 candidate rules, F1 scoring).
exp78 extends exp21 along two axes:

1. Multi-schema eval (held-out relation names, held-out compositions).
2. Optional LM pruner that proposes a relation subset before search.

## Falsification criterion (set NOW, do not move)

- **TL alone (no LM):** recovers the gold rule for `grandparent`, `uncle`,
  `great_uncle` from ≤20 pos + ≤20 neg examples in ≤1s search time per target.
  If this fails, search-space pruning (typing, mode declarations) is the
  prerequisite — LM cannot help yet.
- **LM-guided TL:** matches TL-alone accuracy with ≥3× speedup on schemas with
  ≥10 relations. If LM-guided is slower or less accurate, the LM is not adding
  value as a pruner.

## Data format

```json
{
  "schema": [
    {"rel": "parent", "args": ["person", "person"]},
    {"rel": "sibling", "args": ["person", "person"]}
  ],
  "facts": [["parent", "alice", "bob"], ["parent", "bob", "carol"]],
  "target": {"rel": "grandparent", "args": ["person", "person"]},
  "positive": [["alice", "carol"]],
  "negative": [["alice", "dave"]],
  "gold_rule": {
    "head": ["grandparent", "X", "Y"],
    "body": [["parent", "X", "Z"], ["parent", "Z", "Y"]]
  }
}
```

## Eval splits

| Split | Tests | Construction |
|---|---|---|
| `seen_schema` | Memorization + selection | Train and eval over family schema |
| `heldout_schema` | Compositional transfer | Train family, eval workplace (`manages`, `reports_to`) with same rule shapes |
| `heldout_composition` | Novel rule shapes | Eval requires 3-hop chains never seen at train time |
| `distractor` | Robustness to irrelevant relations | Schema includes 5+ unused relations |

`heldout_schema` is the load-bearing number. Report all four.

## Procedure

### v1: TL-only induction (no LM)

1. For each target `R(X,Y)`, enumerate body conjunctions of length ∈ {1, 2, 3}
   over schema relations, respecting argument types.
2. Score each candidate by F1 against pos/neg examples.
3. Tie-break by minimal body length.
4. Return top rule + score.

Implementation: extend `exp21_rule_induction.py`'s `score_rule` to accept
arbitrary schema (not hardcoded `Parent`/`Sibling`) and arbitrary body length.

### v2: LM-guided pruning

1. LM input: schema + question + a few positive/negative examples.
2. LM output (constrained JSON):
   ```json
   {"relevant_relations": ["parent", "sibling"], "max_body_length": 2}
   ```
3. TL search runs only over the LM-proposed subset.
4. If search fails (no rule scores ≥0.9 F1), fall back to full search.

LM = Qwen2.5-0.5B-Instruct. Constrained decoding via outlines/grammar.

## Metrics

- **Rule found**: any rule scoring F1 ≥ 0.9 on examples
- **Semantic equivalence**: gold and induced rule produce identical answer sets
  on 100 randomly generated worlds (better than string match)
- **Held-out answer accuracy**: induced rule applied to held-out facts
- **Search cost**: number of candidates evaluated
- **LM pruning value (v2 only)**: speedup vs. v1 with no accuracy loss

## Out of scope (do not let scope creep)

- Fact extraction from raw text
- Rule promotion / "rule factory" loop
- Multi-domain demos (inbox, codebase, jobs)
- Differentiable rule scoring with semirings (cool, but separate experiment)
- Recursive rules

## Deliverables

- `experiments/exp78_rule_induction.py` — single script, two functions
  (`induce_tl_only`, `induce_lm_guided`), four eval splits
- `experiments/exp78_data/` — generated schemas + examples
- `experiments/exp78_data/results.json` — metrics per split per condition
- One chart: held-out-schema accuracy, TL-only vs LM-guided
