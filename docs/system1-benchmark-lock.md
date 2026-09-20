# System-1 benchmark upstream lock

## Laya

Inspected upstream:
- repository: NandhaKishorM/laya
- main commit: d113dca2512fb3eaca313534bc54c7162d87c1d4
- package version: 0.3.4
- license: Apache-2.0

Public inference API used by the adapter:
- laya.load(...)
- Agent.predict(state, questions)
- primitives: choice, score, noul
- choice response: top label + per-option probabilities + confidence
- score response: expected ordinal score + indexed probability distribution + confidence
- noul response: P(true) + confidence

The benchmark adapter does not copy Laya's model architecture. It only translates
our backend-neutral benchmark questions/results to/from Laya's public SDK shape.

## Claim boundaries

- Laya's Jev values are external published/third-party reference numbers, not a
  controlled run by Laya's author.
- Do not compare Laya and Jev as head-to-head until identical cases run through
  both live backends.
- Zero-shot, task-fine-tuned, and calibration-refit settings must be reported
  separately.
- Model routing/preloading state is part of the systems benchmark because cold
  model reload time is not equivalent to hot inference latency.
- High-cardinality label experiments must record option count and Laya
  head_max_len/max_len settings rather than hiding token-budget constraints.
