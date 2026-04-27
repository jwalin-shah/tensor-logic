# Benchmarks

Stable benchmark entrypoints go here. The first reusable core extraction lives
in `tensor_logic/`; new benchmark files should import from that package instead
of copying code out of `experiments/exp*.py`.

Suggested order:

1. `closure.py` — dense TL recurrence vs sparse BFS reachability.
2. `kinship.py` — multi-relation joins, negation, and provenance.
3. `parity.py` — standard sigmoid TL failure vs GF(2) control.
