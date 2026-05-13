# Architecture Issue Candidates - 2026-05-12

Do not implement yet. These are planning candidates from a read-only architecture review.

## 1. Deepen the rule surface

Files: `tensor_logic/program.py`, `tensor_logic/rules.py`, `tensor_logic/proofs.py`, `tensor_logic/provenance.py`, `tests/test_tensor_logic_core.py`, `tests/test_reason.py`

Acceptance criteria:
- One canonical Rule representation is used by parsing, tensor evaluation, proof search, and provenance adapters.
- Existing Program rule examples still pass.
- Parser tests cover arity, negation, disjunction, and expression methods.

Validation: `python -m pytest tests/test_tensor_logic_core.py tests/test_reason.py tests/test_proof_recursion.py -v`

## 2. Add a proof engine seam

Files: `tensor_logic/proofs.py`, `tensor_logic/proof_result.py`, `tensor_logic/proof_tree_viewer.py`, `tensor_logic/execution.py`, `tensor_logic/http_api.py`

Acceptance criteria:
- CLI, HTTP, and web paths call one proof engine Interface.
- Positive, negative, recursive, and do-intervention proof modes keep current behavior.
- Binary-relation limitations are explicit in the Interface.

Validation: `python -m pytest tests/test_proof_recursion.py tests/test_reason.py tests/test_web_workbench.py -v`

## 3. Extract experiment harness conventions

Files: `experiments/exp79_self_play_loop.py`, `experiments/exp81_sweep.py`, `experiments/exp81_optimize_rule_induction.py`, `experiments/exp83_slot_attention.py`, `tensor_logic/research/utils.py`

Acceptance criteria:
- Common config/result/quick-mode behavior lives in a research harness Module.
- Existing experiment outputs keep their file locations.
- New harness tests prove deterministic result metadata.

Validation: `python -m pytest tests/test_exp79.py tests/test_exp81.py tests/test_exp83_slot_attention.py -v`
