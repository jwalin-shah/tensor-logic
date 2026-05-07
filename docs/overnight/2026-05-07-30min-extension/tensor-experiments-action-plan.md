# tensor-experiments 30-minute extension action plan

Date: 2026-05-07
Queue item: `tensor-experiments-30min-action-plan`
Branch: `codex/goal-tensor-experiments-30min-action-plan`
Repo: `tensor-experiments`
Starting HEAD: `5565791a8c888511bbbff1107d2d357164f88baa`

## Scope

This was read-only planning/synthesis with one allowed write: this report.
I did not edit product code, generated experiment data, tests, CI, external
trackers, deployments, remote jobs, pushes, or PRs.

The purpose of this extension was not to repeat the broad overnight audits. The
useful next move is to reconcile those reports and convert their repeated risks
into implementation-ready slices with owned files, proof commands, and stop
conditions.

## Prior overnight reconciliation

Previous tensor-experiments reports exist outside this checkout under the
2026-05-07 overnight-marathon worktrees:

- `tensor-experiments-architecture-map.md` mapped the reusable package layers:
  `tensor_logic/language.py`, `tensor_logic/program.py`,
  `tensor_logic/proofs.py`, `tensor_logic/file_format.py`,
  `tensor_logic/execution.py`, `tensor_logic/http_api.py`,
  `tensor_logic/research/*`, and the web workbench. It already covered the
  canonical rule-language direction and the broad module ownership map.
- `tensor-experiments-docs-claims.md` identified stale README claims, missing
  frozen result artifacts for headline exp53/54 numbers, the `python` vs
  `python3` local mismatch, and exp79/exp83 status ambiguity.
- `tensor-experiments-workflow-handoff.md` identified the exp79 artifact
  collision, missing experiment workflow manifest, optional dependency gaps, and
  the need for no-write experiment smoke commands.
- `tensor-experiments-validation-map.md` proved the full local suite under
  `python3` and documented that the advertised `python` command fails in this
  worker shell.
- `tensor-experiments-risk-register.md` covered local HTTP/workbench exposure,
  unrestricted `.tl` includes, archive extraction, FAFSA user-facing wording,
  and run-protocol enforcement gaps.
- `tensor-experiments-dependency-surface.md` mapped declared vs observed
  dependencies, generated artifacts, ignored state, and missing optional extras.

What those reports already covered well: repository architecture, validation
surface, dependency drift, docs drift, local security risks, and runner
handoff risks.

What was still missing: an explicit implementation queue that picks the first
few work slices, defines file ownership, acceptance criteria, and local proof
commands. This report fills that gap. The repeated findings below should move
into implementation work instead of producing more general audit reports.

## Concrete file observations

1. `README.md:85-92` documents worker validation with
   `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`, but
   `python` is absent in this local shell. `python3 -m pytest tests/ -v` passed.

2. `.github/workflows/ci.yml:18-30` pins CI to Python 3.11 and uses `python`.
   The local run used Python 3.12.8, so the docs need to distinguish CI Python
   from local macOS `python3`.

3. `tests/test_packaging_ci.py:16-42` asserts the current packaging/CI/README
   validation contract. Any validation-doc change must update this test, not
   only README prose.

4. `README.md:71-73` says the repo contains `experiments/exp1` through
   `exp54`, while the worktree contains experiment scripts through
   `experiments/exp83_slot_attention.py`. This is stale navigation for future
   agents.

5. `README.md:111-114` says `demos/` has five headline runnable demos, while
   `README.md:59-68` lists eight demo files. The five-item conceptual ladder at
   `README.md:118-124` is different from the runnable demo inventory.

6. `docs/RUN_PROTOCOL.md:65-85` requires committed `results.json` plus
   `ADAPTER.md` pointers for remote-trained artifacts, and
   `docs/RUN_PROTOCOL.md:87-103` makes `ADAPTER.md` a structure invariant.
   `git ls-files 'experiments/exp*_data/ADAPTER.md'` found no adapter pointers.

7. `experiments/exp78_rule_induction.py:7-10` states the TL-only induction gate:
   recover gold rules from <=20 positive and <=20 negative examples in <=1s per
   target. A default local run with `--out /tmp/tensor-exp78-default-extension-results.json`
   recovered all rules and semantic equivalence, but distractor targets took
   about 3.66s each and the script printed `family search <=1s : FAIL`.

8. `experiments/exp78_rule_induction.py:170-179` runs brute-force induction and
   semantic equivalence after optional LM pruning. The distractor split at
   `experiments/exp78_rule_induction.py:219-220` expands the candidate cost to
   14,424 per target in the observed run.

9. `experiments/exp79_self_play_loop.py:436-482` defaults `--out` to
   `experiments/exp79_data/results.json`. The easy-mode no-write redirected run
   passed when output was sent to `/tmp`.

10. `experiments/exp79_lewm_tl.py:49`, `experiments/exp79_lewm_tl.py:519`, and
    `experiments/exp79_lewm_tl.py:594-596` also write into
    `experiments/exp79_data/`. This is the artifact namespace collision already
    identified by the workflow-handoff audit.

11. `experiments/exp83_slot_attention.py:10-11` sets a 90% probe gate, and
    `experiments/exp83_slot_attention.py:488-517` records `probe`, `tl_only`,
    and `e2e` gates. The checked-in `experiments/exp83_slot_data/results.json`
    fails the probe gate in prior audit evidence; the docs should not imply Slot
    Attention solved the exp79 limitation.

12. `experiments/exp83_slot_attention.py:113-114` imports
    `tensor_logic.research.slot_attention` and `scipy.optimize.linear_sum_assignment`;
    `pyproject.toml` declares `torch` runtime and only `matplotlib`, `numpy`,
    and `pytest` dev extras. `scipy` is an undeclared lane dependency.

13. `experiments/exp79_lewm_tl.py:53-110` and
    `experiments/exp83_slot_attention.py:53-110` duplicate the same scene and
    relation generation logic. The same duplication exists again for TL wiring:
    `experiments/exp79_lewm_tl.py:335-401` and
    `experiments/exp83_slot_attention.py:267-333`.

14. `tensor_logic/program.py:12-34` now has canonical `Atom` and `Rule` with
    variadic args and `negated=False` by default. `tensor_logic/rules.py:5-26`
    imports those canonical dataclasses for `<tl_rule>` parsing, so the
    2026-05-04 rule-language spec is partly implemented.

15. `tensor_logic/rules.py:60-155` still has an explicitly binary evaluator and
    head-variable restrictions for negated atoms. That is fine as a current
    boundary, but follow-up rule-language work must preserve this as a consumer
    limit, not redefine the canonical shape.

16. `tensor_logic/proofs.py:44-81` exposes binary positive and negative proof
    entrypoints, and `tensor_logic/proofs.py:249-305` binds exactly two head
    args. The arbitrary-arity design remains future work even though the
    canonical `Atom` is variadic.

17. `experiments/exp80_fafsa_kb.py:11-15` scopes FAFSA to the 2024-25 guide,
    Formula A dependent students, with Formula B/C marked TODO.

18. `experiments/exp80_fafsa_wizard.py:197-207` prints user-facing Pell
    qualification language. The current tests in `tests/test_exp80.py:15-77`
    cover synthetic invariants and regressions, not official ED worked examples.

19. `experiments/exp80_validate_synthetic.py:290-306` rewrites
    `experiments/exp80_spot_check_cases.json` as part of validation. In this
    run the file did not dirty the worktree, but future validation commands
    should note that this script writes a tracked artifact path.

20. `CLAUDE.md:1-16` tells agents to run `python tools/code_index.py --lookup`
    before planning changes under `tensor_logic/`. In this local shell that
    command fails because `python` is missing; `python3 tools/code_index.py
    --dump` and `--status` worked.

## Validation evidence

Commands run locally:

| Command | Result | Notes |
|---|---:|---|
| `llm-tldr tree .` | Pass | Mapped repo structure. |
| `python3 tools/code_index.py --dump` | Pass | Required before planning tasks touching `tensor_logic/`. |
| `python tools/code_index.py --dump` | Fail, exit 127 | `python` is not on PATH in this worker shell. |
| `python3 -m pytest tests/ -v` | Pass | `140 passed in 38.05s`. |
| `python3 experiments/exp80_validate_synthetic.py` | Pass | `1015/1015` families, invariants pass; writes spot-check JSON path. |
| `python3 experiments/exp78_rule_induction.py --out /tmp/tensor-exp78-default-extension-results.json` | Script exited 0 but falsification gate failed | All rules/equiv passed; `family search <=1s` failed due distractor targets around 3.66s. |
| `python3 experiments/exp79_self_play_loop.py --mode easy --out /tmp/tensor-exp79-easy-extension-results.json` | Pass | Easy mode answered 10/10 and printed `Overall: ALL PASS`. |
| `python3 experiments/exp78_rule_induction.py --help` | Pass | Supports `--out`, `--n-pos`, `--n-neg`, `--n-entities`, `--max-len`, `--lm`. |
| `python3 experiments/exp79_self_play_loop.py --help` | Pass | Supports safe `--out` and `--mode`. |
| `python3 experiments/exp83_slot_attention.py --help` | Pass after Matplotlib cache delay | Warned that user Matplotlib/font caches are not writable; help says `--skip-train` is not supported in this PoC. |
| `python3 -m tensor_logic --help` | Pass | CLI subcommands are present. |
| `python3 tools/code_index.py --status` | Pass | `tools/index.json` is fresh, ignored local state. |

The queue-required validation command is still `git status --short`; that is
recorded in the handoff below after this report write.

## Known blockers and risks

- No external verification was performed. Official FAFSA worked examples,
  current paper links, PyPI package-version reproduction, and remote adapter
  pointers remain non-local work.
- No remote jobs, model downloads, package downloads, deploys, pushes, or PRs
  were in scope.
- The local shell lacks `python`; docs and tests currently assert `python`
  command strings even though `python3` is the usable local interpreter.
- Committing from these worktrees may be blocked by git metadata outside the
  writable root, as previous tensor-experiments overnight reports observed.
- `exp78` currently exits 0 even when its printed falsification verdict is
  `FAIL`. Implementation work should decide whether that is intentional
  research behavior or whether a `--strict`/test gate is needed.
- `exp83 --help` imports Matplotlib and SciPy-era dependencies before showing
  help. In this sandbox it had to build a temporary Matplotlib cache because the
  default user cache directories were not writable.

## Implementation-ready follow-up tasks

### Task 1: Normalize the validation contract and local Python commands

Problem: local workers following README/CLAUDE literally hit `python: command
not found`, while CI uses `python` from `actions/setup-python`. The mismatch is
small but repeatedly blocks handoffs and code-index instructions.

Owned files:

- `README.md`
- `CLAUDE.md`
- `tests/test_packaging_ci.py`
- `.github/workflows/ci.yml` only if maintainers want CI wording updated

Acceptance criteria:

- README keeps the CI command but also documents a local macOS-safe `python3`
  command block.
- `CLAUDE.md` says to use `python3 tools/code_index.py --lookup <Symbol>` when
  `python` is absent.
- `tests/test_packaging_ci.py` asserts the intended dual contract instead of
  only checking the old `python` strings.
- No product code changes.

Validation:

```bash
python3 -m pytest tests/test_packaging_ci.py -q
python3 tools/code_index.py --status
git status --short
```

Stop condition: if maintainers want to require a venv that supplies `python`,
document that explicitly instead of switching examples wholesale to `python3`.

### Task 2: Add an experiment workflow and result provenance manifest

Problem: the repo has enough active experiments that future workers need a
single non-chat contract for entrypoints, default outputs, dependencies,
tracked artifacts, and safe smoke commands. The run protocol requires adapter
pointers, but current data directories do not have them.

Owned files:

- New `docs/EXPERIMENT_WORKFLOW.md` or `docs/experiment-manifest.md`
- `README.md` link to the manifest
- `docs/RUN_PROTOCOL.md` only for pointer/checker wording
- Optional `tests/test_packaging_ci.py` or new docs test

Acceptance criteria:

- Manifest lists at least exp53, exp54, exp60d, exp76c, exp78,
  exp79_self_play_loop, exp79_lewm_tl, exp80, exp81, and exp83.
- Each row includes entrypoint, default output path, safe `/tmp` output form,
  tracked artifacts, optional dependencies, external/network/model risk, and
  validation command.
- Existing `experiments/*_data/` dirs are annotated as one of: local data,
  checked-in result snapshot, remote-adapter pointer required, or legacy/no
  adapter expected.
- The manifest explicitly states that audit workers should not run experiments
  that write tracked result paths unless the issue asks for result refresh.

Validation:

```bash
python3 -m pytest tests/test_packaging_ci.py -q
git ls-files 'experiments/exp*_data/results.json' 'experiments/exp*_data/ADAPTER.md'
git status --short
```

Stop condition: if maintainers need to decide whether old `results.json` files
are canonical or historical snapshots, capture that as an open decision and do
not rewrite artifacts in this task.

### Task 3: Fix exp79/exp83 artifact and scene-utility drift

Problem: `exp79_self_play_loop.py` and `exp79_lewm_tl.py` share
`experiments/exp79_data/results.json`, while exp79 LeWM and exp83 duplicate
scene generation and TL wiring. That creates both artifact-overwrite risk and
test drift.

Owned files:

- `experiments/exp79_self_play_loop.py`
- `experiments/exp79_lewm_tl.py`
- `experiments/exp83_slot_attention.py`
- New or existing module under `tensor_logic/research/`
- `tests/test_exp79.py`
- New focused tests if output-path checks do not fit `test_exp79.py`
- Manifest/docs from Task 2, if already present

Acceptance criteria:

- Self-play and LeWM defaults no longer write the same `results.json`.
- Any migrated or renamed tracked results are documented; existing data is not
  silently overwritten.
- Shared scene constants, `generate_sequence`, `compute_relations`,
  `generate_labeled_frames`, `build_tl_state`, `remove_object`, and
  `compute_gt_derived` live in one `tensor_logic.research` module if both exp79
  LeWM and exp83 still use them.
- Tests assert default output paths are distinct without training.
- `exp83 --help` remains cheap and does not require training.

Validation:

```bash
python3 -m pytest tests/test_exp79.py -q
python3 experiments/exp79_self_play_loop.py --mode easy --out /tmp/exp79-self-play-smoke.json
python3 experiments/exp83_slot_attention.py --help
git status --short --ignored
```

Stop condition: if historical links require `experiments/exp79_data/results.json`
to stay as the self-play artifact, preserve that path and move only the LeWM
default.

### Task 4: Make exp78 falsification and performance actionable

Problem: the default TL-only exp78 run exits 0 but prints a failed falsification
verdict in this environment because distractor targets exceed the <=1s search
gate. This is exactly the kind of research result that should turn into either
an optimization slice or an explicit revised gate.

Owned files:

- `experiments/exp78_rule_induction.py`
- `tensor_logic/research/utils.py`
- `tests/test_exp81.py` or a new `tests/test_exp78.py`
- `docs/exp78_rule_induction_spec.md`
- `notes/EXPERIMENTS.md` only if the claim/gate is intentionally revised

Acceptance criteria:

- Add a test or script-level strict mode that makes the falsification verdict
  machine-checkable without parsing human text.
- Either optimize/prune TL-only distractor search so the default run satisfies
  the <=1s gate on a normal local machine, or update the spec/results to state
  the current gate honestly and define the next optimization target.
- Preserve semantic-equivalence checks; do not trade correctness for speed.
- The safe validation command writes results to `/tmp`, not the tracked
  `experiments/exp78_data/results.json`.

Validation:

```bash
python3 experiments/exp78_rule_induction.py --out /tmp/exp78-default-results.json
python3 -m pytest tests/test_exp81.py -q
python3 -m pytest tests/test_tensor_logic_core.py -k "rule or proof" -q
git status --short
```

Stop condition: if timing depends too heavily on machine class, make the test
assert candidate cost or relative speedup instead of wall-clock seconds, then
record the decision in the spec.

### Task 5: Add FAFSA safety and official-parity gate before user-facing claims

Problem: exp80 is a useful real-world proof-engine prototype, but the wizard is
worded like an estimator and tells users whether they likely qualify for Pell
aid. Current tests prove internal invariants and synthetic cases, not official
ED worked-example parity.

Owned files:

- `experiments/exp80_fafsa_kb.py`
- `experiments/exp80_fafsa_wizard.py`
- `experiments/exp80_validate_synthetic.py`
- `tests/test_exp80.py`
- New docs file under `docs/` if a validation matrix is added

Acceptance criteria:

- Wizard output clearly states research/demo status, 2024-25 only, Formula A
  dependent-student only, and not financial-aid advice.
- User-facing "likely qualify" wording is softened or gated behind an explicit
  official-parity status.
- Add a validation matrix distinguishing synthetic invariants, named
  spot-check cases, and official ED worked examples.
- Add at least one test that locks the disclaimer/boundary text if the wizard
  remains interactive.
- Do not add Formula B/C in this task; the scope is safety and validation
  boundary, not coverage expansion.

Validation:

```bash
python3 -m pytest tests/test_exp80.py -q
python3 experiments/exp80_validate_synthetic.py
git status --short
```

Stop condition: if official ED examples must be fetched from the web or manually
transcribed, stop at the validation matrix and create a separate data-gathering
task with source citations.

## Work that should not move into this implementation queue yet

- Re-running exp53/exp54 headline package downloads. Those need frozen package
  versions, safe extraction, and network approval first.
- Remote/Kaggle SFT or adapter runs. The run protocol is not yet enforced and
  adapter pointers are missing.
- Exp83 full training. It is exploratory, has an undeclared SciPy dependency,
  and its checked-in result already shows the probe gate did not pass.
- Arbitrary-arity proof expansion. The canonical `Atom` shape supports it, but
  the proof and evaluator consumers are still explicitly binary.
- Public serving of the HTTP API or workbench. The risk register's include and
  timeout guardrails should land first.

## Handoff

Changed files:

- `docs/overnight/2026-05-07-30min-extension/tensor-experiments-action-plan.md`

Product code changed: no.

External services used: no.

PR created: no, out of scope.

Commit created: no. Current HEAD remains
`5565791a8c888511bbbff1107d2d357164f88baa`.

Required validation command:

```bash
git status --short
```

Observed result after the report write:

```text
?? docs/overnight/
```

The exact untracked file is:

```text
?? docs/overnight/2026-05-07-30min-extension/tensor-experiments-action-plan.md
```

Blockers:

- No blocker to writing the report.
- Local docs-only commit was attempted and blocked by sandbox permissions:
  `git add docs/overnight/2026-05-07-30min-extension/tensor-experiments-action-plan.md`
  failed with `fatal: Unable to create
  '/Users/jwalinshah/projects/tensor/experiments/.git/worktrees/tensor-experiments-30min-action-plan/index.lock':
  Operation not permitted`.
