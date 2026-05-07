# tensor-experiments docs-claims audit

Queue item: `tensor-experiments-docs-claims`
Branch: `codex/goal-tensor-experiments-docs-claims`
Repo path: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-experiments-docs-claims`
Audit focus: README/docs claims, supported evidence, unsupported claims, and next safe docs work.

## Decision summary

This repo is a Python research workspace plus reusable `tensor_logic` package. The docs are useful and unusually candid about failed experiments, but several claims are now stale or depend on non-local evidence. I made no product-code changes and did not run external downloads, model loads, deploys, pushes, or PR creation.

I wrote only this report. Local validation and demo smoke checks used `python3` because this shell has no `python` command.

## Repo state

- Purpose from `README.md:1-7`: a learning/research project around Tensor Logic and cognition, not a product.
- Shared-library claim from `README.md:3`: the repo now serves as a shared `tensor_logic` dependency for other workspace projects.
- Current branch observation: `git status --short --branch` initially printed `## codex/goal-tensor-experiments-docs-claims` with no dirty files.
- Current HEAD observation: `git rev-parse HEAD` printed `5565791a8c888511bbbff1107d2d357164f88baa`.
- Remote observation: `git remote -v` points to `https://github.com/jwalin-shah/tensor-logic.git` for fetch and push.
- Recent history observation: `git log --oneline -5` starts with `5565791 Refactor: shared prove path, exp78 dedupe, reason helpers, core module (#31)`.
- Structure observation: `llm-tldr tree .` found package code in `tensor_logic/`, many scripts in `experiments/`, demos in `demos/`, test files in `tests/`, and docs under `docs/`, `notes/`, `benchmarks/`, and `web_workbench/`.
- Experiment surface observation: `git ls-files 'experiments/exp*.py' | wc -l` printed `85`, and `git ls-files 'experiments/exp*.py' | tail -20` includes scripts through `experiments/exp83_slot_attention.py`.
- Docs planning surface observation: `git ls-files 'docs/superpowers/specs/*.md' 'docs/superpowers/plans/*.md' | wc -l` printed `13`.
- Dirty-state observation before writing this report: `git status --short` printed nothing.

## Supported claims

These docs claims have direct local support from files and/or commands I ran.

1. `tensor_logic/` is an importable, reusable package.
   - `pyproject.toml:23-24` packages `tensor_logic*`.
   - `tensor_logic/__main__.py:13-41` exposes CLI subcommands for `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`.
   - Command: `python3 -m tensor_logic --help` exited 0 and listed the CLI subcommands.

2. Worker validation is encoded in README, CI, and a packaging test.
   - `README.md:85-92` documents `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`.
   - `.github/workflows/ci.yml:24-30` runs the same install and test commands on GitHub Actions.
   - `tests/test_packaging_ci.py:16-42` asserts the pyproject, CI workflow, and README validation contract.
   - Command: `python3 -m pytest tests/test_packaging_ci.py -q` passed: `3 passed in 0.01s`.

3. The `tensor_logic` core has real local test coverage.
   - `tests/test_tensor_logic_core.py` covers queries, proof trees, negative proof behavior, includes, recursion, and source locations.
   - Command: `python3 -m pytest tests/test_tensor_logic_core.py -q` passed: `54 passed in 14.73s`.

4. The web workbench docs match the server shape, with one portability caveat noted below.
   - `web_workbench/README.md:1-13` describes a local browser shell that writes editor contents to a temp `.tl` file and invokes `python -m tensor_logic`.
   - `web_workbench/server.py:53-83` writes a temporary `.tl` file and uses `sys.executable -m tensor_logic`.
   - Command: `python3 -m pytest tests/test_web_workbench.py -q` passed: `2 passed in 9.83s`.

5. The main README's demo inventory is mostly runnable locally with `python3`.
   - `README.md:94-107` lists eight demos and says they run on CPU in seconds to a few minutes.
   - Commands run successfully: `python3 demos/transitive_closure.py`, `python3 demos/tensor_language.py`, `python3 demos/program_rules.py`, `python3 demos/provenance_kb.py`, `python3 demos/train_kg.py`, `python3 demos/joint_lm_kg.py`, `python3 demos/throwing.py`, and `python3 demos/catastrophic_forgetting.py`.
   - Sample outputs included transitive-closure fixpoint output, program-rule queries, provenance proof trees, KG training to 1.000 accuracy, throwing final MSE `0.0218`, joint LM+KG final cell-wise accuracy `1.000`, and EWC retaining Task A at `0.998`.

6. The proof/reasoning feature claims are grounded in code, not only prose.
   - `tensor_logic/program.py:33-90` implements domains, relations, facts, rules, queries, and fixpoints.
   - `tensor_logic/proofs.py:35-85` implements positive and negative proof entrypoints.
   - `tensor_logic/closure.py:6-46` implements dense and BFS reachability closures.
   - `README.md:73` claims named-index language, dense/sparse closure, Datalog-style joins, stratified negation, provenance proof trees, and semiring helpers; this is directionally supported by package files and tests.

7. The exp78 spec and checked-in results are mostly aligned.
   - `docs/exp78_rule_induction_spec.md:21-29` defines the TL-only and LM-pruner falsification criteria.
   - `docs/exp78_rule_induction_spec.md:103-109` lists expected deliverables including `experiments/exp78_rule_induction.py` and `experiments/exp78_data/results.json`.
   - `experiments/exp78_rule_induction.py:51-142` implements `lm_prune()` with MLX/transformers fallback behavior.
   - `experiments/exp78_data/results.json` exists and shows `f1: 1.0`, `semantic_equiv: 1.0`, and search times under 1 second for the listed targets; the distractor `uncle` row reports `pruner_speedup: 14.07`.

8. The exp79 implementation exists despite one stale design-doc status line.
   - `docs/superpowers/specs/2026-04-28-exp79-lewm-tl-design.md:1-6` says status is "pending implementation".
   - `experiments/exp79_lewm_tl.py:1-14` exists and describes the implemented experiment.
   - `experiments/exp79_lewm_tl.py:528-599` contains a runnable `main()` that writes `experiments/exp79_data/results.json`.
   - `experiments/exp79_data/results.json` exists.

9. A broad local suite, excluding one model-dependent proposer smoke test, passes.
   - Command: `python3 -m pytest tests/ -q -k 'not make_proposer'` passed: `139 passed, 1 deselected in 33.22s`.
   - The deselected test is `tests/test_exp81.py::test_make_proposer_returns_valid_json`, which calls the LM proposer path through `make_proposer()`.

## Partially supported or stale claims

1. README's headline numeric tables are plausible but not locally reproducible from checked-in result artifacts alone.
   - `README.md:11-25` claims a 3-scalar recurrence generalizes to eight OSS import graphs, with package-level F1 numbers.
   - `README.md:27-40` claims larger-package F1 numbers up to `n=1,532`.
   - `experiments/exp53_real_imports.py:1-20` and `experiments/exp54_big_imports.py:1-15` describe pipelines that download package source tarballs with `pip download`, train models, and compute those tables.
   - There is no checked-in `experiments/exp53_data/results.json` or `experiments/exp54_data/results.json`; `git ls-files 'experiments/exp*_data/results.json'` returned only exp78, exp79, and exp83 results.
   - Risk: the README presents fixed numeric claims whose reproduction depends on current package versions and network downloads unless frozen manifests are added.

2. README's "Repo layout" is stale.
   - `README.md:111-115` says `experiments/ exp1..exp54`.
   - Local command output shows 85 `experiments/exp*.py` files, including scripts through `exp83`.
   - `notes/EXPERIMENTS.md:9-11` logs recent experiment rows 80, 79, and 78, so the stale layout is isolated mostly to README.

3. README describes "5 headline runnable demos" while its table lists eight demo files.
   - `README.md:57-69` lists eight demo files.
   - `README.md:111-113` says `demos/` contains "5 headline runnable demos".
   - Interpretation: five may be the conceptual ladder in `README.md:118-124`, but the wording can confuse workers deciding what is canonical.

4. The exp79 design-doc gate does not match current code and results.
   - Design doc gate: `docs/superpowers/specs/2026-04-28-exp79-lewm-tl-design.md:63-64` says above/left_of validation accuracy must be at least 90%.
   - Current code gate: `experiments/exp79_lewm_tl.py:568-570` uses at least 70% for above/left_of and prints a Slot Attention suggestion below that.
   - Result file exists, but the design status line still says pending implementation at `docs/superpowers/specs/2026-04-28-exp79-lewm-tl-design.md:6`.

5. exp83 results include a failed probe gate that is easy to miss from docs.
   - `experiments/exp83_slot_attention.py:7-11` states the Slot Attention claim and falsification criteria.
   - `experiments/exp83_slot_attention.py:488-517` writes gates into `experiments/exp83_slot_data/results.json`.
   - `experiments/exp83_slot_data/results.json` reports `gates.probe: false`, with `above_acc` about `0.858` and `left_of_acc` about `0.872`, while TL-only and end-to-end gates are true.
   - Risk: morning readers may infer Slot Attention solved the exp79 limitation unless docs explicitly say the probe gate failed.

6. Run-protocol provenance invariants are not fully followed by existing data dirs.
   - `docs/RUN_PROTOCOL.md:65-85` says local mirrors should commit `results.json` plus `ADAPTER.md`.
   - `docs/RUN_PROTOCOL.md:87-103` repeats `ADAPTER.md` as a repo-structure invariant.
   - Command: `git ls-files 'experiments/exp*_data/ADAPTER.md'` printed nothing.
   - Existing data dirs include `experiments/exp60_data/`, `experiments/exp76_data/`, `experiments/exp78_data/`, `experiments/exp79_data/`, and `experiments/exp83_slot_data/`; only exp78/79/83 have checked-in `results.json`.

7. Local command portability is weaker than the docs imply.
   - `README.md:89-91`, `.github/workflows/ci.yml:26-30`, and `web_workbench/README.md:7-13` use `python`.
   - Command: `command -v python` exited 1 with no output.
   - Command: `python -m pytest tests/test_packaging_ci.py -q` failed locally with exit 127: `/opt/homebrew/bin/bash: line 1: python: command not found`.
   - Command: `command -v python3` printed `/usr/local/bin/python3`, and the equivalent `python3` test commands passed.
   - CI may be fine because `actions/setup-python` usually provides `python`, but local docs should account for this worktree shell.

8. Optional model/backend claims need dependency framing.
   - `CLAUDE.md:21-28` says `lm_prune()` supports MLX and transformers, with a known VLM limitation.
   - `pyproject.toml:12-21` declares only `torch` plus dev dependencies `matplotlib`, `numpy`, and `pytest`; it does not declare `transformers`, `mlx_lm`, `scipy`, `peft`, `datasets`, or `accelerate`.
   - `experiments/exp78_rule_induction.py:60-63` handles missing `transformers` by falling back to all relations.
   - `experiments/exp83_slot_attention.py:113-115` imports `scipy.optimize.linear_sum_assignment`, but `scipy` is not in `pyproject.toml`.
   - Risk: model-backed docs claims are true only in enriched environments, not from the documented base install.

## Unsupported or non-local claims

These claims may be true, but I did not find enough local evidence to treat them as self-contained.

- External paper/citation claims in `README.md:5`, `README.md:155-176`, and related notes were not web-verified during this local audit.
- README's exp53/exp54 package-level F1 tables are supported by scripts, not immutable checked-in run outputs.
- Notes rows describing remote SFT/Kaggle/Colab outcomes, especially exp60d and exp76-related data, do not have local adapter pointers or manifests in the repo. The report should treat them as log claims until the run protocol is applied retroactively.
- `README.md:107` says each listed demo runs in seconds to a few minutes. I verified all eight demos with `python3` in this environment, but I did not run the exact `uv run --with torch python ...` commands from the README.
- `README.md:3` says `fafsa-engine` and later `officeqa-*` consume this repo as a shared dependency. I did not inspect sibling repos because this queue item is scoped to the current repo and branch.

## Risks and stale assumptions

1. Numeric headline claims can drift with network state.
   - exp53/54 download package tarballs at run time, so package versions and source layouts can change unless results are frozen with input package versions and manifests.

2. Provenance policy is stricter than repo reality.
   - The run protocol requires `ADAPTER.md` pointers, but none are tracked. Morning review should not assume all remote-trained claims are reproducible from this checkout.

3. README is behind the experiment surface.
   - It still describes exp1..exp54 even though local files and notes reach exp83. New agents may miss later work or duplicate already-run experiments.

4. Exp79/exp83 status is ambiguous.
   - Exp79's design doc says pending implementation and uses a 90% probe gate; current code uses 70%. Exp83 appears to be the object-centric follow-up, but its checked-in results show the probe gate failed.

5. The default install contract is too small for some documented experiments.
   - Base/dev dependencies do not cover `transformers`, `mlx_lm`, `scipy`, or SFT tooling mentioned in docs and experiment scripts.

6. Local validation docs assume a `python` executable.
   - This shell has only `python3`, so copied worker commands fail locally even though CI likely passes.

7. Large experiment scripts are not cheap proof commands.
   - exp53/54 retrain and download external packages. They are not suitable as overnight validation commands without network approval and frozen expected outputs.

8. Checked-in result files lack a uniform schema.
   - exp78, exp79, and exp83 all have `results.json`, but shapes differ and only some include gates. A review script cannot yet summarize them consistently.

## Next safe work

These are independently grabbable, docs-first tasks. None require product-code changes unless the assignee chooses to add docs tests.

1. Reconcile README scope and current experiment surface.
   - Acceptance criteria: `README.md` accurately states that experiments now run through exp83; the demos wording distinguishes the eight runnable demos from the five-item conceptual ladder; headline tables explicitly say whether results are frozen artifacts or reproducible script outputs.
   - Validation: `python3 -m pytest tests/test_packaging_ci.py -q`; `python3 demos/transitive_closure.py`; `git diff -- README.md`.

2. Add a result provenance index for headline experiments.
   - Acceptance criteria: create or update a docs file that maps exp44/47/52/53/54/60d/78/79/80/83 claims to script path, checked-in artifact path, exact command, external inputs, and whether the run is locally reproducible without network/model access.
   - Validation: `git ls-files 'experiments/exp*_data/results.json'`; `python3 -m pytest tests/test_packaging_ci.py -q`; optional docs test that required headline result rows include artifact status.

3. Normalize local Python command documentation.
   - Acceptance criteria: README, web workbench docs, and agent docs either use `python3` for local commands or state that `python` means the active virtualenv interpreter; docs mention `python3 -m pip install -e ".[dev]"` as the macOS-safe local form.
   - Validation: `command -v python3`; `python3 -m tensor_logic --help`; `python3 -m pytest tests/test_packaging_ci.py -q`.

4. Resolve exp79 and exp83 status docs.
   - Acceptance criteria: exp79 design/spec docs state implemented status and explain the current 70% probe gate; exp83 docs or notes explicitly record `gates.probe=false` and what that means for the Slot Attention claim.
   - Validation: `python3 -m pytest tests/test_exp79.py -q`; `python3 -m pytest tests/ -q -k 'not make_proposer'`; inspect `experiments/exp79_data/results.json` and `experiments/exp83_slot_data/results.json`.

5. Backfill remote-run provenance pointers or mark non-applicable.
   - Acceptance criteria: every `experiments/exp*_data/` directory has either an `ADAPTER.md`/manifest pointer or a short `PROVENANCE.md` explaining why no adapter is expected; exp60/76 remote-training claims link to durable artifact references if available.
   - Validation: `git ls-files 'experiments/exp*_data/ADAPTER.md' 'experiments/exp*_data/PROVENANCE.md'`; no external download required.

6. Define optional dependency groups for model-heavy experiments.
   - Acceptance criteria: docs clearly separate base package install, dev/test install, web/workbench install, LM-pruner install, SFT install, and Slot Attention/scipy install; experiments with optional deps fail with actionable messages.
   - Validation: `python3 -m pytest tests/test_packaging_ci.py -q`; targeted import smoke tests for base package still pass without optional extras.

## Validation command candidates

| Command | Status observed or expected | Notes |
|---|---|---|
| `git status --short` | Required queue validation; exits 0. After this report is written, expected output is the report path as untracked/modified. | This is the queue item's required validation command. |
| `python -m pytest tests/test_packaging_ci.py -q` | Observed fail locally, exit 127. | `python` is not on PATH in this shell. |
| `python3 -m pytest tests/test_packaging_ci.py -q` | Observed pass: `3 passed in 0.01s`. | Validates README/CI/package validation docs. |
| `python3 -m pytest tests/test_tensor_logic_core.py -q` | Observed pass: `54 passed in 14.73s`. | Cheap core proof/package confidence. |
| `python3 -m pytest tests/test_web_workbench.py -q` | Observed pass: `2 passed in 9.83s`. | Supports web workbench docs. |
| `python3 -m pytest tests/ -q -k 'not make_proposer'` | Observed pass: `139 passed, 1 deselected in 33.22s`. | Avoids one LM/model-dependent proposer smoke test. |
| `python3 -m tensor_logic --help` | Observed pass. | Confirms CLI surface. |
| `python3 demos/*.py` individually | Observed pass for all eight README demos when run as individual commands. | I did not use the exact `uv run --with torch python ...` wrapper. |
| `python3 experiments/exp53_real_imports.py` | Expected not safe for this queue item. | Downloads package source tarballs and retrains models; external network/provenance work. |
| `python3 experiments/exp54_big_imports.py` | Expected not safe for this queue item. | Same external/download issue, plus larger package compute. |
| `python3 experiments/exp83_slot_attention.py --skip-train` | Expected not useful. | Code help says skip-train is not supported in this PoC yet. |

## Commands run

- `pwd`
- `git status --short --branch`
- `git status --short`
- `git rev-parse --show-toplevel`
- `git rev-parse HEAD`
- `git remote -v`
- `git log --oneline -5`
- `llm-tldr tree .`
- `rg --files`
- `fd -a 'AGENTS.md|CLAUDE.md|README|docs|pyproject|package.json|requirements|Makefile|Dockerfile|compose|.env|ISSUE.md' .`
- `rtk read CLAUDE.md`
- `rtk read README.md`
- `rtk read pyproject.toml`
- `rtk read benchmarks/README.md`
- `rtk read web_workbench/README.md`
- `rtk read docs/RUN_PROTOCOL.md`
- `rtk read docs/SYMPHONY_RUN_PROTOCOL.md`
- `rtk read docs/exp78_rule_induction_spec.md`
- `rtk read docs/agents/domain.md`
- `rtk read docs/agents/issue-tracker.md`
- `rtk read docs/agents/triage-labels.md`
- `rtk grep "TODO|FIXME|Known limitation|Out of scope|not implemented|skip|xfail|TODO|requires|Kaggle|external|weights|results.json|ADAPTER|python -m pytest|uv run|pip install" .`
- `rg -n "^#|python -m|uv run|pytest|pip install|results|F1|accuracy|seconds|minutes|Kaggle|Out of scope|Known limitation|TODO|FIXME|not implemented|does not|fails|failed|unsupported|stable|runnable|should|must" README.md docs notes benchmarks web_workbench tests pyproject.toml CLAUDE.md`
- `rtk read experiments/exp53_real_imports.py`
- `rtk read experiments/exp54_big_imports.py`
- `rtk read experiments/exp52_mlp_capacity.py`
- `rtk read notes/EXPERIMENTS.md`
- `git ls-files 'experiments/*_data/*'`
- `git ls-files '*ADAPTER.md'`
- `rtk read experiments/exp78_data/results.json`
- `rtk read experiments/exp79_data/results.json`
- `rtk read experiments/exp83_slot_data/results.json`
- `rg -n "exp53|exp54|exp52|exp78|exp79|exp83|ADAPTER|manifest|results\\.json|validation|Worker validation|Run|Repo layout|demos/|experiments/" README.md notes/EXPERIMENTS.md docs/RUN_PROTOCOL.md docs/exp78_rule_induction_spec.md docs/superpowers/specs docs/superpowers/plans CLAUDE.md tests`
- `rg --files -g '.github/**' -g '!.git/**'`
- `rtk read tests/test_packaging_ci.py`
- `rtk read .github/workflows/ci.yml`
- `python -m pytest tests/test_packaging_ci.py -q`
- `python -m pytest tests/test_tensor_logic_core.py -q`
- `python3 --version`
- `python3 -m pytest tests/test_packaging_ci.py -q`
- `python3 -m pytest tests/test_tensor_logic_core.py -q`
- `python3 -m pytest tests/test_web_workbench.py -q`
- `python3 -m pytest tests/ -q -k 'not make_proposer'`
- `rtk read tensor_logic/__main__.py`
- `rtk read tensor_logic/program.py`
- `rtk read tensor_logic/proofs.py`
- `rtk read tensor_logic/closure.py`
- `rtk read web_workbench/server.py`
- `rtk read tensor_logic/http_api.py`
- `nl -ba README.md`
- `nl -ba docs/RUN_PROTOCOL.md`
- `nl -ba CLAUDE.md`
- `nl -ba docs/exp78_rule_induction_spec.md`
- `nl -ba experiments/exp83_slot_attention.py`
- `nl -ba docs/superpowers/specs/2026-04-28-exp79-lewm-tl-design.md`
- `nl -ba experiments/exp79_lewm_tl.py`
- `nl -ba experiments/exp78_rule_induction.py`
- `nl -ba experiments/exp81_sweep.py`
- `nl -ba tests/test_exp79.py`
- `nl -ba tests/test_exp81.py`
- `rtk read experiments/exp81_optimize_rule_induction.py`
- `git ls-files 'experiments/exp*.py' | wc -l`
- `git ls-files 'experiments/exp*.py' | tail -20`
- `git ls-files 'docs/superpowers/specs/*.md' 'docs/superpowers/plans/*.md' | wc -l`
- `git ls-files 'docs/superpowers/specs/*.md' 'docs/superpowers/plans/*.md'`
- `git ls-files 'experiments/exp*_data/results.json'`
- `git ls-files 'experiments/exp*_data/ADAPTER.md'`
- `nl -ba notes/EXPERIMENTS.md | sed -n '1,35p'`
- `nl -ba .github/workflows/ci.yml`
- `nl -ba tests/test_packaging_ci.py`
- `nl -ba web_workbench/README.md`
- `nl -ba web_workbench/server.py | sed -n '1,120p'`
- `command -v python`
- `command -v python3`
- `command -v uv`
- `python3 -m tensor_logic --help`
- `python3 demos/transitive_closure.py`
- `python3 demos/program_rules.py`
- `python3 demos/tensor_language.py`
- `python3 demos/provenance_kb.py`
- `python3 demos/train_kg.py`
- `python3 demos/throwing.py`
- `python3 demos/joint_lm_kg.py`
- `python3 demos/catastrophic_forgetting.py`

## Non-goals

- I did not edit source code, tests, README, package metadata, run protocol docs, or experiment artifacts.
- I did not install packages, run `pip download`, launch remote jobs, use Kaggle, use model providers, create external artifacts, push, or open a PR.
- I did not inspect sibling repos such as `fafsa-engine`; sibling validation is out of scope for this queue item.
- I did not mark any external tracker done.
- I did not rerun expensive or network-dependent experiment scripts like exp53/54.

## Unknowns

- Whether the README's exp53/exp54 numeric tables still reproduce with current PyPI source tarballs.
- Whether remote-trained exp60/76 adapters still exist and where their canonical datasets live.
- Whether GitHub Actions currently passes on `origin/main`; local CI metadata is present, but I did not query GitHub.
- Whether the external paper and reference links are current and correctly cited.
- Whether downstream repos actually consume `tensor_logic` from this checkout or from a package/pinned commit.
- Whether `python` is available in the maintainer's normal interactive shell; it is absent in this worker shell.
