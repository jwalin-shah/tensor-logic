# tensor-logic docs-claims audit

Queue item: `tensor-logic-docs-claims`  
Repo path: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-logic-docs-claims`  
Branch observed: `codex/goal-tensor-logic-docs-claims`  
Focus: docs claims, supported evidence, unsupported claims, and safe next work.

## Executive summary

`tensor-logic` is a research repo plus reusable Python package for tensor-logic experiments, proof/query execution, and support/stability experiments. The strongest current claims are backed by local code and tests: reusable `tensor_logic` exports, TL file parsing, CLI/HTTP/workbench execution, proof trees, stratified negation, support/stability perfect-state evaluation, and committed support/stability result JSON for experiments 87-95.

The main stale surface is `README.md`. It still frames the repo as `experiments/exp1` through `exp54` and says `demos/` has five headline runnable demos, while the checkout contains experiments and tests through `exp95` and eight demo scripts. The README headline transitive-closure tables cite runnable scripts but do not have committed result JSON/manifests for `exp53_real_imports.py` or `exp54_big_imports.py`; rerunning them requires network-dependent `pip download` of external packages. The support/stability docs and `notes/EXPERIMENTS.md` are much fresher and preserve caveats, but the root README has not absorbed those newer claims.

## Commands run

- `pwd` confirmed the worktree path.
- `git status --short --branch` initially reported only `## codex/goal-tensor-logic-docs-claims`, with no dirty files.
- `git rev-parse --show-toplevel` confirmed the repo root is the worktree root.
- `git log --oneline -5` showed current HEAD at `c9e2cfc Merge pull request #46 from jwalin-shah/codex/SYM-220-scored-object-hypotheses`.
- `llm-tldr tree .` showed the top-level surfaces: `README.md`, `tensor_logic/`, `experiments/`, `tests/`, `docs/`, `notes/`, `web_workbench/`, `phase_training/`, `examples/`, and `benchmarks/`.
- `rg --files` showed experiments and tests through `exp95`, including committed result files under `experiments/exp87_support_data/` through `experiments/exp95_scored_object_hypotheses_data/`.
- `find .. -maxdepth 3 -name repos.json -print` found sibling goal-pack repo manifests but this audit stayed scoped to this worktree.
- `rg -n` and `nl -ba` were used across README, docs, tests, package modules, demos/examples, and result JSONs to cite local evidence.

## Repo purpose and state

- Purpose from `README.md:1-7`: "Tensor Logic -> Cognition", a learning/research project, not a product, and now also a shared `tensor_logic` library dependency for sibling projects.
- Package contract from `pyproject.toml:5-24`: project name `tensor-logic`, Python `>=3.11`, runtime dependency `torch>=2.0`, dev extras `matplotlib`, `numpy`, `pytest`, and setuptools package include `tensor_logic*`.
- CI contract from `.github/workflows/ci.yml:24-30`: installs `".[dev]"` and runs `python -m pytest tests/ -v` on pull requests and `main` pushes.
- Project instruction from `CLAUDE.md:1-17`: before implementation plans touching `tensor_logic/`, run `python tools/code_index.py --lookup <RelevantSymbol>`; test suite command is `pytest tests/ -v`.
- Initial dirty state: clean branch, observed by `git status --short --branch`.
- Expected final dirty state for this queue item: one docs report at `docs/overnight/tensor-logic-docs-claims.md`.

## Claims audit

### Root README claims

Supported or mostly supported:

- The reusable-package claim is supported. `README.md:73` says `tensor_logic/` is a reusable extraction with named-index language, dense/sparse closure, binary Datalog-style joins, stratified negation, provenance proof trees, and semiring helpers. Local code backs this: package exports in `tensor_logic/__init__.py:3-66`, narrow semantics surface in `tensor_logic/core.py:1-30`, named-index relations/fixpoint in `tensor_logic/language.py:123-197`, closure helpers in `tensor_logic/closure.py:6-53`, rule parsing/evaluation with negation in `tensor_logic/rules.py:13-162`, proof/negative-proof types in `tensor_logic/proofs.py:13-118`, and semiring helpers in `tensor_logic/semirings.py:4-15`.
- CLI/file-format claims are supported. `tensor_logic/__main__.py:13-90` exposes `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`. `tensor_logic/file_format.py:32-82` loads domains, relations, facts, rules, includes, queries, and proofs. `examples/code_dependencies.tl:1-20` and `examples/permissions.tl:1-19` are concrete local `.tl` examples. `tests/test_tensor_logic_core.py:173-179` verifies loading `examples/code_dependencies.tl`; `tests/test_tensor_logic_core.py:726-787` verifies CLI `run`, `query`, and JSON `prove`.
- The support/stability validation fast path in `README.md:85-102` is backed by packaging and CI tests. `tests/test_packaging_ci.py:16-25` checks pyproject dependencies/package include, `tests/test_packaging_ci.py:28-35` checks CI runs the full worker validation, and `tests/test_packaging_ci.py:38-44` checks README documents the commands.
- The CPU support/stability lane is well backed by newer files. `docs/superpowers/plans/2026-05-05-support-stability.md:7-24` states the perfect-object-state scope and non-goals. `experiments/exp84_support_data.py:1-4` is pure geometry, exact labels, no learned components. `experiments/exp85_support_tl.py:1-6` is the deterministic support/stability engine, and `experiments/exp85_support_tl.py:189-279` implements fixpoint-style stability, support proofs, and fall labels. `experiments/exp87_support_data/results.json` records full-run TL accuracy and gates as passed, including `experiment` at line 43, gates at lines 44-73, thesis `passed` at line 206, and TL split accuracies at lines 207-257.
- The web workbench README is supported. `web_workbench/README.md:1-13` says it is a minimal browser shell that writes editor contents to a temporary `.tl` file and invokes `python -m tensor_logic`. `web_workbench/server.py:44-84` does exactly that, and `tests/test_web_workbench.py:20-32` verifies query API behavior.

Stale, ambiguous, or weakly supported:

- `README.md:71` says the repo contains `experiments/exp1`-`exp54`; `rg --files` shows experiments and tests through `exp95`, and `notes/EXPERIMENTS.md:7-20` has current rows 95 down to 78 at the top.
- `README.md:121-123` says `demos/` has "5 headline runnable demos" and `experiments/` is `exp1..exp54`. The table at `README.md:61-68` actually lists eight demo scripts, and the file tree has those eight under `demos/`.
- `README.md:117` says each demo runs on CPU in seconds to a few minutes. This is plausible for the listed local scripts but was not validated in this audit. It is also broad because `demos/joint_lm_kg.py`, `demos/throwing.py`, and `demos/catastrophic_forgetting.py` depend on `torch` training loops rather than pure deterministic checks.
- `README.md:11-25` and `README.md:27-40` publish exact transitive-closure tables for external packages. The backing scripts exist (`experiments/exp53_real_imports.py:1-23`, `experiments/exp54_big_imports.py:1-18`), but no committed `exp53` or `exp54` result JSON was found by `find experiments -maxdepth 2 -name 'results*.json' -print`. Reproduction requires networked `pip download` in `experiments/exp53_real_imports.py:106-133` and `experiments/exp54_big_imports.py:98-128`, so the exact README numbers are not independently auditable from committed local artifacts alone.
- `README.md:136-138` says one tensor equation expresses transformers, GNNs, RNNs, Datalog programs, probabilistic graphical models, and kernel machines, with GPU/autodiff plus sound logical semantics. Local code supports small pieces of this broad framing, but the statement is essay-level and not scoped to what this package currently implements. It should be marked conceptual, not an implementation claim.
- `README.md:154` says "The pieces exist. The integration is the open problem." The repo contains many pieces, but local evidence also records failed/partial pieces and hard caveats in `notes/EXPERIMENTS.md:100-109`.

### Notes and experiment-log claims

- `notes/EXPERIMENTS.md:1-7` is a maintained experiment log with status legend and a "do not delete rows" instruction. This is useful claim hygiene because failed and superseded experiments stay visible.
- Recent support/stability rows are backed by committed scripts, tests, and result JSONs. Rows 87-95 at `notes/EXPERIMENTS.md:9-17` cite exact local result files from `experiments/exp87_support_data/results.json` through `experiments/exp95_scored_object_hypotheses_data/results.json`.
- `notes/EXPERIMENTS.md:17` makes the clean V1 support/stability claim and explicitly says it is not yet a pixel or learned-rule result. This matches `docs/superpowers/plans/2026-05-05-support-stability.md:17-24`.
- `notes/EXPERIMENTS.md:16` records deterministic TL brittleness under tiny coordinate noise, and rows 89-95 then narrow the interface toward confidence, intervals, pixel stubs, detector anomaly signals, object hypotheses, and scored non-oracle hypotheses. This is fresher than the root README and should be promoted into a "current research lane" section.
- The transitive-closure positive claim has local narrative support in `notes/EXPERIMENTS.md:93-98`, but the exact README `exp53/exp54` external-package tables still need committed manifests if morning review wants auditable numbers without reruns.

### Support/stability docs claims

- The support/stability plan has good non-goal language. `docs/superpowers/plans/2026-05-05-support-stability.md:17-24` explicitly says V1 is not pixels, learned perception, learned rules, trajectories, collision simulation, or partial physics priors.
- The plan's falsification gate at `docs/superpowers/plans/2026-05-05-support-stability.md:111-119` requires 100% TL deterministic accuracy, 100% counterfactual retraction, and at least 10 percentage-point OOD margin over best neural baseline. `tests/test_exp87_support_eval.py:38-41` verifies those gate keys exist and pass in a quick test run, while `experiments/exp87_support_data/results.json` records full-run passed gates.
- The plan's vertical slices at `docs/superpowers/plans/2026-05-05-support-stability.md:200-267` line up with local artifacts: `exp84`, `exp85`, `exp86`, `exp87`, and their tests exist.
- Risk: the plan is written as a future implementation plan, not as a final memo. It includes Week 1-4 execution text at `docs/superpowers/plans/2026-05-05-support-stability.md:284-308`, but the repo already has later `exp88`-`exp95`. A morning reviewer may misread it as current backlog rather than historical plan unless a current status note is added.

### Worker and handoff docs claims

- `docs/SYMPHONY_RUN_PROTOCOL.md:5-16` defines branch, validation, handoff, and status-management rules for issue work. It also says implementation work is not complete until a real GitHub PR at `docs/SYMPHONY_RUN_PROTOCOL.md:17-26`.
- This queue item explicitly says product code, pushes, and PR creation are out of scope, so this audit does not follow the implementation PR-publishing path. The report should remain a local artifact for the goal-pack runner.
- `docs/RUN_PROTOCOL.md:1-15` documents Kaggle persistence after losing `exp76c` adapter output; `docs/RUN_PROTOCOL.md:65-85` says weights should not be committed, only pointer metadata. This is a useful docs claim and a safety constraint for future remote-run issues.
- `docs/agents/domain.md:5-12` says read `CONTEXT.md`, `CONTEXT-MAP.md`, and `docs/adr/` if present, but this repo currently has none of those tracked paths. This is not a failure; the doc says to proceed silently.

### Benchmarks claims

- `benchmarks/README.md:3-11` says stable benchmark entrypoints go there and gives a suggested order (`closure.py`, `kinship.py`, `parity.py`), but only the README exists. This is a forward-looking placeholder, not an implemented benchmark suite. It should be labeled as such if linked from user-facing docs.

## Risks and stale assumptions

1. Root README recency risk: the README is behind the actual experiment lane. It names `exp1`-`exp54` and five demos, while the repo has `exp95`, support/stability results, web workbench, CI, examples, and agent docs.
2. Auditable-result risk: the headline `exp53` and `exp54` tables are not backed by committed result manifests. The scripts download current external package source tarballs, so reruns can drift as packages release new versions.
3. Conceptual-overclaim risk: `README.md:136-138` compresses a very broad Tensor Logic thesis into one paragraph. Local implementation covers a subset; docs should separate "paper/conceptual background" from "this repo implements/tests".
4. Validation-scope risk: README's full validation command is correct but potentially expensive because tests include many experiment tests. The docs provide fast paths for support/stability, but no similarly crisp cheap proof exists for the root README transitive-closure tables.
5. Historical-plan risk: `docs/superpowers/plans/` contains detailed plans with unchecked task checkboxes and future-tense language. Some are complete or superseded, but not clearly marked as historical.
6. Workbench security/scope risk: `web_workbench/server.py:53-72` writes source to a temp file and invokes `python -m tensor_logic`; this is suitable for local workbench use but should not be documented as safe for remote/multi-user deployment.

## Missing validation

- I did not run `python -m pytest tests/ -v`; the queue validation command is `git status --short`, and the audit is docs-only.
- I did not run `python -m pip install -e ".[dev]"`; dependency installation is outside the requested validation and could mutate the environment.
- I did not run `experiments/exp53_real_imports.py` or `experiments/exp54_big_imports.py`; both require networked package downloads and are outside the local read-only audit scope.
- I did not run demo scripts; README's "seconds to a few minutes" timing claim remains unverified by this audit.
- I did not use external services, GitHub, Linear, Kaggle, or web browsing.

## Validation command candidates

- Required queue validation: `git status --short`
  - Expected after report creation: one local docs report path, likely `?? docs/overnight/` or `?? docs/overnight/tensor-logic-docs-claims.md` depending on git's untracked-directory display.
  - Status: passed for the required queue check; observed output was `?? docs/overnight/`.
- Docs/package contract smoke: `python -m pytest tests/test_packaging_ci.py -v`
  - Expected: pass. It only checks pyproject, CI workflow, README validation commands, and Symphony protocol strings.
- Reusable package and CLI smoke: `python -m pytest tests/test_tensor_logic_core.py -v`
  - Expected: likely pass, but larger than needed for this audit.
- Support/stability docs-backed fast path: `python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v`
  - Expected: likely pass based on local test/code alignment.
- Full documented worker validation: `python -m pytest tests/ -v`
  - Expected: intended pass per CI contract, but not run here due docs-only queue validation.
- Neural baseline smoke: `python experiments/exp86_support_baselines.py --quick`
  - Expected: intended pass per README and protocol, but not run here because it trains models and is not needed to validate a docs report.
- External-package result reproduction: `python experiments/exp53_real_imports.py` and `python experiments/exp54_big_imports.py`
  - Expected: unknown locally because they use `pip download` against external packages and may drift or fail without network.

## Next safe work

1. Update root README recency without changing product code.
   - Acceptance criteria: README no longer says `exp1`-`exp54` or "5 headline runnable demos"; it names the current support/stability lane and links to `notes/EXPERIMENTS.md` rows 87-95.
   - Validation: `python -m pytest tests/test_packaging_ci.py -v` and `git diff -- README.md`.

2. Add committed result manifests for the headline external-package tables.
   - Acceptance criteria: `experiments/exp53_data/results.json` and `experiments/exp54_data/results.json` exist with package versions, source archive refs, git SHA, command, device, TL/MLP parameters, rows, and mean metrics; README tables cite those files.
   - Validation: a new test reads the manifests and asserts README table package names/metrics match, plus `python -m pytest tests/test_packaging_ci.py -v`.
   - Stop condition: if reruns need network approval, create the manifest schema and leave the actual numbers blocked until an approved remote/local run.

3. Mark docs plans as historical/current-state.
   - Acceptance criteria: `docs/superpowers/plans/2026-05-05-support-stability.md` starts with a short status note pointing to implemented `exp84`-`exp95` artifacts, and future-tense tasks are clearly historical plan context.
   - Validation: docs-only diff review and `rg -n "exp95|current status|historical" docs/superpowers/plans/2026-05-05-support-stability.md`.

4. Add a "claim ledger" page for research claims.
   - Acceptance criteria: a new docs page maps each public claim to local evidence: code path, tests, result JSON, command to rerun, caveats, and whether it is conceptual, implemented, validated, stale, or falsified.
   - Validation: `rg -n "exp53|exp54|exp87|exp95|conceptual|validated|stale" docs/`.

5. Add local-only workbench warning.
   - Acceptance criteria: `web_workbench/README.md` says the server shells out to local Python and is for trusted localhost use, not remote deployment.
   - Validation: `python -m pytest tests/test_web_workbench.py -v`.

## Non-goals for this audit

- No product-code edits.
- No experiment reruns that download external packages.
- No dependency installation.
- No generated data, model weights, or cache changes.
- No GitHub PR, push, merge, or external tracker status update.
- No claim about scientific correctness beyond what local files and commands support.

## Unknowns

- Whether the exact README `exp53` and `exp54` numbers still reproduce against current package source distributions.
- Whether all demo scripts still satisfy the "CPU in seconds to a few minutes" timing claim on a clean machine.
- Whether sibling repos currently depend on package APIs not covered by this repo's tests.
- Whether a maintainer wants the README to stay essay-like or become a stricter evidence-index landing page.
- Whether the goal-pack runner expects reports to remain as dirty local files or be committed by workers; this queue item only required the report and `git status --short`.

## Handoff

- Files changed: `docs/overnight/tensor-logic-docs-claims.md`.
- Product code changed: none.
- Commit/HEAD observed before edits: `c9e2cfc`.
- Required validation result: `git status --short` passed and reported only `?? docs/overnight/`.
- PR URL: none, not created by design for this local overnight docs audit.
- Blockers: none for the local report. Reproducing external package tables would require networked reruns and likely package version pinning.
