# Validation Matrix

Use this matrix to choose the cheapest proof that matches the change. The default
validation path must stay local, CPU-friendly, and free of package-download,
model-download, GPU, remote-job, or external FAFSA/ISIR data requirements.

Before changing experiment claims, result tables, README benchmark language, or
status summaries, read `CONTEXT.md` and `docs/EXPERIMENT_PROVENANCE.md`. The
validation command proves that files still parse and contract checks still pass;
it does not upgrade the evidence tier behind a claim.

## Canonical CI Validation

GitHub Actions validates pull requests and pushes to `main` with the editable dev
install and the full default test tree:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

If a local machine does not provide `python`, use the equivalent `python3`
fallback:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

For documentation, packaging, or validation-boundary changes, this narrower proof
is the canonical cheap preflight before the full test tree:

```bash
python3 -m pytest tests/test_packaging_ci.py tests/test_code_index.py -v
python3 -m pytest tests/ -v
```

## Cheap CI Tests

These commands are expected to run in CI and on a clean laptop checkout after
`.[dev]` installation:

```bash
python -m pytest tests/ -v
python -m pytest tests/test_packaging_ci.py tests/test_code_index.py -v
```

The default `tests/` path is for deterministic unit and smoke coverage only. Do
not add tests there if they require remote services, external datasets, model
downloads, package-download experiments, GPU-only execution, or long training
runs.

## Code-Index Commands

The structured API index is local tooling for agent planning. It writes
`tools/index.json`, which is intentionally gitignored.

```bash
python tools/code_index.py --status
python tools/code_index.py --lookup Program
python tools/code_index.py --rebuild
```

Use `--lookup <RelevantSymbol>` before planning changes that touch
`tensor_logic/`. Rebuild the index only when it is stale or a validation command
needs a fresh index.

## Lightweight Build/Import Proof

Use this proof when the package metadata, import surface, or editable install
contract changes. It does not build wheels, run demos, download models, or touch
remote services:

```bash
python -m pip install -e ".[dev]"
python -c "import tensor_logic; from tensor_logic import Program; print('tensor_logic import ok')"
```

Fallback when `python` is absent:

```bash
python3 -m pip install -e ".[dev]"
python3 -c "import tensor_logic; from tensor_logic import Program; print('tensor_logic import ok')"
```

## Demo Smoke Commands

Demos are useful after changes to examples, tensor-language ergonomics, or
research-facing workflows. They are not part of CI because they can be slower
than unit tests and may require explicit runtime package selection:

```bash
uv run --with torch python demos/transitive_closure.py
uv run --with torch python demos/tensor_language.py
uv run --with torch python demos/program_rules.py
uv run --with torch python demos/provenance_kb.py
```

Longer training demos such as `demos/train_kg.py`, `demos/joint_lm_kg.py`,
`demos/throwing.py`, and `demos/catastrophic_forgetting.py` should be run only
when the touched code justifies that extra runtime.

## Optional Dependency Boundaries

Default dependencies are deliberately small. CI installs only the package default
dependencies plus `.[dev]`.

| Dependency set | Packages | Intended use | CI status |
| --- | --- | --- | --- |
| default | `torch` | reusable `tensor_logic` package and deterministic tests | installed in CI |
| `dev` extra | `pytest`, `numpy`, `matplotlib` | unit tests, local plotting helpers, lightweight experiment smoke tests | installed in CI |
| `lm` extra | `transformers` | local LM-backed experiments such as exp32/36/77/78 | non-CI; may download models |
| `sft` extra | `transformers`, `peft`, `datasets`, `accelerate` | exp60-style LoRA/SFT training and eval | non-CI; download/GPU/remote candidate |
| `science` extra | `scipy` | exp83 slot-attention matching and similar scientific helpers | non-CI unless a future issue promotes a cheap test |

Other local-only dependencies are intentionally documented instead of installed
by CI:

- `mlx-lm` is for Apple Silicon MLX model experiments in exp78/exp81-style runs.
- Packages used as import-graph subjects, such as `networkx`, `sqlalchemy`,
  `django`, `fastapi`, `sympy`, or `scikit-learn`, belong to explicit
  import-graph experiment environments, not the repo test environment.

Do not add `transformers`, `peft`, `datasets`, `accelerate`, `scipy`, `mlx-lm`,
or import-graph subject packages to default dependencies without a product
decision and a new local proof command.

## Heavyweight And Remote Experiments

The following checks are outside the default `tests/` path:

- LM/VLM inference or SFT runs that load Hugging Face, MLX, Kaggle, Colab, Modal,
  or other remote/provider artifacts.
- GPU/MPS training runs, long sweeps, and batch experiment runners.
- External package-download experiments for real import-graph benchmarks.
- External FAFSA/ISIR validation against official or private datasets.

Run these only from an explicit work order. Record the command, dependency set,
runtime location, input dataset/model references, output path, and blockers.
For durable outputs, follow `docs/RUN_PROTOCOL.md` and record enough provenance
to satisfy `docs/EXPERIMENT_PROVENANCE.md`.

## Claim And Provenance Checks

Docs-only changes that touch provenance, experiment claims, validation wording,
or no-overclaim rules should use the narrow contract check first:

```bash
python3 -m pytest tests/test_packaging_ci.py -v
```

When those changes also touch code-index guidance, add:

```bash
python3 -m pytest tests/test_code_index.py -v
```

This is sufficient for docs-only contract edits. It is not sufficient when a
metric, result table, or experiment verdict changes; those edits require the
exact experiment command or an explicit note that the artifact is historical and
was not refreshed.

## Expected Artifacts

Cheap CI proofs should leave no durable artifact except the gitignored
`tools/index.json` when the code-index command rebuilds it.

Experiment runs may intentionally write artifacts under experiment-specific
directories, for example:

- `experiments/exp78_data/results.json`
- `experiments/exp79_data/results.json`
- `experiments/exp79_data/complexity_curve.png`
- `experiments/exp83_slot_data/results.json`
- `experiments/exp83_slot_data/complexity_curve.png`

Do not create, refresh, or commit experiment artifacts as part of default
validation unless the issue explicitly owns those outputs.
