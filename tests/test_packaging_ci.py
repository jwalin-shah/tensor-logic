from pathlib import Path
import re
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DOC = REPO_ROOT / "docs" / "VALIDATION.md"
CONTEXT_DOC = REPO_ROOT / "CONTEXT.md"
PROVENANCE_DOC = REPO_ROOT / "docs" / "EXPERIMENT_PROVENANCE.md"
STATUS_DOC = REPO_ROOT / "docs" / "EXPERIMENT_STATUS.md"


def _requirement_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0].replace("_", "-").lower()
        for requirement in requirements
    }


def test_pyproject_declares_worker_dev_install_contract():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    dependencies = _requirement_names(project["dependencies"])
    dev_dependencies = _requirement_names(project["optional-dependencies"]["dev"])
    package_include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "torch" in dependencies
    assert {"pytest", "numpy", "matplotlib"} <= dev_dependencies
    assert "tensor_logic*" in package_include


def test_heavy_ml_dependencies_are_optional_not_default_ci():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    default_dependencies = _requirement_names(project["dependencies"])
    dev_dependencies = _requirement_names(project["optional-dependencies"]["dev"])
    optional_dependency_names = set()
    for requirements in project["optional-dependencies"].values():
        optional_dependency_names.update(_requirement_names(requirements))

    heavy_dependencies = {"transformers", "peft", "datasets", "accelerate", "scipy"}
    assert heavy_dependencies.isdisjoint(default_dependencies)
    assert heavy_dependencies.isdisjoint(dev_dependencies)
    assert heavy_dependencies <= optional_dependency_names


def test_github_actions_runs_worker_validation_commands():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "main" in workflow
    assert 'python -m pip install -e ".[dev]"' in workflow
    assert "python -m pytest tests/ -v" in workflow


def test_readme_documents_worker_validation_commands():
    readme = (REPO_ROOT / "README.md").read_text()

    assert 'python -m pip install -e ".[dev]"' in readme
    assert "python -m pytest tests/ -v" in readme
    assert "tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v" in readme
    assert "python experiments/exp86_support_baselines.py --quick" in readme
    assert 'python3 -m pip install -e ".[dev]"' in readme
    assert "python3 -m pytest tests/ -v" in readme
    assert "docs/VALIDATION.md" in readme


def test_symphony_protocol_requires_real_github_prs_and_support_fast_path():
    protocol = (REPO_ROOT / "docs" / "SYMPHONY_RUN_PROTOCOL.md").read_text()

    assert "codex/SYM-25-short-title" in protocol
    assert "real GitHub PR" in protocol
    assert "gh pr view" in protocol
    assert "tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v" in protocol
    assert "python experiments/exp86_support_baselines.py --quick" in protocol


def test_validation_doc_defines_local_tiers_and_boundaries():
    validation = VALIDATION_DOC.read_text()

    expected_sections = [
        "## Canonical CI Validation",
        "## Cheap CI Tests",
        "## Code-Index Commands",
        "## Lightweight Build/Import Proof",
        "## Demo Smoke Commands",
        "## Optional Dependency Boundaries",
        "## Heavyweight And Remote Experiments",
        "## Claim And Provenance Checks",
        "## Expected Artifacts",
    ]
    for section in expected_sections:
        assert section in validation

    assert 'python -m pip install -e ".[dev]"' in validation
    assert "python -m pytest tests/ -v" in validation
    assert 'python3 -m pip install -e ".[dev]"' in validation
    assert "python3 -m pytest tests/ -v" in validation
    assert "python3 -m pytest tests/test_packaging_ci.py tests/test_code_index.py -v" in validation
    assert "remote services, external datasets, model" in validation
    assert "External FAFSA/ISIR validation" in validation
    assert "docs/EXPERIMENT_PROVENANCE.md" in validation


def test_validation_doc_maps_heavy_dependencies_outside_ci():
    validation = VALIDATION_DOC.read_text()

    for dependency in ["transformers", "peft", "datasets", "accelerate", "scipy"]:
        assert dependency in validation

    for extra in ["`lm` extra", "`sft` extra", "`science` extra"]:
        assert extra in validation

    assert "Do not add `transformers`, `peft`, `datasets`, `accelerate`, `scipy`" in validation


def test_lightweight_import_proof_runs_without_remote_or_gpu():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tensor_logic; from tensor_logic import Program; print('tensor_logic import ok')",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, result.stderr
    assert "tensor_logic import ok" in result.stdout


def test_context_doc_defines_no_overclaim_rules_and_agent_boundary():
    context = CONTEXT_DOC.read_text()

    assert "No-overclaim rule" in context
    assert "docs/EXPERIMENT_PROVENANCE.md" in context
    assert "exp87" in context
    assert "exp95" in context
    assert "exp53" in context
    assert "exp54" in context
    assert "`AGENTS.md` is memjuice-managed" in context


def test_context_doc_preserves_agent_domain_map():
    context = CONTEXT_DOC.read_text()

    for section in [
        "## Repo Identity",
        "## Domain Vocabulary",
        "## Package Map",
        "## Experiment Map",
        "## Maintainer Notes For Agents",
    ]:
        assert section in context

    for required_term in [
        "**Tensor Logic (TL)**",
        "**Program**",
        "**Domain**",
        "**Relation**",
        "**Fact**",
        "**Rule**",
        "**Atom**",
        "**Proof**",
        "**TL file**",
        "**Repo graph**",
    ]:
        assert required_term in context

    for required_path in [
        "`tensor_logic/language.py`",
        "`tensor_logic/program.py`",
        "`tensor_logic/file_format.py`",
        "`tensor_logic/execution.py`",
        "`tensor_logic/proofs.py`",
        "`tensor_logic/research/`",
        "`web_workbench/`",
    ]:
        assert required_path in context


def test_experiment_provenance_doc_maps_current_artifacts_and_claim_boundaries():
    provenance = PROVENANCE_DOC.read_text()

    expected_sections = [
        "## Evidence Tiers",
        "## Required Provenance For New Results",
        "## Current Result Artifact Index",
        "## No-Overclaim Rules",
        "## Validation For Claim Edits",
    ]
    for section in expected_sections:
        assert section in provenance

    for exp_id in range(87, 96):
        assert f"`exp{exp_id}`" in provenance
        assert f"experiments/exp{exp_id}" in provenance

    assert "experiments/exp78_data/results.json" in provenance
    assert "experiments/exp79_data/results.json" in provenance
    assert "experiments/exp83_slot_data/results.json" in provenance
    assert "No checked-in `exp53_data/results.json`" in provenance
    assert "python3 -m pytest tests/test_packaging_ci.py -v" in provenance


def test_experiment_status_snapshot_records_active_contract_boundaries():
    status = STATUS_DOC.read_text()

    expected_sections = [
        "# Experiment Status Snapshot",
        "## Current Contract Status",
        "## Claim Status Summary",
        "## Required Next Review Triggers",
    ]
    for section in expected_sections:
        assert section in status

    for linked_doc in [
        "docs/EXPERIMENT_PROVENANCE.md",
        "docs/VALIDATION.md",
        "notes/EXPERIMENTS.md",
    ]:
        assert linked_doc in status

    for required_phrase in [
        "Provenance contract",
        "No-overclaim rule",
        "Validation matrix",
        "Result artifact index",
        "do not claim fully reproducible from checked-in artifacts alone",
        "must retain missing-object and merge/cardinality caveats",
    ]:
        assert required_phrase in status
