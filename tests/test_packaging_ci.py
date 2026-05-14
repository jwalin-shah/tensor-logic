from pathlib import Path
import importlib.util
import re
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DOC = REPO_ROOT / "docs" / "VALIDATION.md"
CONTEXT_DOC = REPO_ROOT / "CONTEXT.md"
PROVENANCE_DOC = REPO_ROOT / "docs" / "EXPERIMENT_PROVENANCE.md"
STATUS_DOC = REPO_ROOT / "docs" / "EXPERIMENT_STATUS.md"


def _local_validation_module():
    spec = importlib.util.spec_from_file_location(
        "local_validation", REPO_ROOT / "tools" / "local_validation.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert "fetch-depth: 0" in workflow
    assert 'python -m pip install -e ".[dev]"' in workflow
    assert "python tools/local_validation.py" in workflow


def test_readme_documents_worker_validation_commands():
    readme = (REPO_ROOT / "README.md").read_text()

    assert 'python3 -m pip install -e ".[dev]"' in readme
    assert "python3 tools/local_validation.py" in readme
    assert "python tools/local_validation.py" in readme
    assert "tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v" in readme
    assert "python experiments/exp86_support_baselines.py --quick" in readme
    assert 'python -m pip install -e ".[dev]"' in readme
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
        "## Canonical Local And CI Validation",
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

    assert 'python3 -m pip install -e ".[dev]"' in validation
    assert "python3 tools/local_validation.py" in validation
    assert 'python -m pip install -e ".[dev]"' in validation
    assert "python tools/local_validation.py" in validation
    assert "python3 -m pytest tests/test_packaging_ci.py tests/test_code_index.py -v" in validation
    assert "remote services, external datasets, model" in validation
    assert "External FAFSA/ISIR validation" in validation
    assert "docs/EXPERIMENT_PROVENANCE.md" in validation


def test_local_validation_gate_runs_executable_checks():
    gate = (REPO_ROOT / "tools" / "local_validation.py").read_text()

    assert '"-m", "pytest", "-q"' in gate
    assert "_committed_diff_check_command()" in gate
    assert '"git", "diff", "--check", f"{merge_base}..HEAD"' in gate


def test_local_validation_gate_checks_committed_diff_from_base_ref(monkeypatch):
    local_validation = _local_validation_module()
    monkeypatch.setenv("LOCAL_VALIDATION_BASE_REF", "origin/release")
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_git_output(args):
        assert args == ["merge-base", "origin/release", "HEAD"]
        return "abc123"

    monkeypatch.setattr(local_validation, "_git_output", fake_git_output)

    assert local_validation._committed_diff_check_command() == [
        "git",
        "diff",
        "--check",
        "abc123..HEAD",
    ]


def test_local_validation_gate_uses_github_base_ref_for_pr_ci(monkeypatch):
    local_validation = _local_validation_module()
    monkeypatch.delenv("LOCAL_VALIDATION_BASE_REF", raising=False)
    monkeypatch.setenv("GITHUB_BASE_REF", "main")

    def fake_git_output(args):
        assert args == ["merge-base", "origin/main", "HEAD"]
        return "def456"

    monkeypatch.setattr(local_validation, "_git_output", fake_git_output)

    assert local_validation._committed_diff_check_command() == [
        "git",
        "diff",
        "--check",
        "def456..HEAD",
    ]


def test_local_validation_gate_falls_back_to_head_parent(monkeypatch):
    local_validation = _local_validation_module()
    monkeypatch.delenv("LOCAL_VALIDATION_BASE_REF", raising=False)
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_git_output(args):
        if args == ["rev-parse", "--verify", "--quiet", "origin/main"]:
            return None
        if args == ["rev-parse", "--verify", "--quiet", "HEAD^"]:
            return "parent123"
        raise AssertionError(args)

    monkeypatch.setattr(local_validation, "_git_output", fake_git_output)

    assert local_validation._committed_diff_check_command() == [
        "git",
        "diff",
        "--check",
        "HEAD^..HEAD",
    ]


def test_local_validation_gate_exits_nonzero_when_pytest_fails(tmp_path):
    failing_test = tmp_path / "test_failing_gate.py"
    failing_test.write_text("def test_fails():\n    assert False\n")

    result = subprocess.run(
        [sys.executable, "tools/local_validation.py", str(failing_test)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode != 0
    assert "test_fails" in result.stdout


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
