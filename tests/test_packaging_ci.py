from pathlib import Path
import json
import re
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DOC = REPO_ROOT / "docs" / "VALIDATION.md"
VALIDATION_REGISTRY = REPO_ROOT / "docs" / "validation-registry.json"


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
    assert (
        pyproject["tool"]["tensor-logic"]["validation"]["registry"]
        == "docs/validation-registry.json"
    )


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
        "## Expected Artifacts",
    ]
    for section in expected_sections:
        assert section in validation

    assert 'python -m pip install -e ".[dev]"' in validation
    assert "python -m pytest tests/ -v" in validation
    assert 'python3 -m pip install -e ".[dev]"' in validation
    assert "python3 -m pytest tests/ -v" in validation
    registry = json.loads(VALIDATION_REGISTRY.read_text())
    cheap_commands = (
        registry["tiers"]["packaging-ci"]["commands"]
        + registry["tiers"]["packaging-ci"]["fallback_commands"]
    )
    for command in cheap_commands:
        assert command in validation
    assert "remote services, external datasets, model" in validation
    assert "External FAFSA/ISIR validation" in validation
    assert "[`docs/validation-registry.json`](validation-registry.json)" in validation


def test_validation_doc_maps_heavy_dependencies_outside_ci():
    validation = VALIDATION_DOC.read_text()

    for dependency in ["transformers", "peft", "datasets", "accelerate", "scipy"]:
        assert dependency in validation

    for extra in ["`lm` extra", "`sft` extra", "`science` extra"]:
        assert extra in validation

    assert "Do not add `transformers`, `peft`, `datasets`, `accelerate`, `scipy`" in validation


def test_validation_registry_maps_change_classes_to_executable_tiers():
    registry = json.loads(VALIDATION_REGISTRY.read_text())

    expected_change_classes = {
        "docs-only",
        "validation-policy",
        "packaging",
        "code-index",
        "tensor-logic-core",
        "demo",
        "lm-or-training-experiment",
    }
    assert expected_change_classes <= registry["change_classes"].keys()

    for change_class, metadata in registry["change_classes"].items():
        assert metadata["tiers"], change_class
        for tier_name in metadata["tiers"]:
            tier = registry["tiers"][tier_name]
            assert "cost" in tier
            assert "ci" in tier
            assert "heavyweight" in tier
            if tier["ci"]:
                assert tier["commands"], tier_name
            if tier["cost"] == "cheap":
                assert tier.get("fallback_commands"), tier_name
                for command in tier["commands"]:
                    assert command.replace("python ", "python3 ", 1) in tier["fallback_commands"]


def test_validation_registry_encodes_non_ci_boundaries():
    registry = json.loads(VALIDATION_REGISTRY.read_text())

    heavy_tier = registry["tiers"]["heavy-experiment"]
    assert heavy_tier["ci"] is False
    assert heavy_tier["heavyweight"] is True
    assert heavy_tier["requires_explicit_work_order"] is True
    assert "command" in heavy_tier["record_required"]
    assert any("FAFSA/ISIR" in boundary for boundary in heavy_tier["boundaries"])

    dependency_boundaries = registry["dependency_boundaries"]
    assert dependency_boundaries["ci_allowed_extras"] == ["dev"]
    for extra in ["lm", "sft", "science"]:
        assert extra in dependency_boundaries["non_ci_extras"]


def test_validation_registry_matches_documented_cheap_commands():
    registry = json.loads(VALIDATION_REGISTRY.read_text())
    validation = VALIDATION_DOC.read_text()

    cheap_tiers = [
        tier
        for tier in registry["tiers"].values()
        if tier["cost"] == "cheap" and tier["commands"]
    ]
    assert cheap_tiers
    for tier in cheap_tiers:
        for command in tier["commands"]:
            assert command in validation


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
