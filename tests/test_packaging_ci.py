from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DOC = REPO_ROOT / "docs" / "VALIDATION.md"


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
    assert "web_workbench*" in package_include
    assert project["scripts"]["tensor-logic-workbench"] == "web_workbench.server:main"
    assert "static/*" in pyproject["tool"]["setuptools"]["package-data"]["web_workbench"]


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


def test_workbench_readme_scopes_console_script_to_local_install():
    readme = (REPO_ROOT / "web_workbench" / "README.md").read_text()

    assert "After installing the package locally" in readme
    assert "python -m pip install -e ." in readme
    assert "tensor-logic-workbench --host 127.0.0.1 --port 8080" in readme
    assert "source checkout without the console script installed" in readme
    assert "python web_workbench/server.py --host 127.0.0.1 --port 8080" in readme


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
    assert "python3 -m pytest tests/test_packaging_ci.py tests/test_code_index.py -v" in validation
    assert "remote services, external datasets, model" in validation
    assert "External FAFSA/ISIR validation" in validation


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


def test_built_wheel_contains_web_workbench_package_and_assets():
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                "--wheel-dir",
                td,
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        assert result.returncode == 0, result.stderr
        wheel = next(Path(td).glob("tensor_logic-*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())

    assert "web_workbench/__init__.py" in names
    assert "web_workbench/server.py" in names
    assert "web_workbench/static/index.html" in names
