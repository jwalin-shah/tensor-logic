from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent


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
