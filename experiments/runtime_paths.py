"""Local runtime output paths for experiment scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_OUTPUT_ROOT = REPO_ROOT / "runtime_outputs" / "experiments"


def experiment_output_dir(experiment_name: str) -> Path:
    return RUNTIME_OUTPUT_ROOT / experiment_name


def result_path(experiment_name: str, quick: bool = False) -> Path:
    filename = "results_quick.json" if quick else "results.json"
    return experiment_output_dir(experiment_name) / filename


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)
