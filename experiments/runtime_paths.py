"""Runtime output paths for experiment scripts.

Committed ``experiments/*_data`` files are evidence fixtures. Default script
runs should write to ignored local state unless the caller passes an explicit
output path to refresh a fixture intentionally.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_EXPERIMENTS_DIR = REPO_ROOT / ".runtime" / "experiments"


def default_runtime_result_path(result_dir: Path, *, quick: bool) -> Path:
    filename = "results_quick.json" if quick else "results.json"
    return RUNTIME_EXPERIMENTS_DIR / result_dir.name / filename


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)
