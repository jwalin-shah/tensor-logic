from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, command: Sequence[str]) -> int:
    print(f"==> {label}")
    print("$ " + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def main() -> int:
    checks = [
        ("Full pytest suite", [sys.executable, "-m", "pytest", "tests/", "-v"]),
        ("PR whitespace diff check", ["git", "diff", "--check", "origin/main...HEAD"]),
        ("Staged whitespace diff check", ["git", "diff", "--cached", "--check"]),
        ("Working tree whitespace diff check", ["git", "diff", "--check"]),
    ]

    for label, command in checks:
        returncode = _run(label, command)
        if returncode != 0:
            print(f"{label} failed with exit code {returncode}")
            return returncode

    print("Local validation gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
