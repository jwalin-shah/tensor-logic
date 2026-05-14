#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local pre-handoff validation gate.")
    parser.add_argument(
        "pytest_targets",
        nargs="*",
        default=["tests/"],
        help="Optional pytest targets; defaults to the full tests/ tree.",
    )
    args = parser.parse_args(argv)

    pytest_rc = _run([sys.executable, "-m", "pytest", "-q", *args.pytest_targets])
    if pytest_rc != 0:
        return pytest_rc

    return _run(["git", "diff", "--check"])


if __name__ == "__main__":
    raise SystemExit(main())
