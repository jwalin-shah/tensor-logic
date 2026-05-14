#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def _git_output(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _committed_diff_check_command() -> list[str]:
    base_ref = os.environ.get("LOCAL_VALIDATION_BASE_REF")
    if base_ref is None:
        github_base_ref = os.environ.get("GITHUB_BASE_REF")
        if github_base_ref:
            base_ref = f"origin/{github_base_ref}"
        elif _git_output(["rev-parse", "--verify", "--quiet", "origin/main"]):
            base_ref = "origin/main"

    if base_ref:
        merge_base = _git_output(["merge-base", base_ref, "HEAD"])
        if merge_base:
            return ["git", "diff", "--check", f"{merge_base}..HEAD"]

    if _git_output(["rev-parse", "--verify", "--quiet", "HEAD^"]):
        return ["git", "diff", "--check", "HEAD^..HEAD"]

    return ["git", "diff", "--check", "--cached"]


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

    committed_diff_rc = _run(_committed_diff_check_command())
    if committed_diff_rc != 0:
        return committed_diff_rc

    return _run(["git", "diff", "--check"])


if __name__ == "__main__":
    raise SystemExit(main())
