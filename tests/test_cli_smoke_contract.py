from __future__ import annotations

import subprocess
import sys


def test_tensor_logic_cli_help_smoke_runs_without_secrets():
    result = subprocess.run(
        [sys.executable, "-m", "tensor_logic", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: tensor-logic" in result.stdout
    assert "{run,query,prove,ingest-python,repl,repo-graph,serve}" in result.stdout


def test_tensor_logic_cli_query_smoke_parses_and_executes_temp_tl(tmp_path):
    tl_file = tmp_path / "smoke.tl"
    tl_file.write_text(
        "\n".join(
            [
                "domain Node { a, b }",
                "relation edge(Node, Node)",
                "fact edge(a, b)",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "tensor_logic", "query", str(tl_file), "edge", "a", "b"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "edge(a, b) = True"


def test_tensor_logic_cli_bad_input_returns_clear_nonzero_failure(tmp_path):
    missing_file = tmp_path / "missing.tl"

    result = subprocess.run(
        [sys.executable, "-m", "tensor_logic", "run", str(missing_file)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Error:" in result.stderr
    assert str(missing_file) in result.stderr
    assert "Traceback" not in result.stderr
