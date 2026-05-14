from __future__ import annotations

import subprocess
import sys


def test_tensor_logic_cli_run_smoke(tmp_path):
    source = tmp_path / "smoke.tl"
    source.write_text(
        "\n".join(
            [
                "domain Node { a b }",
                "relation edge(Node, Node)",
                "fact edge(a, b)",
                "query edge(a, b)",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "tensor_logic", "run", str(source)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "edge(a, b) = True" in result.stdout


def test_tensor_logic_cli_bad_input_reports_clear_failure(tmp_path):
    source = tmp_path / "bad.tl"
    source.write_text("not valid tl\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "tensor_logic", "run", str(source)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "tensor-logic: error:" in result.stderr
    assert "unrecognized statement" in result.stderr
    assert "Traceback" not in result.stderr
