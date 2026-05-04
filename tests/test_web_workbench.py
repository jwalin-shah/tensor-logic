import unittest
import json
import subprocess
import sys
from http import HTTPStatus
import tempfile
from pathlib import Path
from web_workbench.server import run_tensor_logic_action

_SAMPLE_SOURCE = """
domain Person { alice bob cara dave }
relation parent(Person, Person)
relation ancestor(Person, Person)
fact parent(alice, bob)
fact parent(bob, cara)
rule ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()
"""

class TestWebWorkbench(unittest.TestCase):
    def test_query_api_smoke(self):
        payload = {
            "source": _SAMPLE_SOURCE,
            "relation": "ancestor",
            "arg1": "alice",
            "arg2": "cara",
            "recursive": True,
        }
        result, status = run_tensor_logic_action("query", payload)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["action"], "query")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("ancestor(alice, cara) = True", result["stdout"])

    def test_why_not_json_parity_with_http_api(self):
        from tensor_logic.http_api import prove_source

        # HTTP API output
        http_api_res = prove_source(
            _SAMPLE_SOURCE,
            "ancestor",
            ["alice", "dave"],
            recursive=True,
            why_not=True,
            format_type="json",
        )
        self.assertFalse(http_api_res["answer"])
        self.assertIn("explanation", http_api_res)

        # We also want to verify CLI produces matching JSON parity when asked to.
        # Run CLI directly
        with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as handle:
            handle.write(_SAMPLE_SOURCE)
            temp_path = Path(handle.name)
        try:
            cmd = [
                sys.executable, "-m", "tensor_logic", "prove",
                str(temp_path), "ancestor", "alice", "dave",
                "--recursive", "--why-not", "--format", "json"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            cli_json = json.loads(proc.stdout)

            # Smoke test for parity: CLI vs HTTP
            self.assertEqual(cli_json["answer"], http_api_res["answer"])
            self.assertEqual(cli_json["explanation"]["head"], http_api_res["explanation"]["head"])
            self.assertEqual(cli_json["explanation"]["reason"], http_api_res["explanation"]["reason"])
        finally:
            temp_path.unlink(missing_ok=True)

        # Let's test the workbench `why-not` output. Currently it uses tree text (no json format).
        payload = {
            "source": _SAMPLE_SOURCE,
            "relation": "ancestor",
            "arg1": "alice",
            "arg2": "dave",
            "recursive": True,
        }
        wb_res, wb_status = run_tensor_logic_action("why-not", payload)
        self.assertEqual(wb_status, HTTPStatus.OK)
        self.assertIn("ancestor(alice, dave) = False", wb_res["stdout"])
