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
        self.assertEqual(result["payload"]["answer"], True)
        self.assertEqual(result["payload"]["relation"], "ancestor")
        self.assertEqual(result["payload"]["args"], ["alice", "cara"])
        self.assertIn("duration_ms", result)

    def test_prove_action_exposes_structured_payload(self):
        payload = {
            "source": _SAMPLE_SOURCE,
            "relation": "ancestor",
            "arg1": "alice",
            "arg2": "cara",
            "recursive": True,
        }
        result, status = run_tensor_logic_action("prove", payload)
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["payload"]["answer"], True)
        self.assertEqual(result["payload"]["proof"]["head"], ["ancestor", "alice", "cara"])
        self.assertIn("ancestor(alice, cara)", result["stdout"])

    def test_ingest_python_action_loads_dependency_program(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / "pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "api.py").write_text("from . import db\n")
            (pkg / "db.py").write_text("from . import models\n")
            (pkg / "models.py").write_text("")

            result, status = run_tensor_logic_action("ingest-python", {"path": str(root)})

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["action"], "ingest-python")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("relation depends_on(Module, Module)", result["payload"]["source"])
        self.assertIn(["pkg_api", "pkg_db"], result["payload"]["imports"])
        self.assertEqual(result["payload"]["suggested_query"]["relation"], "depends_on")
        self.assertEqual(result["payload"]["suggested_query"]["arg1"], "pkg_api")
        self.assertEqual(result["payload"]["suggested_query"]["arg2"], "pkg_models")
        self.assertEqual(result["payload"]["symbol_to_module"]["pkg_api"], "pkg.api")
        self.assertTrue(result["payload"]["symbol_to_file"]["pkg_api"].endswith("pkg/api.py"))

    def test_repo_impact_action_reports_blast_radius(self):
        source = """
domain Module { api db models web }
relation imports(Module, Module)
relation depends_on(Module, Module)
fact imports(web, api)
fact imports(api, db)
fact imports(db, models)
rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()
"""
        result, status = run_tensor_logic_action(
            "repo-impact",
            {
                "source": source,
                "module": "models",
                "metadata": {
                    "symbol_to_module": {
                        "api": "pkg.api",
                        "db": "pkg.db",
                        "models": "pkg.models",
                        "web": "pkg.web",
                    },
                    "symbol_to_file": {
                        "models": "/tmp/pkg/models.py",
                    },
                },
            },
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["payload"]["module"], "models")
        self.assertEqual(result["payload"]["direct_imports"], [])
        self.assertEqual(result["payload"]["direct_imported_by"], ["db"])
        self.assertEqual(result["payload"]["transitive_dependents"], ["api", "db", "web"])
        self.assertEqual(result["payload"]["dependent_paths"]["web"], ["web", "api", "db", "models"])
        self.assertEqual(result["payload"]["module_details"]["models"]["module"], "pkg.models")
        self.assertEqual(result["payload"]["module_details"]["models"]["file"], "/tmp/pkg/models.py")
        self.assertIn("transitive dependents (3)", result["stdout"])
        self.assertIn("module: pkg.models", result["stdout"])

    def test_repo_overview_action_ranks_hotspots_and_cycles(self):
        source = """
domain Module { api db models web a b }
relation imports(Module, Module)
relation depends_on(Module, Module)
fact imports(web, api)
fact imports(api, db)
fact imports(db, models)
fact imports(a, b)
fact imports(b, a)
rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()
"""
        result, status = run_tensor_logic_action(
            "repo-overview",
            {
                "source": source,
                "metadata": {
                    "symbol_to_module": {"models": "pkg.models", "web": "pkg.web"},
                    "symbol_to_file": {"models": "/tmp/pkg/models.py"},
                },
            },
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["payload"]["modules"], 6)
        self.assertEqual(result["payload"]["imports"], 5)
        self.assertEqual(result["payload"]["top_dependents"][0], {"module": "models", "count": 3})
        self.assertIn({"module": "web", "count": 3}, result["payload"]["top_dependencies"])
        self.assertIn("web", result["payload"]["entrypoints"])
        self.assertIn("models", result["payload"]["leaves"])
        self.assertIn(["a", "b"], result["payload"]["cycles"])
        self.assertEqual(result["payload"]["module_details"]["models"]["module"], "pkg.models")
        self.assertIn("cycles=1", result["stdout"])

    def test_repo_brief_action_outputs_actionable_change_plan(self):
        source = """
domain Module { api db models web }
relation imports(Module, Module)
relation depends_on(Module, Module)
fact imports(web, api)
fact imports(api, db)
fact imports(db, models)
rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()
"""
        result, status = run_tensor_logic_action(
            "repo-brief",
            {
                "source": source,
                "module": "models",
                "metadata": {
                    "symbol_to_module": {
                        "api": "pkg.api",
                        "db": "pkg.db",
                        "models": "pkg.models",
                        "web": "pkg.web",
                    },
                    "symbol_to_file": {
                        "api": "/tmp/pkg/api.py",
                        "db": "/tmp/pkg/db.py",
                        "models": "/tmp/pkg/models.py",
                        "web": "/tmp/pkg/web.py",
                    },
                },
            },
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["action"], "repo-brief")
        self.assertEqual(result["payload"]["risk_level"], "medium")
        self.assertEqual(result["payload"]["blast_radius"], 3)
        self.assertEqual(result["payload"]["read_first"][0]["file"], "/tmp/pkg/models.py")
        self.assertIn({"relation": "depends_on", "arg1": "web", "arg2": "models", "recursive": True}, result["payload"]["proof_checks"])
        self.assertIn("Change brief: models (pkg.models)", result["stdout"])
        self.assertIn("prove depends_on(web, models) recursive", result["stdout"])

    def test_repo_compare_action_reports_dependency_drift(self):
        before_source = """
domain Module { api db models web }
relation imports(Module, Module)
relation depends_on(Module, Module)
fact imports(web, api)
fact imports(api, db)
rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()
"""
        after_source = """
domain Module { api db models web worker }
relation imports(Module, Module)
relation depends_on(Module, Module)
fact imports(web, api)
fact imports(api, models)
fact imports(worker, api)
rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()
"""
        result, status = run_tensor_logic_action(
            "repo-compare",
            {
                "before_source": before_source,
                "after_source": after_source,
                "after_metadata": {
                    "symbol_to_module": {
                        "api": "pkg.api",
                        "db": "pkg.db",
                        "models": "pkg.models",
                        "web": "pkg.web",
                        "worker": "pkg.worker",
                    },
                    "symbol_to_file": {"worker": "/tmp/pkg/worker.py"},
                },
            },
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(result["action"], "repo-compare")
        self.assertEqual(result["payload"]["added_modules"], ["worker"])
        self.assertEqual(result["payload"]["removed_modules"], [])
        self.assertIn(["api", "models"], result["payload"]["added_imports"])
        self.assertIn(["worker", "api"], result["payload"]["added_imports"])
        self.assertIn(["api", "db"], result["payload"]["removed_imports"])
        self.assertIn({"module": "api", "before": 1, "after": 2, "delta": 1}, result["payload"]["blast_radius_deltas"])
        self.assertIn(
            {
                "kind": "added dependency",
                "action": "prove",
                "relation": "depends_on",
                "arg1": "worker",
                "arg2": "api",
                "recursive": True,
                "expected": True,
            },
            result["payload"]["suggested_checks"],
        )
        self.assertEqual(result["payload"]["module_details"]["worker"]["file"], "/tmp/pkg/worker.py")
        self.assertIn("Repo graph compare", result["stdout"])
        self.assertIn("Added imports", result["stdout"])

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
        self.assertNotIn("unknown(?, ?)", wb_res["stdout"])
        self.assertEqual(wb_res["payload"]["answer"], False)
        self.assertEqual(wb_res["payload"]["explanation"]["head"], ["ancestor", "alice", "dave"])
