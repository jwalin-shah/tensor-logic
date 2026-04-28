import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "tensor_logic" / "proof_tree_viewer.py"
_spec = importlib.util.spec_from_file_location("proof_tree_viewer", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
build_proof_tree_view = _module.build_proof_tree_view
render_proof_tree = _module.render_proof_tree

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


class ProofTreeViewerTest(unittest.TestCase):
    def test_positive_tree_renders_confidence_and_source(self):
        payload = json.loads((_FIXTURE_DIR / "proof_tree_positive.json").read_text())
        tree = build_proof_tree_view(payload)

        rendered = render_proof_tree(tree)

        self.assertIn("depends_on(worker, models)", rendered)
        self.assertIn("(0.81)", rendered)
        self.assertIn("[examples/code_dependencies.tl:14]", rendered)
        self.assertIn("imports(api, models)", rendered)

    def test_negative_tree_renders_reason(self):
        payload = json.loads((_FIXTURE_DIR / "proof_tree_negative.json").read_text())
        tree = build_proof_tree_view(payload)

        rendered = render_proof_tree(tree)

        self.assertIn("depends_on(models, worker) = False", rendered)
        self.assertIn("[all_rules_failed]", rendered)
        self.assertIn("imports(models, worker) = False", rendered)

    def test_collapsed_nodes_hide_subtrees(self):
        payload = json.loads((_FIXTURE_DIR / "proof_tree_positive.json").read_text())
        tree = build_proof_tree_view(payload)

        rendered = render_proof_tree(tree, collapsed={"0.1"})

        self.assertIn("▸ depends_on(api, models)", rendered)
        self.assertNotIn("imports(api, models)", rendered)

    def test_supports_nested_negative_child_explanation_shape(self):
        payload = {
            "answer": False,
            "explanation": {
                "head": ["edge", "a", "z"],
                "reason": "all_rules_failed",
                "body": [
                    {
                        "answer": False,
                        "explanation": {
                            "head": ["edge", "a", "z"],
                            "reason": "no_rules",
                            "body": []
                        },
                    }
                ],
            },
        }

        tree = build_proof_tree_view(payload)
        rendered = render_proof_tree(tree)

        self.assertIn("edge(a, z) = False", rendered)
        self.assertIn("[no_rules]", rendered)


if __name__ == "__main__":
    unittest.main()
