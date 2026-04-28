from pathlib import Path
import ast
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

def test_index_json_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "tools/index.json" in gitignore


from code_index import _extract_module, build_index

def test_extract_class_init_args(tmp_path):
    src = tmp_path / "mymod.py"
    src.write_text("""
class Foo:
    def __init__(self, name: str, count: int):
        self.name = name
    def bar(self): pass
    def _private(self): pass
""")
    result = _extract_module(src, module_prefix="tensor_logic")
    assert "tensor_logic.mymod" in result
    sym = result["tensor_logic.mymod"]["Foo"]
    assert sym["kind"] == "class"
    assert sym["init_args"] == ["name: str", "count: int"]
    assert sym["methods"] == ["bar"]  # _private excluded

def test_extract_function_args_and_return(tmp_path):
    src = tmp_path / "mymod.py"
    src.write_text("""
def load(path: str) -> int:
    return 0

def _internal(x): pass
""")
    result = _extract_module(src, module_prefix="tensor_logic")
    syms = result["tensor_logic.mymod"]
    assert "load" in syms
    assert syms["load"]["kind"] == "function"
    assert syms["load"]["args"] == ["path: str"]
    assert syms["load"]["returns"] == "int"
    assert "_internal" not in syms

def test_build_index_covers_tensor_logic():
    index = build_index()
    assert "_meta" in index
    assert "built_at" in index["_meta"]
    assert "tensor_logic.program" in index
    assert "Program" in index["tensor_logic.program"]
    assert index["tensor_logic.program"]["Program"]["kind"] == "class"

def test_build_index_excludes_private_modules():
    index = build_index()
    assert not any(k.endswith(".__init__") or k.endswith(".__main__") for k in index)
