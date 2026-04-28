from pathlib import Path
import ast
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

def test_index_json_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "tools/index.json" in gitignore


import os
from code_index import _extract_module, build_index, is_stale, ensure_fresh, lookup

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


def test_is_stale_when_index_missing(tmp_path):
    index_path = tmp_path / "index.json"
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "mod.py").write_text("def foo(): pass")
    assert is_stale(source_dir=source_dir, index_path=index_path) is True

def test_is_stale_when_source_newer(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    src_file = source_dir / "mod.py"
    src_file.write_text("def foo(): pass")
    index_path = tmp_path / "index.json"
    index_path.write_text("{}")
    old_time = src_file.stat().st_mtime - 10
    os.utime(index_path, (old_time, old_time))
    assert is_stale(source_dir=source_dir, index_path=index_path) is True

def test_is_not_stale_when_index_fresh(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    src_file = source_dir / "mod.py"
    src_file.write_text("def foo(): pass")
    index_path = tmp_path / "index.json"
    index_path.write_text("{}")
    new_time = src_file.stat().st_mtime + 10
    os.utime(index_path, (new_time, new_time))
    assert is_stale(source_dir=source_dir, index_path=index_path) is False

def test_ensure_fresh_creates_index(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "mymod.py").write_text("def hello(x: int) -> str: pass")
    index_path = tmp_path / "index.json"
    result = ensure_fresh(source_dir=source_dir, index_path=index_path)
    assert index_path.exists()
    assert "tensor_logic.mymod" in result
    assert "hello" in result["tensor_logic.mymod"]

def test_ensure_fresh_does_not_rebuild_when_current(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    src = source_dir / "mymod.py"
    src.write_text("def hello(): pass")
    index_path = tmp_path / "index.json"
    index_path.write_text('{"_meta": {"built_at": "old", "note": ""}, "tensor_logic.mymod": {}}')
    new_time = src.stat().st_mtime + 10
    os.utime(index_path, (new_time, new_time))
    result = ensure_fresh(source_dir=source_dir, index_path=index_path)
    assert result["tensor_logic.mymod"] == {}


import subprocess

def test_lookup_found(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "rules.py").write_text("""
class Schema:
    def __init__(self, name: str, relations: list):
        pass
    def add_rule(self): pass
    def _internal(self): pass
""")
    index_path = tmp_path / "index.json"
    out_file = tmp_path / "out.txt"
    rc = lookup("Schema", source_dir=source_dir, index_path=index_path, out=out_file)
    assert rc == 0
    output = out_file.read_text()
    assert "Schema [class]" in output
    assert "name: str" in output
    assert "add_rule" in output
    assert "_internal" not in output

def test_lookup_not_found(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "rules.py").write_text("def foo(): pass")
    index_path = tmp_path / "index.json"
    rc = lookup("Nonexistent", source_dir=source_dir, index_path=index_path, out=tmp_path / "out.txt")
    assert rc == 1

def test_cli_lookup_real_symbol():
    result = subprocess.run(
        [sys.executable, "tools/code_index.py", "--lookup", "Program"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0
    assert "Program [class]" in result.stdout

def test_cli_status_exits_0_when_fresh():
    subprocess.run([sys.executable, "tools/code_index.py", "--rebuild"], cwd=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "tools/code_index.py", "--status"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0
    assert "fresh" in result.stdout

def test_cli_lookup_missing_exits_1():
    result = subprocess.run(
        [sys.executable, "tools/code_index.py", "--lookup", "DoesNotExistXYZ"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 1
    assert "symbol not found" in result.stderr
