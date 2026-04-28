# Code Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tools/code_index.py` — a stdlib-only AST extractor that indexes `tensor_logic/` API signatures into `tools/index.json` with lazy rebuild, CLI lookup, and a project CLAUDE.md planning instruction.

**Architecture:** Single Python script walks `tensor_logic/*.py` top-level AST nodes to extract classes (init args + public methods) and functions (args + return type). On every invocation it checks mtime to auto-rebuild if stale. CLI exposes `--lookup`, `--dump`, `--rebuild`, and `--status`.

**Tech Stack:** Python stdlib only (`ast`, `json`, `pathlib`, `argparse`). No external deps. Existing CocoIndex (`.cocoindex_code/`) is a separate vector search layer — unrelated, untouched.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tools/code_index.py` | Create | Extractor + freshness check + CLI |
| `tools/index.json` | Generated (gitignored) | Snapshot of current API signatures |
| `tests/test_code_index.py` | Create | Unit + integration tests |
| `.gitignore` | Modify | Add `tools/index.json` |
| `CLAUDE.md` | Create | Project-level planning instruction |

---

## Task 1: Gitignore + directory setup

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_code_index.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_index_json_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "tools/index.json" in gitignore
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_code_index.py::test_index_json_is_gitignored -v
```
Expected: FAIL — `AssertionError`

- [ ] **Step 3: Add entry to `.gitignore`**

Add one line at the end of `.gitignore`:
```
tools/index.json
```

- [ ] **Step 4: Create `tools/` directory**

```bash
mkdir -p tools
```

- [ ] **Step 5: Run to verify it passes**

```bash
pytest tests/test_code_index.py::test_index_json_is_gitignored -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add .gitignore tests/test_code_index.py
git commit -m "test: gitignore tools/index.json, scaffold test file"
```

---

## Task 2: AST extractor core

**Files:**
- Create: `tools/code_index.py`
- Test: `tests/test_code_index.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_code_index.py`:

```python
import ast
import sys
sys.path.insert(0, str(REPO_ROOT / "tools"))
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
    # Schema is defined in tensor_logic/rules.py as a class named Schema? 
    # Actually there's no Schema in rules.py — let's check Program in program.py
    assert "tensor_logic.program" in index
    assert "Program" in index["tensor_logic.program"]
    assert index["tensor_logic.program"]["Program"]["kind"] == "class"

def test_build_index_excludes_private_modules():
    index = build_index()
    # __init__ and __main__ should be excluded (start with _)
    assert not any(k.endswith(".__init__") or k.endswith(".__main__") for k in index)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_code_index.py -k "extract or build_index" -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'code_index'`

- [ ] **Step 3: Implement `_extract_module` and `build_index`**

Create `tools/code_index.py`:

```python
#!/usr/bin/env python3
"""
AST extractor for tensor_logic/ API signatures.

Usage:
    python tools/code_index.py --rebuild
    python tools/code_index.py --lookup Schema
    python tools/code_index.py --dump
    python tools/code_index.py --status
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_DIR = _REPO_ROOT / "tensor_logic"
_INDEX_PATH = Path(__file__).resolve().parent / "index.json"


def _extract_module(path: Path, module_prefix: str = "tensor_logic") -> dict:
    """Parse one .py file and return {module_key: {symbol: entry}}."""
    source = path.read_text()
    tree = ast.parse(source)
    module_name = f"{module_prefix}.{path.stem}"
    symbols: dict = {}

    for node in tree.body:
        if isinstance(node, (ast.ClassDef,)) and not node.name.startswith("_"):
            init_args: list[str] = []
            methods: list[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__":
                        for arg in item.args.args[1:]:  # skip self
                            ann = ast.unparse(arg.annotation) if arg.annotation else ""
                            init_args.append(f"{arg.arg}: {ann}" if ann else arg.arg)
                    elif not item.name.startswith("_"):
                        methods.append(item.name)
            symbols[node.name] = {
                "kind": "class",
                "init_args": init_args,
                "methods": methods,
            }

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            args: list[str] = []
            for arg in node.args.args:
                ann = ast.unparse(arg.annotation) if arg.annotation else ""
                args.append(f"{arg.arg}: {ann}" if ann else arg.arg)
            returns = ast.unparse(node.returns) if node.returns else ""
            symbols[node.name] = {
                "kind": "function",
                "args": args,
                "returns": returns,
            }

    return {module_name: symbols}


def build_index(source_dir: Path = _SOURCE_DIR) -> dict:
    """Build the full index from all non-private .py files in source_dir."""
    index: dict = {
        "_meta": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "note": "AST-extracted; runtime behavior (kwargs, dynamic dispatch) not captured",
        }
    }
    for path in sorted(source_dir.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        index.update(_extract_module(path))
    return index
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_code_index.py -k "extract or build_index" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/code_index.py tests/test_code_index.py
git commit -m "feat: code_index _extract_module + build_index — AST extractor core"
```

---

## Task 3: Freshness check and lazy rebuild

**Files:**
- Modify: `tools/code_index.py`
- Test: `tests/test_code_index.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_code_index.py`:

```python
from code_index import is_stale, ensure_fresh

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
    # Make index older than source
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
    # Make index newer than source
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
    # Write a stale index manually
    index_path.write_text('{"_meta": {"built_at": "old", "note": ""}, "tensor_logic.mymod": {}}')
    # Make it newer than source
    new_time = src.stat().st_mtime + 10
    os.utime(index_path, (new_time, new_time))
    result = ensure_fresh(source_dir=source_dir, index_path=index_path)
    # Should return the stale content — did not rebuild
    assert result["tensor_logic.mymod"] == {}
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_code_index.py -k "stale or ensure_fresh" -v
```
Expected: FAIL — `ImportError: cannot import name 'is_stale'`

- [ ] **Step 3: Implement `is_stale` and `ensure_fresh`**

Append to `tools/code_index.py` (after `build_index`):

```python
def is_stale(source_dir: Path = _SOURCE_DIR, index_path: Path = _INDEX_PATH) -> bool:
    """Return True if index.json is missing or older than any source file."""
    if not index_path.exists():
        return True
    index_mtime = index_path.stat().st_mtime
    source_files = list(source_dir.glob("*.py"))
    if not source_files:
        return False
    return max(p.stat().st_mtime for p in source_files) > index_mtime


def ensure_fresh(
    source_dir: Path = _SOURCE_DIR,
    index_path: Path = _INDEX_PATH,
) -> dict:
    """Return current index, rebuilding from source_dir if stale."""
    if is_stale(source_dir=source_dir, index_path=index_path):
        index = build_index(source_dir=source_dir)
        index_path.write_text(json.dumps(index, indent=2))
        return index
    return json.loads(index_path.read_text())
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_code_index.py -k "stale or ensure_fresh" -v
```
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add tools/code_index.py tests/test_code_index.py
git commit -m "feat: code_index is_stale + ensure_fresh — lazy rebuild on mtime check"
```

---

## Task 4: CLI interface

**Files:**
- Modify: `tools/code_index.py`
- Test: `tests/test_code_index.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_code_index.py`:

```python
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
    from code_index import lookup
    rc = lookup("Schema", source_dir=source_dir, index_path=index_path, out=tmp_path / "out.txt")
    assert rc == 0
    output = (tmp_path / "out.txt").read_text()
    assert "Schema [class]" in output
    assert "name: str" in output
    assert "add_rule" in output
    assert "_internal" not in output

def test_lookup_not_found(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "rules.py").write_text("def foo(): pass")
    index_path = tmp_path / "index.json"
    from code_index import lookup
    rc = lookup("Nonexistent", source_dir=source_dir, index_path=index_path, out=tmp_path / "out.txt")
    assert rc == 1

def test_cli_lookup_real_symbol():
    result = subprocess.run(
        ["python", "tools/code_index.py", "--lookup", "Program"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0
    assert "Program [class]" in result.stdout

def test_cli_status_exits_0_when_fresh():
    # Force a rebuild first
    subprocess.run(["python", "tools/code_index.py", "--rebuild"], cwd=str(REPO_ROOT))
    result = subprocess.run(
        ["python", "tools/code_index.py", "--status"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0
    assert "fresh" in result.stdout

def test_cli_lookup_missing_exits_1():
    result = subprocess.run(
        ["python", "tools/code_index.py", "--lookup", "DoesNotExistXYZ"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 1
    assert "symbol not found" in result.stderr
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_code_index.py -k "lookup or cli or status" -v
```
Expected: FAIL — `ImportError: cannot import name 'lookup'`

- [ ] **Step 3: Implement `lookup`, `dump`, `status`, and `main`**

Append to `tools/code_index.py`:

```python
def lookup(
    symbol: str,
    source_dir: Path = _SOURCE_DIR,
    index_path: Path = _INDEX_PATH,
    out=None,
) -> int:
    """Print entry for symbol. Returns 0 if found, 1 if not found."""
    import io
    index = ensure_fresh(source_dir=source_dir, index_path=index_path)
    matches = [
        (module_key, symbols[symbol])
        for module_key, symbols in index.items()
        if module_key != "_meta" and symbol in symbols
    ]
    if not matches:
        print(f"symbol not found: {symbol}", file=sys.stderr)
        return 1

    buf = io.StringIO()
    for module_key, entry in matches:
        buf.write(f"{module_key}.{symbol} [{entry['kind']}]\n")
        if entry["kind"] == "class":
            args_str = ", ".join(entry["init_args"]) if entry["init_args"] else ""
            buf.write(f"  __init__({args_str})\n")
            if entry["methods"]:
                buf.write(f"  methods: {', '.join(entry['methods'])}\n")
        elif entry["kind"] == "function":
            args_str = ", ".join(entry["args"])
            returns = f" -> {entry['returns']}" if entry["returns"] else ""
            buf.write(f"  ({args_str}){returns}\n")

    text = buf.getvalue()
    if out is not None:
        Path(out).write_text(text)
    else:
        print(text, end="")
    return 0


def dump(source_dir: Path = _SOURCE_DIR, index_path: Path = _INDEX_PATH) -> int:
    """Print all symbols in compact form."""
    index = ensure_fresh(source_dir=source_dir, index_path=index_path)
    for module_key, symbols in index.items():
        if module_key == "_meta":
            continue
        for name, entry in symbols.items():
            if entry["kind"] == "class":
                args_str = ", ".join(entry["init_args"]) if entry["init_args"] else ""
                print(f"{module_key}.{name} [class]  __init__({args_str})")
            elif entry["kind"] == "function":
                args_str = ", ".join(entry["args"])
                returns = f" -> {entry['returns']}" if entry["returns"] else ""
                print(f"{module_key}.{name} [function]  ({args_str}){returns}")
    return 0


def status(source_dir: Path = _SOURCE_DIR, index_path: Path = _INDEX_PATH) -> int:
    """Print freshness status. Exits 1 if stale."""
    if is_stale(source_dir=source_dir, index_path=index_path):
        print("stale: index.json is missing or older than source files")
        return 1
    print(f"fresh: {index_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tensor_logic/ code index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    group.add_argument("--lookup", metavar="SYMBOL", help="Look up a symbol")
    group.add_argument("--dump", action="store_true", help="Print all symbols")
    group.add_argument("--status", action="store_true", help="Check staleness")
    args = parser.parse_args(argv)

    if args.rebuild:
        index = build_index()
        _INDEX_PATH.write_text(json.dumps(index, indent=2))
        print(f"rebuilt: {len(index) - 1} modules indexed")
        return 0
    if args.lookup:
        return lookup(args.lookup)
    if args.dump:
        return dump()
    if args.status:
        return status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_code_index.py -k "lookup or cli or status" -v
```
Expected: 5 PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```
Expected: all pre-existing tests pass, new tests pass

- [ ] **Step 6: Commit**

```bash
git add tools/code_index.py tests/test_code_index.py
git commit -m "feat: code_index CLI — lookup, dump, status, main"
```

---

## Task 5: Project CLAUDE.md

**Files:**
- Create: `CLAUDE.md` (project root)

- [ ] **Step 1: Create `CLAUDE.md`**

```markdown
# tensor — Project Instructions

## Planning

Before writing any implementation plan that touches `tensor_logic/`, run:

```bash
python tools/code_index.py --lookup <RelevantSymbol>
```

to get current constructor signatures and function args. This prevents plan/code divergence from stale API knowledge. Use `--dump` to see all indexed symbols.

## Test suite

```bash
pytest tests/ -v
```

## Notes

- `AGENTS.md` is memjuice-managed — do not edit it manually.
- `.cocoindex_code/` contains a vector embedding index (semantic search) — separate from `tools/index.json` (structured API signatures).
```

- [ ] **Step 2: Verify it renders**

```bash
cat CLAUDE.md
```
Expected: file prints cleanly with no encoding issues

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add project CLAUDE.md with code_index planning instruction"
```

---

## Task 6: Smoke test end-to-end

**Files:** none (validation only)

- [ ] **Step 1: Delete index.json if it exists, verify lookup triggers auto-rebuild**

```bash
rm -f tools/index.json
python tools/code_index.py --lookup Program
```
Expected output:
```
tensor_logic.program.Program [class]
  __init__(...)
  methods: ...
```

- [ ] **Step 2: Verify status shows fresh after lookup**

```bash
python tools/code_index.py --status
```
Expected: `fresh: .../tools/index.json` with exit code 0

- [ ] **Step 3: Touch a source file, verify status shows stale**

```bash
touch tensor_logic/rules.py
python tools/code_index.py --status
```
Expected: `stale: index.json is missing or older than source files` with exit code 1

- [ ] **Step 4: Verify lookup auto-rebuilds from stale**

```bash
python tools/code_index.py --lookup Rule
```
Expected output:
```
tensor_logic.rules.Rule [class]
  __init__(...)
  methods: ...
```

- [ ] **Step 5: Run full test suite one final time**

```bash
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Final commit**

```bash
git add tools/index.json 2>/dev/null; true  # only if not gitignored already
git status  # verify tools/index.json does NOT appear (gitignored)
git commit --allow-empty -m "chore: smoke test complete — code index end-to-end verified"
```
