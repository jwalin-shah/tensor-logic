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
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
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
