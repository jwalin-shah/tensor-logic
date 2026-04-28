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
