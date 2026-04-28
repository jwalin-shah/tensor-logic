from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re


SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}


@dataclass(frozen=True)
class PythonImportGraph:
    modules: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    symbols: dict[str, str]


def ingest_python(root: str | Path) -> PythonImportGraph:
    root_path = Path(root).resolve()
    module_root = root_path.parent if root_path.is_file() else root_path
    files = tuple(_iter_python_files(root_path))
    module_by_file = {path: _module_name(module_root, path) for path in files}
    known = set(module_by_file.values())
    edges: set[tuple[str, str]] = set()
    for path, module in module_by_file.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _resolve_known(alias.name, known)
                    if target is not None and target != module:
                        edges.add((module, target))
            elif isinstance(node, ast.ImportFrom):
                candidates = _import_from_candidates(node, module, path.name == "__init__.py")
                for candidate in candidates:
                    target = _resolve_known(candidate, known)
                    if target is not None and target != module:
                        edges.add((module, target))
    modules = tuple(sorted(known))
    symbols = _module_symbols(modules)
    return PythonImportGraph(modules, tuple(sorted(edges)), symbols)


def render_python_imports_tl(graph: PythonImportGraph) -> str:
    symbols = [graph.symbols[module] for module in graph.modules]
    lines = [
        "domain Module {",
        "  " + " ".join(symbols),
        "}",
        "",
        "relation imports(Module, Module)",
        "relation depends_on(Module, Module)",
        "",
    ]
    for src, dst in graph.edges:
        lines.append(f"fact imports({graph.symbols[src]}, {graph.symbols[dst]})")
    lines.extend(
        [
            "",
            "rule depends_on(x,z) := (imports(x,z) + depends_on(x,y) * imports(y,z)).step()",
            "",
        ]
    )
    return "\n".join(lines)


def _iter_python_files(root: Path):
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        yield path


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else path.stem


def _import_from_candidates(
    node: ast.ImportFrom,
    current_module: str,
    is_package: bool,
) -> list[str]:
    package = current_module
    if not is_package:
        package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    base = node.module or ""
    if node.level:
        parts = package.split(".") if package else []
        keep = max(0, len(parts) - node.level + 1)
        prefix = ".".join(parts[:keep])
        base = f"{prefix}.{base}" if prefix and base else prefix or base
    candidates = []
    if node.module:
        candidates.append(base)
    for alias in node.names:
        if alias.name == "*":
            continue
        candidates.append(f"{base}.{alias.name}" if base else alias.name)
    return candidates


def _resolve_known(name: str, known: set[str]) -> str | None:
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in known:
            return candidate
    return None


def _module_symbols(modules: tuple[str, ...]) -> dict[str, str]:
    used: set[str] = set()
    symbols: dict[str, str] = {}
    for module in modules:
        base = re.sub(r"\W+", "_", module)
        if not re.match(r"[A-Za-z_]", base):
            base = f"m_{base}"
        symbol = base
        i = 2
        while symbol in used:
            symbol = f"{base}_{i}"
            i += 1
        used.add(symbol)
        symbols[module] = symbol
    return symbols
