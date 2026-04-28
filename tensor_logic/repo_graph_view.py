from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .file_format import load_tl
from .proofs import Proof, fmt_proof_tree, prove


@dataclass(frozen=True)
class RepoGraphData:
    modules: tuple[str, ...]
    imports: tuple[tuple[str, str], ...]


def load_repo_graph(path: str) -> RepoGraphData:
    loaded = load_tl(path)
    program = loaded.program
    if "Module" not in program.domains:
        raise ValueError("domain 'Module' is required for repo graph view")
    if "imports" not in program.relations:
        raise ValueError("relation 'imports' is required for repo graph view")
    modules = tuple(program.domains["Module"].symbols)
    imports_rel = program.relations["imports"]
    edges = [
        (src, dst)
        for src in modules
        for dst in modules
        if imports_rel.value(src, dst, semiring="boolean") > 0
    ]
    return RepoGraphData(modules=modules, imports=tuple(edges))


def build_adjacency(modules: Iterable[str], imports: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    adjacency = {module: [] for module in modules}
    for src, dst in imports:
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, [])
    for dsts in adjacency.values():
        dsts.sort()
    return adjacency


def filter_modules(modules: Iterable[str], query: str) -> list[str]:
    query = query.strip().lower()
    if not query:
        return sorted(modules)
    return sorted(module for module in modules if query in module.lower())


def imports_path(modules: Iterable[str], imports: Iterable[tuple[str, str]], src: str, dst: str) -> list[str] | None:
    adjacency = build_adjacency(modules, imports)
    if src not in adjacency or dst not in adjacency:
        return None
    queue = deque([src])
    parent: dict[str, str | None] = {src: None}
    while queue:
        node = queue.popleft()
        if node == dst:
            break
        for nxt in adjacency[node]:
            if nxt in parent:
                continue
            parent[nxt] = node
            queue.append(nxt)
    if dst not in parent:
        return None
    path = [dst]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def extract_path_from_proof(proof: Proof, base_relation: str = "imports") -> list[str] | None:
    edges = _proof_edges(proof, base_relation)
    if not edges:
        return None
    modules = {symbol for edge in edges for symbol in edge}
    return imports_path(modules, edges, proof.head[1], proof.head[2])


def format_adjacency(adjacency: dict[str, list[str]]) -> str:
    lines = []
    for module in sorted(adjacency):
        imports = ", ".join(adjacency[module]) if adjacency[module] else "(none)"
        lines.append(f"- {module}: {imports}")
    return "\n".join(lines)


def dependency_report(path: str, module: str | None = None, src: str | None = None, dst: str | None = None) -> str:
    loaded = load_tl(path)
    graph = load_repo_graph(path)
    adjacency = build_adjacency(graph.modules, graph.imports)
    lines = ["Repo dependency graph report", f"modules={len(graph.modules)} imports={len(graph.imports)}"]
    if module is not None:
        if module not in adjacency:
            raise ValueError(f"Unknown module: {module}")
        direct = adjacency[module]
        lines.append(f"direct imports({module}): {', '.join(direct) if direct else '(none)'}")
    if src is not None and dst is not None:
        answer = bool(loaded.program.query("depends_on", src, dst, recursive=True))
        lines.append(f"depends_on({src}, {dst}) = {answer}")
        if answer:
            proof = prove(loaded.program, "depends_on", src, dst, recursive=True)
            if proof is not None:
                path = extract_path_from_proof(proof)
                if path is not None:
                    lines.append("path: " + " -> ".join(path))
                lines.append(fmt_proof_tree(proof))
    lines.append("adjacency:")
    lines.append(format_adjacency(adjacency))
    return "\n".join(lines)


def repo_graph_repl(path: str, out=None) -> None:
    import sys

    if out is None:
        out = sys.stdout
    loaded = load_tl(path)
    graph = load_repo_graph(path)
    adjacency = build_adjacency(graph.modules, graph.imports)
    print("tensor-logic repo dependency graph view", file=out)
    print("Commands: search <text> | show <module> | depends <src> <dst> | adj | help | exit", file=out)
    while True:
        try:
            line = input("repo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            break
        if not line:
            continue
        if line in {"exit", "quit"}:
            break
        if line == "help":
            print("search <text>     Filter Module symbols", file=out)
            print("show <module>     Direct imports for one module", file=out)
            print("depends a b       Query/prove depends_on(a,b)", file=out)
            print("adj               Full import adjacency list", file=out)
            continue
        if line == "adj":
            print(format_adjacency(adjacency), file=out)
            continue
        if line.startswith("search "):
            hits = filter_modules(graph.modules, line[len("search "):])
            print("\n".join(f"- {hit}" for hit in hits) if hits else "No matching modules.", file=out)
            continue
        if line.startswith("show "):
            module = line[len("show "):].strip()
            if module not in adjacency:
                print(f"Unknown module: {module}", file=out)
                continue
            direct = adjacency[module]
            print(f"direct imports({module}): {', '.join(direct) if direct else '(none)'}", file=out)
            continue
        if line.startswith("depends "):
            parts = line.split()
            if len(parts) != 3:
                print("Usage: depends <src> <dst>", file=out)
                continue
            src, dst = parts[1], parts[2]
            if src not in adjacency or dst not in adjacency:
                print("Unknown module symbol(s)", file=out)
                continue
            print(dependency_report(path, src=src, dst=dst), file=out)
            continue
        print(f"Unknown command: {line}", file=out)


def _proof_edges(proof: Proof, base_relation: str) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    rel, src, dst = proof.head
    if rel == base_relation:
        edges.add((src, dst))
    for child in proof.body:
        edges.update(_proof_edges(child, base_relation))
    return edges
