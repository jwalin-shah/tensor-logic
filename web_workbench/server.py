from __future__ import annotations

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tensor_logic.http_api import ApiError, prove_source, query_source, run_source
from tensor_logic.execution import load_tl_source
from tensor_logic.ingest import ingest_python, render_python_imports_tl
from tensor_logic.repo_graph_view import imports_path


STATIC_DIR = ROOT / "static"


class WorkbenchHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self._write_json({"error": "Invalid JSON payload"}, HTTPStatus.BAD_REQUEST)
            return
        action = self.path.rsplit("/", 1)[-1]
        result, status = run_tensor_logic_action(action, payload)
        self._write_json(result, status)

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_tensor_logic_action(action: str, payload: dict) -> tuple[dict, HTTPStatus]:
    if action == "ingest-python":
        return ingest_python_action(payload)
    if action == "repo-impact":
        return repo_impact_action(payload)
    if action == "repo-overview":
        return repo_overview_action(payload)
    if action == "repo-brief":
        return repo_brief_action(payload)
    if action == "repo-compare":
        return repo_compare_action(payload)

    source = str(payload.get("source", ""))
    if not source.strip():
        return {"error": "Missing source"}, HTTPStatus.BAD_REQUEST
    relation = str(payload.get("relation", "")).strip()
    arg1 = str(payload.get("arg1", "")).strip()
    arg2 = str(payload.get("arg2", "")).strip()
    recursive = bool(payload.get("recursive", False))

    start = time.perf_counter()
    try:
        if action == "run":
            result_payload = run_source(source)
            stdout = "\n".join(result_payload.get("outputs", []))
            command = "run source.tl"
        elif action in {"query", "prove", "why-not"}:
            if not (relation and arg1 and arg2):
                return {"error": "relation, arg1, and arg2 are required"}, HTTPStatus.BAD_REQUEST
            args = [arg1, arg2]
            if action == "query":
                result_payload = query_source(source, relation, args, recursive=recursive)
                stdout = render_query_text(result_payload)
                command = f"query {relation}({arg1}, {arg2})"
            else:
                why_not = action == "why-not"
                result_payload = prove_source(
                    source,
                    relation,
                    args,
                    recursive=recursive,
                    why_not=why_not,
                    format_type="json",
                )
                stdout = render_proof_text(result_payload, relation, args)
                command = f"{'why-not' if why_not else 'prove'} {relation}({arg1}, {arg2})"
            if action == "why-not":
                command += " --why-not"
            if recursive:
                command += " --recursive"
        else:
            return {"error": f"Unknown action: {action}"}, HTTPStatus.NOT_FOUND
    except ApiError as exc:
        return {"error": exc.message, "exit_code": 1, "stderr": exc.message}, HTTPStatus(exc.status_code)
    except Exception as exc:
        message = str(exc)
        return {"error": message, "exit_code": 1, "stderr": message}, HTTPStatus.BAD_REQUEST

    return (
        {
            "action": action,
            "command": command,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "payload": result_payload,
        },
        HTTPStatus.OK,
    )


def repo_impact_action(payload: dict) -> tuple[dict, HTTPStatus]:
    source = str(payload.get("source", ""))
    module = str(payload.get("module", "")).strip()
    if not source.strip():
        return {"error": "Missing source", "exit_code": 1}, HTTPStatus.BAD_REQUEST
    if not module:
        return {"error": "module is required", "exit_code": 1}, HTTPStatus.BAD_REQUEST

    start = time.perf_counter()
    try:
        graph = repo_graph_from_source(source)
        if module not in graph["modules"]:
            return {"error": f"Unknown module: {module}", "exit_code": 1}, HTTPStatus.BAD_REQUEST
        impact = build_repo_impact(graph, module)
        impact = attach_module_metadata(impact, payload.get("metadata"))
    except Exception as exc:
        message = str(exc)
        return {"error": message, "exit_code": 1, "stderr": message}, HTTPStatus.BAD_REQUEST

    stdout = render_repo_impact_text(impact)
    return (
        {
            "action": "repo-impact",
            "command": f"repo-impact {module}",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "payload": impact,
        },
        HTTPStatus.OK,
    )


def repo_overview_action(payload: dict) -> tuple[dict, HTTPStatus]:
    source = str(payload.get("source", ""))
    if not source.strip():
        return {"error": "Missing source", "exit_code": 1}, HTTPStatus.BAD_REQUEST

    start = time.perf_counter()
    try:
        graph = repo_graph_from_source(source)
        overview = build_repo_overview(graph)
        overview = attach_overview_metadata(overview, payload.get("metadata"))
    except Exception as exc:
        message = str(exc)
        return {"error": message, "exit_code": 1, "stderr": message}, HTTPStatus.BAD_REQUEST

    stdout = render_repo_overview_text(overview)
    return (
        {
            "action": "repo-overview",
            "command": "repo-overview",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "payload": overview,
        },
        HTTPStatus.OK,
    )


def repo_brief_action(payload: dict) -> tuple[dict, HTTPStatus]:
    source = str(payload.get("source", ""))
    module = str(payload.get("module", "")).strip()
    if not source.strip():
        return {"error": "Missing source", "exit_code": 1}, HTTPStatus.BAD_REQUEST
    if not module:
        return {"error": "module is required", "exit_code": 1}, HTTPStatus.BAD_REQUEST

    start = time.perf_counter()
    try:
        graph = repo_graph_from_source(source)
        if module not in graph["modules"]:
            return {"error": f"Unknown module: {module}", "exit_code": 1}, HTTPStatus.BAD_REQUEST
        impact = build_repo_impact(graph, module)
        impact = attach_module_metadata(impact, payload.get("metadata"))
        brief = build_repo_change_brief(impact)
    except Exception as exc:
        message = str(exc)
        return {"error": message, "exit_code": 1, "stderr": message}, HTTPStatus.BAD_REQUEST

    return (
        {
            "action": "repo-brief",
            "command": f"repo-brief {module}",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "exit_code": 0,
            "stdout": brief["markdown"],
            "stderr": "",
            "payload": brief,
        },
        HTTPStatus.OK,
    )


def repo_compare_action(payload: dict) -> tuple[dict, HTTPStatus]:
    before_source = str(payload.get("before_source", ""))
    after_source = str(payload.get("after_source", ""))
    if not before_source.strip():
        return {"error": "before_source is required", "exit_code": 1}, HTTPStatus.BAD_REQUEST
    if not after_source.strip():
        return {"error": "after_source is required", "exit_code": 1}, HTTPStatus.BAD_REQUEST

    start = time.perf_counter()
    try:
        before_graph = repo_graph_from_source(before_source)
        after_graph = repo_graph_from_source(after_source)
        compare = build_repo_compare(
            before_graph,
            after_graph,
            payload.get("before_metadata"),
            payload.get("after_metadata") or payload.get("metadata"),
        )
    except Exception as exc:
        message = str(exc)
        return {"error": message, "exit_code": 1, "stderr": message}, HTTPStatus.BAD_REQUEST

    return (
        {
            "action": "repo-compare",
            "command": "repo-compare baseline current",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "exit_code": 0,
            "stdout": compare["markdown"],
            "stderr": "",
            "payload": compare,
        },
        HTTPStatus.OK,
    )


def repo_graph_from_source(source: str) -> dict:
    loaded = load_tl_source(source, prefix="tensor_logic_workbench_repo_")
    program = loaded.program
    if "Module" not in program.domains:
        raise ValueError("domain 'Module' is required")
    if "imports" not in program.relations:
        raise ValueError("relation 'imports' is required")
    if "depends_on" not in program.relations:
        raise ValueError("relation 'depends_on' is required")
    modules = tuple(program.domains["Module"].symbols)
    imports_rel = program.relations["imports"]
    imports = tuple(
        sorted(
            (src, dst)
            for src in modules
            for dst in modules
            if imports_rel.value(src, dst, semiring="boolean") > 0
        )
    )
    return {"program": program, "modules": modules, "imports": imports}


def build_repo_overview(graph: dict) -> dict:
    modules = graph["modules"]
    imports = graph["imports"]
    program = graph["program"]
    direct_imports = {module: 0 for module in modules}
    direct_imported_by = {module: 0 for module in modules}
    for src, dst in imports:
        direct_imports[src] += 1
        direct_imported_by[dst] += 1

    transitive_dependencies: dict[str, int] = {}
    transitive_dependents: dict[str, int] = {}
    for module in modules:
        transitive_dependencies[module] = sum(
            1
            for dst in modules
            if dst != module and bool(program.query("depends_on", module, dst, recursive=True))
        )
        transitive_dependents[module] = sum(
            1
            for src in modules
            if src != module and bool(program.query("depends_on", src, module, recursive=True))
        )

    cycles = strongly_connected_components(modules, imports)
    payload = {
        "modules": len(modules),
        "imports": len(imports),
        "top_dependents": rank_counts(transitive_dependents),
        "top_dependencies": rank_counts(transitive_dependencies),
        "direct_import_hubs": rank_counts(direct_imports),
        "direct_imported_hubs": rank_counts(direct_imported_by),
        "entrypoints": sorted(module for module in modules if direct_imported_by[module] == 0 and direct_imports[module] > 0),
        "leaves": sorted(module for module in modules if direct_imports[module] == 0 and direct_imported_by[module] > 0),
        "cycles": cycles,
    }
    return payload


def rank_counts(counts: dict[str, int], limit: int = 12) -> list[dict]:
    ranked = sorted(
        ({"module": module, "count": count} for module, count in counts.items() if count > 0),
        key=lambda item: (-item["count"], item["module"]),
    )
    return ranked[:limit]


def strongly_connected_components(
    modules: tuple[str, ...],
    imports: tuple[tuple[str, str], ...],
) -> list[list[str]]:
    adjacency = {module: [] for module in modules}
    for src, dst in imports:
        adjacency[src].append(dst)
    for targets in adjacency.values():
        targets.sort()

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)

        for target in adjacency[module]:
            if target not in indices:
                visit(target)
                lowlinks[module] = min(lowlinks[module], lowlinks[target])
            elif target in on_stack:
                lowlinks[module] = min(lowlinks[module], indices[target])

        if lowlinks[module] == indices[module]:
            component: list[str] = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == module:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for module in modules:
        if module not in indices:
            visit(module)

    return sorted(components, key=lambda component: (-len(component), component))


def build_repo_impact(graph: dict, module: str) -> dict:
    modules = graph["modules"]
    imports = graph["imports"]
    program = graph["program"]
    direct_imports = sorted(dst for src, dst in imports if src == module)
    direct_imported_by = sorted(src for src, dst in imports if dst == module)
    dependencies = sorted(
        dst
        for dst in modules
        if dst != module and bool(program.query("depends_on", module, dst, recursive=True))
    )
    dependents = sorted(
        src
        for src in modules
        if src != module and bool(program.query("depends_on", src, module, recursive=True))
    )
    return {
        "module": module,
        "modules": len(modules),
        "imports": len(imports),
        "direct_imports": direct_imports,
        "direct_imported_by": direct_imported_by,
        "transitive_dependencies": dependencies,
        "transitive_dependents": dependents,
        "dependency_paths": path_map(modules, imports, module, dependencies),
        "dependent_paths": path_map(modules, imports, None, dependents, target=module),
    }


def build_repo_change_brief(impact: dict) -> dict:
    module = impact["module"]
    details = impact.get("module_details", {})
    module_detail = details.get(module, {"symbol": module, "module": module, "file": None})
    blast_radius = len(impact["transitive_dependents"])
    coupling = len(impact["transitive_dependencies"])
    risk_level = change_risk_level(blast_radius, coupling)

    read_symbols = unique_symbols(
        [module]
        + impact["direct_imports"]
        + impact["direct_imported_by"]
        + impact["transitive_dependencies"][:4]
    )
    regression_symbols = unique_symbols(impact["direct_imported_by"] + impact["transitive_dependents"])
    path_examples = build_path_examples(impact)
    proof_checks = build_proof_checks(impact)

    brief = {
        "module": module,
        "module_details": details,
        "risk_level": risk_level,
        "blast_radius": blast_radius,
        "coupling": coupling,
        "summary": (
            f"{display_module(module_detail)} has {blast_radius} transitive dependents "
            f"and {coupling} transitive dependencies."
        ),
        "read_first": detail_rows(read_symbols, details),
        "regression_targets": detail_rows(regression_symbols[:12], details),
        "path_examples": path_examples,
        "proof_checks": proof_checks,
    }
    brief["markdown"] = render_repo_brief_markdown(brief)
    return brief


def build_repo_compare(before_graph: dict, after_graph: dict, before_metadata, after_metadata) -> dict:
    before_modules = set(before_graph["modules"])
    after_modules = set(after_graph["modules"])
    before_imports = set(before_graph["imports"])
    after_imports = set(after_graph["imports"])

    added_modules = sorted(after_modules - before_modules)
    removed_modules = sorted(before_modules - after_modules)
    added_imports = sorted(after_imports - before_imports)
    removed_imports = sorted(before_imports - after_imports)
    before_cycles = cycle_keys(strongly_connected_components(before_graph["modules"], before_graph["imports"]))
    after_cycles = cycle_keys(strongly_connected_components(after_graph["modules"], after_graph["imports"]))
    introduced_cycles = sorted(after_cycles - before_cycles)
    resolved_cycles = sorted(before_cycles - after_cycles)
    before_blast = transitive_dependent_counts(before_graph)
    after_blast = transitive_dependent_counts(after_graph)
    blast_radius_deltas = rank_blast_radius_deltas(before_blast, after_blast)

    touched = touched_modules(added_modules, removed_modules, added_imports, removed_imports)
    suggested_checks = build_compare_checks(after_graph, added_imports, removed_imports)
    metadata = merge_metadata(before_metadata, after_metadata)
    detail_symbols = set(touched)
    for item in blast_radius_deltas:
        detail_symbols.add(item["module"])
    for cycle in set(introduced_cycles) | set(resolved_cycles):
        detail_symbols.update(cycle)

    compare = {
        "before": {"modules": len(before_modules), "imports": len(before_imports), "cycles": len(before_cycles)},
        "after": {"modules": len(after_modules), "imports": len(after_imports), "cycles": len(after_cycles)},
        "risk_level": compare_risk_level(added_imports, removed_imports, introduced_cycles, blast_radius_deltas),
        "added_modules": added_modules,
        "removed_modules": removed_modules,
        "added_imports": [list(edge) for edge in added_imports],
        "removed_imports": [list(edge) for edge in removed_imports],
        "introduced_cycles": [list(cycle) for cycle in introduced_cycles],
        "resolved_cycles": [list(cycle) for cycle in resolved_cycles],
        "blast_radius_deltas": blast_radius_deltas,
        "touched_modules": touched,
        "suggested_checks": suggested_checks,
        "module_details": module_details(detail_symbols, metadata),
    }
    compare["summary"] = compare_summary(compare)
    compare["markdown"] = render_repo_compare_markdown(compare)
    return compare


def cycle_keys(cycles: list[list[str]]) -> set[tuple[str, ...]]:
    return {tuple(cycle) for cycle in cycles}


def transitive_dependent_counts(graph: dict) -> dict[str, int]:
    modules = graph["modules"]
    program = graph["program"]
    return {
        module: sum(
            1
            for src in modules
            if src != module and bool(program.query("depends_on", src, module, recursive=True))
        )
        for module in modules
    }


def rank_blast_radius_deltas(before: dict[str, int], after: dict[str, int], limit: int = 12) -> list[dict]:
    rows = []
    for module in sorted(set(before) | set(after)):
        before_count = before.get(module, 0)
        after_count = after.get(module, 0)
        delta = after_count - before_count
        if delta:
            rows.append({"module": module, "before": before_count, "after": after_count, "delta": delta})
    rows.sort(key=lambda item: (-abs(item["delta"]), item["module"]))
    return rows[:limit]


def touched_modules(
    added_modules: list[str],
    removed_modules: list[str],
    added_imports: list[tuple[str, str]],
    removed_imports: list[tuple[str, str]],
) -> list[str]:
    touched = set(added_modules) | set(removed_modules)
    for src, dst in added_imports + removed_imports:
        touched.add(src)
        touched.add(dst)
    return sorted(touched)


def build_compare_checks(
    after_graph: dict,
    added_imports: list[tuple[str, str]],
    removed_imports: list[tuple[str, str]],
) -> list[dict]:
    checks: list[dict] = []
    for src, dst in added_imports:
        checks.append(
            {
                "kind": "added dependency",
                "action": "prove",
                "relation": "depends_on",
                "arg1": src,
                "arg2": dst,
                "recursive": True,
                "expected": True,
            }
        )
    for src, dst in removed_imports:
        still_reachable = src in after_graph["modules"] and dst in after_graph["modules"] and bool(
            after_graph["program"].query("depends_on", src, dst, recursive=True)
        )
        checks.append(
            {
                "kind": "removed direct import" if still_reachable else "removed dependency",
                "action": "prove" if still_reachable else "why-not",
                "relation": "depends_on",
                "arg1": src,
                "arg2": dst,
                "recursive": True,
                "expected": still_reachable,
            }
        )
    return checks[:12]


def merge_metadata(before_metadata, after_metadata) -> dict:
    merged = {"symbol_to_module": {}, "symbol_to_file": {}}
    for metadata in (before_metadata, after_metadata):
        if not isinstance(metadata, dict):
            continue
        for key in ("symbol_to_module", "symbol_to_file"):
            values = metadata.get(key)
            if isinstance(values, dict):
                merged[key].update(values)
    return merged


def compare_risk_level(
    added_imports: list[tuple[str, str]],
    removed_imports: list[tuple[str, str]],
    introduced_cycles: set[tuple[str, ...]],
    blast_radius_deltas: list[dict],
) -> str:
    max_delta = max((abs(item["delta"]) for item in blast_radius_deltas), default=0)
    change_count = len(added_imports) + len(removed_imports)
    if introduced_cycles or max_delta >= 8 or change_count >= 12:
        return "high"
    if max_delta >= 3 or change_count >= 4:
        return "medium"
    return "low"


def compare_summary(payload: dict) -> str:
    return (
        f"{len(payload['added_imports'])} imports added, {len(payload['removed_imports'])} removed, "
        f"{len(payload['added_modules'])} modules added, {len(payload['removed_modules'])} removed."
    )


def change_risk_level(blast_radius: int, coupling: int) -> str:
    if blast_radius >= 8 or coupling >= 12:
        return "high"
    if blast_radius >= 3 or coupling >= 6:
        return "medium"
    return "low"


def unique_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        if symbol and symbol not in seen:
            seen.add(symbol)
            ordered.append(symbol)
    return ordered


def detail_rows(symbols: list[str], details: dict) -> list[dict]:
    return [details.get(symbol, {"symbol": symbol, "module": symbol, "file": None}) for symbol in symbols]


def build_path_examples(impact: dict) -> list[dict]:
    examples: list[dict] = []
    for module, path in sorted(impact.get("dependent_paths", {}).items()):
        examples.append({"kind": "dependent", "module": module, "path": path})
    for module, path in sorted(impact.get("dependency_paths", {}).items()):
        examples.append({"kind": "dependency", "module": module, "path": path})
    return examples[:12]


def build_proof_checks(impact: dict) -> list[dict]:
    module = impact["module"]
    checks: list[dict] = []
    for dependent in unique_symbols(impact["direct_imported_by"] + impact["transitive_dependents"]):
        checks.append({"relation": "depends_on", "arg1": dependent, "arg2": module, "recursive": True})
    for dependency in unique_symbols(impact["direct_imports"] + impact["transitive_dependencies"]):
        checks.append({"relation": "depends_on", "arg1": module, "arg2": dependency, "recursive": True})
    return checks[:10]


def attach_module_metadata(payload: dict, metadata) -> dict:
    if not isinstance(metadata, dict):
        return payload

    symbols = {payload["module"]}
    for key in (
        "direct_imports",
        "direct_imported_by",
        "transitive_dependencies",
        "transitive_dependents",
    ):
        symbols.update(payload.get(key, []))
    for paths in (payload.get("dependency_paths", {}), payload.get("dependent_paths", {})):
        for symbol, path in paths.items():
            symbols.add(symbol)
            symbols.update(path)

    payload = dict(payload)
    payload["module_details"] = module_details(symbols, metadata)
    return payload


def attach_overview_metadata(payload: dict, metadata) -> dict:
    if not isinstance(metadata, dict):
        return payload
    symbols: set[str] = set(payload.get("entrypoints", [])) | set(payload.get("leaves", []))
    for key in ("top_dependents", "top_dependencies", "direct_import_hubs", "direct_imported_hubs"):
        symbols.update(item["module"] for item in payload.get(key, []))
    for cycle in payload.get("cycles", []):
        symbols.update(cycle)
    payload = dict(payload)
    payload["module_details"] = module_details(symbols, metadata)
    return payload


def module_details(symbols: set[str], metadata) -> dict:
    if not isinstance(metadata, dict):
        return {}
    symbol_to_module = metadata.get("symbol_to_module")
    symbol_to_file = metadata.get("symbol_to_file")
    if not isinstance(symbol_to_module, dict):
        symbol_to_module = {}
    if not isinstance(symbol_to_file, dict):
        symbol_to_file = {}
    return {
        symbol: {
            "symbol": symbol,
            "module": symbol_to_module.get(symbol, symbol),
            "file": symbol_to_file.get(symbol),
        }
        for symbol in sorted(symbols)
    }


def path_map(
    modules: tuple[str, ...],
    imports: tuple[tuple[str, str], ...],
    src: str | None,
    destinations: list[str],
    *,
    target: str | None = None,
) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for item in destinations:
        start = src or item
        end = target or item
        path = imports_path(modules, imports, start, end)
        if path is not None:
            paths[item] = path
    return paths


def render_repo_impact_text(payload: dict) -> str:
    detail = payload.get("module_details", {}).get(payload["module"], {})
    module_label = detail.get("module") or payload["module"]
    lines = [
        f"impact({payload['module']})",
        f"module: {module_label}",
        f"direct imports: {format_list(payload['direct_imports'])}",
        f"direct imported by: {format_list(payload['direct_imported_by'])}",
        f"transitive dependencies ({len(payload['transitive_dependencies'])}): {format_list(payload['transitive_dependencies'])}",
        f"transitive dependents ({len(payload['transitive_dependents'])}): {format_list(payload['transitive_dependents'])}",
    ]
    if payload["dependent_paths"]:
        lines.append("dependent paths:")
        for module, path in payload["dependent_paths"].items():
            lines.append(f"- {module}: {' -> '.join(path)}")
    if payload["dependency_paths"]:
        lines.append("dependency paths:")
        for module, path in payload["dependency_paths"].items():
            lines.append(f"- {module}: {' -> '.join(path)}")
    return "\n".join(lines)


def render_repo_overview_text(payload: dict) -> str:
    lines = [
        "repo overview",
        f"modules={payload['modules']} imports={payload['imports']} cycles={len(payload['cycles'])}",
        "top blast radius:",
    ]
    lines.extend(render_ranked_lines(payload["top_dependents"]))
    lines.append("top dependency fanout:")
    lines.extend(render_ranked_lines(payload["top_dependencies"]))
    lines.append(f"entrypoints: {format_list(payload['entrypoints'][:12])}")
    lines.append(f"leaves: {format_list(payload['leaves'][:12])}")
    if payload["cycles"]:
        lines.append("cycles:")
        for cycle in payload["cycles"][:8]:
            lines.append("- " + " -> ".join(cycle))
    return "\n".join(lines)


def render_repo_brief_markdown(payload: dict) -> str:
    module_detail = payload["module_details"].get(payload["module"], {})
    lines = [
        f"# Change brief: {display_module(module_detail)}",
        "",
        f"Risk: {payload['risk_level']} ({payload['blast_radius']} transitive dependents, {payload['coupling']} transitive dependencies)",
    ]
    if module_detail.get("file"):
        lines.append(f"File: {module_detail['file']}")
    lines.extend(["", "Read first:"])
    lines.extend(render_detail_lines(payload["read_first"]))
    lines.extend(["", "Regression watch:"])
    lines.extend(render_detail_lines(payload["regression_targets"]))
    if payload["path_examples"]:
        lines.extend(["", "Proof paths:"])
        for item in payload["path_examples"]:
            lines.append(f"- {item['kind']}: {' -> '.join(item['path'])}")
    if payload["proof_checks"]:
        lines.extend(["", "Suggested Tensor Logic checks:"])
        for check in payload["proof_checks"]:
            lines.append(f"- prove {check['relation']}({check['arg1']}, {check['arg2']}) recursive")
    return "\n".join(lines)


def render_repo_compare_markdown(payload: dict) -> str:
    lines = [
        "# Repo graph compare",
        "",
        f"Risk: {payload['risk_level']}",
        f"Before: {payload['before']['modules']} modules / {payload['before']['imports']} imports / {payload['before']['cycles']} cycles",
        f"After: {payload['after']['modules']} modules / {payload['after']['imports']} imports / {payload['after']['cycles']} cycles",
        "",
        payload["summary"],
        "",
        "Added imports:",
    ]
    lines.extend(render_edge_lines(payload["added_imports"], payload.get("module_details", {})))
    lines.extend(["", "Removed imports:"])
    lines.extend(render_edge_lines(payload["removed_imports"], payload.get("module_details", {})))
    lines.extend(["", "Blast-radius changes:"])
    if payload["blast_radius_deltas"]:
        for item in payload["blast_radius_deltas"]:
            detail = payload.get("module_details", {}).get(item["module"], {})
            lines.append(
                f"- {display_module(detail)}: {item['before']} -> {item['after']} ({item['delta']:+d})"
            )
    else:
        lines.append("- (none)")
    if payload["introduced_cycles"]:
        lines.extend(["", "Introduced cycles:"])
        for cycle in payload["introduced_cycles"]:
            lines.append("- " + " -> ".join(cycle))
    if payload["resolved_cycles"]:
        lines.extend(["", "Resolved cycles:"])
        for cycle in payload["resolved_cycles"]:
            lines.append("- " + " -> ".join(cycle))
    if payload["suggested_checks"]:
        lines.extend(["", "Suggested Tensor Logic checks:"])
        for check in payload["suggested_checks"]:
            lines.append(
                f"- {check['action']} {check['relation']}({check['arg1']}, {check['arg2']}) recursive"
            )
    return "\n".join(lines)


def render_edge_lines(edges: list[list[str]], details: dict) -> list[str]:
    if not edges:
        return ["- (none)"]
    lines = []
    for src, dst in edges:
        src_label = display_module(details.get(src, {"symbol": src, "module": src}))
        dst_label = display_module(details.get(dst, {"symbol": dst, "module": dst}))
        lines.append(f"- {src_label} -> {dst_label}")
    return lines


def render_detail_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["- (none)"]
    lines = []
    for row in rows:
        text = display_module(row)
        if row.get("file"):
            text += f" - {row['file']}"
        lines.append(f"- {text}")
    return lines


def display_module(detail: dict) -> str:
    symbol = detail.get("symbol")
    module = detail.get("module") or symbol
    if symbol and module and symbol != module:
        return f"{symbol} ({module})"
    return module or symbol or "unknown"


def render_ranked_lines(items: list[dict]) -> list[str]:
    if not items:
        return ["- (none)"]
    return [f"- {item['module']}: {item['count']}" for item in items[:8]]


def format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "(none)"


def ingest_python_action(payload: dict) -> tuple[dict, HTTPStatus]:
    path = str(payload.get("path") or "tensor_logic").strip()
    target = resolve_repo_path(path)
    if not target.exists():
        return {"error": f"Path does not exist: {path}", "exit_code": 1}, HTTPStatus.BAD_REQUEST

    start = time.perf_counter()
    try:
        graph = ingest_python(target)
        source = render_python_imports_tl(graph)
    except Exception as exc:
        message = str(exc)
        return {"error": message, "exit_code": 1, "stderr": message}, HTTPStatus.BAD_REQUEST

    symbol_edges = tuple((graph.symbols[src], graph.symbols[dst]) for src, dst in graph.edges)
    symbols = tuple(graph.symbols[module] for module in graph.modules)
    symbol_to_module = {symbol: module for module, symbol in graph.symbols.items()}
    symbol_to_file = {
        graph.symbols[module]: graph.files[module]
        for module in graph.modules
        if module in graph.files
    }
    suggested_query = suggested_dependency_query(symbols, symbol_edges)
    payload_out = {
        "path": str(target),
        "source": source,
        "modules": list(symbols),
        "imports": [list(edge) for edge in symbol_edges],
        "module_names": list(graph.modules),
        "symbols": graph.symbols,
        "symbol_to_module": symbol_to_module,
        "symbol_to_file": symbol_to_file,
        "suggested_query": suggested_query,
    }
    stdout = (
        f"Loaded Python import graph from {target}\n"
        f"modules = {len(symbols)}\n"
        f"imports = {len(symbol_edges)}"
    )
    if suggested_query is not None:
        stdout += (
            "\n"
            f"suggested query = depends_on({suggested_query['arg1']}, {suggested_query['arg2']})\n"
            f"path = {' -> '.join(suggested_query['path'])}"
        )
    return (
        {
            "action": "ingest-python",
            "command": f"ingest-python {path}",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "payload": payload_out,
        },
        HTTPStatus.OK,
    )


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT.parent / candidate
    return candidate.resolve()


def suggested_dependency_query(
    modules: tuple[str, ...],
    imports: tuple[tuple[str, str], ...],
) -> dict | None:
    best_path: list[str] | None = None
    for src in modules:
        for dst in modules:
            if src == dst:
                continue
            path = imports_path(modules, imports, src, dst)
            if path is None:
                continue
            if best_path is None or len(path) > len(best_path):
                best_path = path
    if best_path is None:
        return None
    return {
        "relation": "depends_on",
        "arg1": best_path[0],
        "arg2": best_path[-1],
        "recursive": True,
        "path": best_path,
    }


def render_query_text(payload: dict) -> str:
    relation = payload["relation"]
    arg0, arg1 = payload["args"]
    return f"{relation}({arg0}, {arg1}) = {payload['answer']}\nvalue = {payload['value']:.3f}"


def render_proof_text(payload: dict, relation: str, args: list[str]) -> str:
    proof = payload.get("proof")
    if isinstance(proof, dict):
        return render_proof_node(proof)
    explanation = payload.get("explanation")
    if isinstance(explanation, dict):
        return f"{relation}({', '.join(args)}) = False\n{render_proof_node(explanation)}"
    if proof is not None:
        return str(proof)
    return f"{relation}({', '.join(args)}) = {payload.get('answer', False)}"


def render_proof_node(node: dict, depth: int = 0) -> str:
    if isinstance(node.get("proof"), dict):
        node = node["proof"]
    if isinstance(node.get("explanation"), dict):
        node = node["explanation"]
    indent = "  " * depth
    head = node.get("head", ["unknown", "?", "?"])
    label = f"{head[0]}({head[1]}, {head[2]})"
    suffix = ""
    if "reason" in node:
        suffix = f"  [{node['reason']}]"
    elif source := node.get("source"):
        suffix = f"  [{source['file']}:{source['lineno']}]"
    lines = [f"{indent}{label}{suffix}"]
    for child in node.get("body", []):
        lines.append(render_proof_node(child, depth + 1))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tensor_logic web workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"Serving tensor_logic workbench at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
