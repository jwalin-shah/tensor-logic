from __future__ import annotations

from dataclasses import dataclass
import os
import re

from .program import Program, FactSource


DOMAIN_RE = re.compile(r"^domain\s+(?P<name>\w+)\s*\{(?P<body>[^}]*)\}\s*$")
RELATION_RE = re.compile(r"^relation\s+(?P<name>\w+)\s*\((?P<domains>[^)]*)\)\s*$")
FACT_RE = re.compile(r"^fact\s+(?P<rel>\w+)\s*\((?P<args>[^)]*)\)(?:\s+(?P<value>\d+(?:\.\d+)?))?\s*$")
RULE_RE = re.compile(r"^rule\s+(?P<rule>.+)$")
QUERY_RE = re.compile(r"^(?P<kind>query|prove)\s+(?P<rel>\w+)\s*\((?P<args>[^)]*)\)(?P<flags>.*)$")
INCLUDE_RE = re.compile(r'^include\s+"(?P<path>[^"]+)"\s*$')


@dataclass(frozen=True)
class Command:
    kind: str
    relation: str
    args: tuple[str, ...]
    recursive: bool = False


@dataclass(frozen=True)
class LoadedProgram:
    program: Program
    commands: list[Command]


def load_tl(path: str) -> LoadedProgram:
    program = Program()
    commands: list[Command] = []
    seen: set[str] = {os.path.realpath(path)}
    _load_into(os.path.realpath(path), program, commands, seen)
    return LoadedProgram(program, commands)


def _load_into(path: str, program: Program, commands: list[Command], seen: set[str]) -> None:
    base_dir = os.path.dirname(path)
    for lineno, line in _logical_lines(path):
        if match := INCLUDE_RE.match(line):
            included = os.path.realpath(os.path.join(base_dir, match.group("path")))
            if included in seen:
                raise ValueError(f"{path}:{lineno}: include cycle detected involving {included}")
            seen.add(included)
            _load_into(included, program, commands, seen)
            continue
        try:
            command = _parse_line(program, line, path=path, lineno=lineno)
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if command is not None:
            commands.append(command)


def _parse_line(program: Program, line: str, path: str = "", lineno: int = 0) -> Command | None:
    if match := DOMAIN_RE.match(line):
        program.domain(match.group("name"), _split_items(match.group("body")))
        return None
    if match := RELATION_RE.match(line):
        program.relation(match.group("name"), *_split_items(match.group("domains")))
        return None
    if match := FACT_RE.match(line):
        source = FactSource(path, lineno) if path else None
        value = float(match.group("value")) if match.group("value") else 1.0
        program.fact(match.group("rel"), *_split_items(match.group("args")), value=value, source=source)
        return None
    if match := RULE_RE.match(line):
        program.rule(match.group("rule"))
        return None
    if match := QUERY_RE.match(line):
        flags = set(_split_items(match.group("flags").strip()))
        return Command(
            match.group("kind"),
            match.group("rel"),
            tuple(_split_items(match.group("args"))),
            recursive="recursive" in flags,
        )
    keywords = "domain, relation, fact, rule, query, prove"
    raise ValueError(f"unrecognized statement: {line!r} (expected one of: {keywords})")


def _logical_lines(path: str):
    pending = ""
    start_lineno = 0
    brace_depth = 0
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if not pending:
                start_lineno = lineno
            pending = f"{pending} {line}".strip()
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0:
                yield start_lineno, pending
                pending = ""
    if pending:
        raise ValueError(f"{path}:{start_lineno}: unterminated statement")


def _split_items(text: str) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,\s]+", text) if item.strip()]
