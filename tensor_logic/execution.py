from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from io import StringIO
from typing import Any

from .file_format import Command, LoadedProgram, load_tl
from .program import Program
from .proof_result import prove_binary_relation_result

QUERY_BINARY_RELATION_ARGS_ERROR = "query requires exactly 2 args"
PROVE_BINARY_RELATION_ARGS_ERROR = "prove requires exactly 2 args"
COMMAND_BINARY_RELATION_ARGS_ERROR = "CLI proof/query currently supports binary relations"


@dataclass(frozen=True)
class CommandResult:
    payload: dict[str, Any]
    text: str


def load_tl_source(source: str, prefix: str = "tensor_logic_source_") -> LoadedProgram:
    fd, path = tempfile.mkstemp(suffix=".tl", prefix=prefix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        return load_tl(path)
    finally:
        os.unlink(path)


def execute_run(loaded: LoadedProgram, format_type: str = "tree") -> dict[str, Any]:
    outputs = []
    for command in loaded.commands:
        outputs.append(execute_command(loaded.program, command, format_type=format_type).text)
    return {"outputs": outputs}


def execute_query(program: Program, relation: str, args: list[str] | tuple[str, ...], recursive: bool = False) -> dict[str, Any]:
    require_binary_relation_args(args, QUERY_BINARY_RELATION_ARGS_ERROR)
    value = program.query(relation, *args, recursive=recursive)
    return {"answer": bool(value), "value": float(value), "relation": relation, "args": list(args), "recursive": recursive}


def execute_prove(
    program: Program,
    relation: str,
    args: list[str] | tuple[str, ...],
    recursive: bool = False,
    why_not: bool = False,
    format_type: str = "tree",
) -> dict[str, Any]:
    require_binary_relation_args(args, PROVE_BINARY_RELATION_ARGS_ERROR)
    return prove_binary_relation_result(
        program,
        relation,
        args[0],
        args[1],
        recursive=recursive,
        why_not=why_not,
        format_type=format_type,
    )


def execute_command(
    program: Program,
    command: Command,
    format_type: str = "tree",
    why_not: bool = False,
) -> CommandResult:
    require_binary_relation_args(command.args, COMMAND_BINARY_RELATION_ARGS_ERROR)
    if command.kind == "query":
        payload = execute_query(program, command.relation, command.args, recursive=command.recursive)
        text = f"{command.relation}({', '.join(command.args)}) = {payload['answer']}"
        return CommandResult(payload=payload, text=text)
    payload = execute_prove(
        program,
        command.relation,
        command.args,
        recursive=command.recursive,
        why_not=why_not,
        format_type=format_type,
    )
    return CommandResult(payload=payload, text=_format_prove_text(payload, command, format_type))


def write_command_result(result: CommandResult, out) -> None:
    print(result.text, file=out)


def render_command(
    program: Program,
    command: Command,
    format_type: str = "tree",
    why_not: bool = False,
) -> str:
    out = StringIO()
    write_command_result(execute_command(program, command, format_type=format_type, why_not=why_not), out)
    return out.getvalue().strip()


def _format_prove_text(payload: dict[str, Any], command: Command, format_type: str) -> str:
    if format_type == "json":
        return json.dumps(payload)
    proof_text = payload.get("proof")
    if proof_text is not None:
        return str(proof_text)
    return f"{command.relation}({', '.join(command.args)}) = {payload['answer']}"


def require_binary_relation_args(args: list[str] | tuple[str, ...], message: str) -> None:
    if len(args) != 2:
        raise ValueError(message)
