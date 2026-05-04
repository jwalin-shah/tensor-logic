from __future__ import annotations

import json

from .file_format import Command
from .proof_result import prove_binary_relation_result


def execute_command(program, command: Command, format_type: str = "tree", why_not: bool = False) -> str:
    if len(command.args) != 2:
        raise ValueError("CLI proof/query currently supports binary relations")
    if command.kind == "query":
        value = program.query(command.relation, *command.args, recursive=command.recursive)
        return f"{command.relation}({', '.join(command.args)}) = {bool(value)}"
    if command.kind == "prove":
        data = prove_binary_relation_result(
            program,
            command.relation,
            command.args[0],
            command.args[1],
            recursive=command.recursive,
            why_not=why_not,
            format_type=format_type,
        )
        if format_type == "json":
            return json.dumps(data)
        if data.get("answer") is True and data.get("proof") is None:
            return f"{command.relation}({', '.join(command.args)}) = True"
        if data.get("answer") is False and data.get("proof") is None:
            return f"{command.relation}({', '.join(command.args)}) = False"
        return data["proof"]
    raise ValueError(f"unknown command kind: {command.kind}")
