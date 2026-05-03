from __future__ import annotations

import json

from .file_format import Command
from .proof_result import format_proof_result
from .proofs import prove, prove_negative


def execute_command(program, command: Command, format_type: str = "tree", why_not: bool = False) -> str:
    if len(command.args) != 2:
        raise ValueError("CLI proof/query currently supports binary relations")
    if command.kind == "query":
        value = program.query(command.relation, *command.args, recursive=command.recursive)
        return f"{command.relation}({', '.join(command.args)}) = {bool(value)}"
    if command.kind == "prove":
        proof = prove(program, command.relation, command.args[0], command.args[1], recursive=command.recursive)
        if proof is None:
            if why_not:
                neg_proof = prove_negative(program, command.relation, command.args[0], command.args[1], recursive=command.recursive)
                if neg_proof is not None:
                    result = format_proof_result(negative_proof=neg_proof, format_type=format_type)
                    if format_type == "json":
                        return json.dumps(result)
                    return result["proof"]
                return f"{command.relation}({', '.join(command.args)}) = True"
            if format_type == "json":
                return json.dumps({"answer": False, "proof": None})
            return f"{command.relation}({', '.join(command.args)}) = False"
        result = format_proof_result(proof=proof, format_type=format_type)
        if format_type == "json":
            return json.dumps(result)
        return result["proof"]
    raise ValueError(f"unknown command kind: {command.kind}")
