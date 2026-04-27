from __future__ import annotations

import argparse
import json

from .file_format import Command, load_tl
from .proofs import fmt_proof_tree, fmt_negative_proof_tree, prove, prove_negative, Proof, NegativeProof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensor-logic")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("file")

    query_p = sub.add_parser("query")
    query_p.add_argument("file")
    query_p.add_argument("relation")
    query_p.add_argument("args", nargs="+")
    query_p.add_argument("--recursive", action="store_true")

    prove_p = sub.add_parser("prove")
    prove_p.add_argument("file")
    prove_p.add_argument("relation")
    prove_p.add_argument("args", nargs="+")
    prove_p.add_argument("--recursive", action="store_true")
    prove_p.add_argument("--format", choices=["tree", "json"], default="tree")
    prove_p.add_argument("--why-not", action="store_true", help="Show explanation of why query is false (if false)")

    args = parser.parse_args(argv)
    loaded = load_tl(args.file)

    if args.cmd == "run":
        for command in loaded.commands:
            _execute_command(loaded.program, command, "tree")
        return 0
    if args.cmd == "query":
        _execute_command(loaded.program, Command("query", args.relation, tuple(args.args), args.recursive), "tree")
        return 0
    if args.cmd == "prove":
        format_type = getattr(args, "format", "tree")
        why_not = getattr(args, "why_not", False)
        _execute_command(loaded.program, Command("prove", args.relation, tuple(args.args), args.recursive), format_type, why_not=why_not)
        return 0
    return 1


def _proof_to_json(proof: Proof) -> dict:
    rel, src, dst = proof.head
    return {
        "head": [rel, src, dst],
        "confidence": proof.confidence,
        "body": [_proof_to_json(child) for child in proof.body],
    }

def _negative_proof_to_json(neg_proof: NegativeProof) -> dict:
    rel, src, dst = neg_proof.head
    return {
        "answer": False,
        "explanation": {
            "head": [rel, src, dst],
            "reason": neg_proof.reason,
            "body": [_negative_proof_to_json(child) for child in neg_proof.body] if neg_proof.body else []
        }
    }


def _execute_command(program, command: Command, format_type: str = "tree", why_not: bool = False) -> None:
    if len(command.args) != 2:
        raise ValueError("CLI proof/query currently supports binary relations")
    if command.kind == "query":
        value = program.query(command.relation, *command.args, recursive=command.recursive)
        print(f"{command.relation}({', '.join(command.args)}) = {bool(value)}")
        return
    proof = prove(program, command.relation, command.args[0], command.args[1], recursive=command.recursive)
    if proof is None:
        if why_not:
            neg_proof = prove_negative(program, command.relation, command.args[0], command.args[1], recursive=command.recursive)
            if neg_proof is not None:
                if format_type == "json":
                    print(json.dumps(_negative_proof_to_json(neg_proof)))
                else:
                    print(fmt_negative_proof_tree(neg_proof))
            else:
                print(f"{command.relation}({', '.join(command.args)}) = True")
        else:
            if format_type == "json":
                print(json.dumps({"answer": False, "proof": None}))
            else:
                print(f"{command.relation}({', '.join(command.args)}) = False")
    else:
        if format_type == "json":
            print(json.dumps({"answer": True, "proof": _proof_to_json(proof)}))
        else:
            print(fmt_proof_tree(proof))


if __name__ == "__main__":
    raise SystemExit(main())
