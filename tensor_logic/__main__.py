from __future__ import annotations

import argparse
import json
import sys

from .file_format import Command, load_tl
from .http_api import serve
from .ingest import ingest_python, render_python_imports_tl
from .proofs import fmt_proof_tree, fmt_negative_proof_tree, prove, prove_negative, Proof, NegativeProof
from .repo_graph_view import dependency_report, repo_graph_repl


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

    ingest_p = sub.add_parser("ingest-python")
    ingest_p.add_argument("path")
    ingest_p.add_argument("-o", "--output")

    sub.add_parser("repl")

    graph_p = sub.add_parser("repo-graph", help="Dependency graph view for Module/imports/depends_on facts")
    graph_p.add_argument("file", nargs="?", default="/tmp/repo.tl")
    graph_p.add_argument("--module")
    graph_p.add_argument("--src")
    graph_p.add_argument("--dst")
    graph_p.add_argument("--interactive", action="store_true")

    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.cmd == "ingest-python":
        text = render_python_imports_tl(ingest_python(args.path))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
        return 0

    if args.cmd == "repl":
        _run_repl()
        return 0
    if args.cmd == "repo-graph":
        if args.interactive:
            repo_graph_repl(args.file)
            return 0
        if (args.src is None) != (args.dst is None):
            parser.error("--src and --dst must be provided together")
        print(dependency_report(args.file, module=args.module, src=args.src, dst=args.dst))
        return 0
    if args.cmd == "serve":
        serve(args.host, args.port)
        return 0

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
    result = {
        "head": [rel, src, dst],
        "confidence": proof.confidence,
        "body": [_proof_to_json(child) for child in proof.body],
    }
    if proof.source is not None:
        result["source"] = {"file": proof.source.file, "lineno": proof.source.lineno}
    return result

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


def _execute_command(program, command: Command, format_type: str = "tree", why_not: bool = False, out=None) -> None:
    if out is None:
        out = sys.stdout
    if len(command.args) != 2:
        raise ValueError("CLI proof/query currently supports binary relations")
    if command.kind == "query":
        value = program.query(command.relation, *command.args, recursive=command.recursive)
        print(f"{command.relation}({', '.join(command.args)}) = {bool(value)}", file=out)
        return
    proof = prove(program, command.relation, command.args[0], command.args[1], recursive=command.recursive)
    if proof is None:
        if why_not:
            neg_proof = prove_negative(program, command.relation, command.args[0], command.args[1], recursive=command.recursive)
            if neg_proof is not None:
                if format_type == "json":
                    print(json.dumps(_negative_proof_to_json(neg_proof)), file=out)
                else:
                    print(fmt_negative_proof_tree(neg_proof), file=out)
            else:
                print(f"{command.relation}({', '.join(command.args)}) = True", file=out)
        else:
            if format_type == "json":
                print(json.dumps({"answer": False, "proof": None}), file=out)
            else:
                print(f"{command.relation}({', '.join(command.args)}) = False", file=out)
    else:
        if format_type == "json":
            print(json.dumps({"answer": True, "proof": _proof_to_json(proof)}), file=out)
        else:
            print(fmt_proof_tree(proof), file=out)


def _repl_eval(program, line: str, out=None) -> None:
    if out is None:
        out = sys.stdout
    line = line.strip()
    if not line or line.startswith("#"):
        return
    from .file_format import _parse_line
    try:
        command = _parse_line(program, line)
    except ValueError as exc:
        print(f"Error: {exc}", file=out)
        return
    if command is not None:
        _execute_command(program, command, "tree", out=out)


def _run_repl() -> None:
    from .program import Program
    program = Program()
    print("tensor-logic REPL. Type 'exit' or Ctrl-D to quit.")
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    while True:
        try:
            line = input("tl> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() in ("exit", "quit"):
            break
        _repl_eval(program, line)


if __name__ == "__main__":
    raise SystemExit(main())
