from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from typing import Any

from .file_format import Command, load_tl
from .ingest import ingest_python, render_python_imports_tl
from .proofs import NegativeProof, Proof, fmt_negative_proof_tree, fmt_proof_tree, prove, prove_negative


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int
    message: str


def ingest_python_source(path: str) -> str:
    return render_python_imports_tl(ingest_python(path))


def run_source(source: str) -> dict[str, Any]:
    loaded = _load_program_from_source(source)
    outputs = []
    for command in loaded.commands:
        out = StringIO()
        _execute_command(loaded.program, command, out=out)
        outputs.append(out.getvalue().strip())
    return {"outputs": outputs}


def query_source(source: str, relation: str, args: list[str], recursive: bool = False) -> dict[str, Any]:
    if len(args) != 2:
        raise ApiError(HTTPStatus.BAD_REQUEST, "query requires exactly 2 args")
    loaded = _load_program_from_source(source)
    value = loaded.program.query(relation, *args, recursive=recursive)
    return {"answer": bool(value), "value": float(value), "relation": relation, "args": args, "recursive": recursive}


def prove_source(
    source: str,
    relation: str,
    args: list[str],
    recursive: bool = False,
    why_not: bool = False,
    format_type: str = "tree",
) -> dict[str, Any]:
    if len(args) != 2:
        raise ApiError(HTTPStatus.BAD_REQUEST, "prove requires exactly 2 args")
    if format_type not in {"tree", "json"}:
        raise ApiError(HTTPStatus.BAD_REQUEST, "format must be 'tree' or 'json'")
    loaded = _load_program_from_source(source)
    proof = prove(loaded.program, relation, args[0], args[1], recursive=recursive)
    if proof is None:
        if not why_not:
            return {"answer": False, "proof": None}
        neg_proof = prove_negative(loaded.program, relation, args[0], args[1], recursive=recursive)
        if neg_proof is None:
            return {"answer": True}
        if format_type == "json":
            return _negative_proof_to_json(neg_proof)
        return {"answer": False, "proof": fmt_negative_proof_tree(neg_proof)}
    if format_type == "json":
        return {"answer": True, "proof": _proof_to_json(proof)}
    return {"answer": True, "proof": fmt_proof_tree(proof)}


class TensorLogicHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
            if self.path == "/ingest-python":
                result = {"source": ingest_python_source(str(payload.get("path", "")))}
            elif self.path == "/run":
                result = run_source(str(payload.get("source", "")))
            elif self.path == "/query":
                result = query_source(
                    source=str(payload.get("source", "")),
                    relation=str(payload.get("relation", "")),
                    args=[str(v) for v in payload.get("args", [])],
                    recursive=bool(payload.get("recursive", False)),
                )
            elif self.path == "/prove":
                result = prove_source(
                    source=str(payload.get("source", "")),
                    relation=str(payload.get("relation", "")),
                    args=[str(v) for v in payload.get("args", [])],
                    recursive=bool(payload.get("recursive", False)),
                    why_not=bool(payload.get("why_not", False)),
                    format_type=str(payload.get("format", "tree")),
                )
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, f"unknown endpoint: {self.path}")
            self._write_json(HTTPStatus.OK, result)
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
        except ApiError as exc:
            self._write_json(exc.status_code, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), TensorLogicHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _load_program_from_source(source: str):
    fd, path = tempfile.mkstemp(suffix=".tl", prefix="tensor_logic_api_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        return load_tl(path)
    finally:
        os.unlink(path)


def _execute_command(program, command: Command, format_type: str = "tree", why_not: bool = False, out=None) -> None:
    if out is None:
        out = StringIO()
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
                print(fmt_negative_proof_tree(neg_proof), file=out)
            else:
                print(f"{command.relation}({', '.join(command.args)}) = True", file=out)
        else:
            print(f"{command.relation}({', '.join(command.args)}) = False", file=out)
        return
    if format_type == "json":
        print(json.dumps({"answer": True, "proof": _proof_to_json(proof)}), file=out)
    else:
        print(fmt_proof_tree(proof), file=out)


def _proof_to_json(proof: Proof) -> dict[str, Any]:
    rel, src, dst = proof.head
    source = None
    if proof.source is not None:
        source = {"file": proof.source.file, "lineno": proof.source.lineno}
    result = {
        "head": [rel, src, dst],
        "confidence": proof.confidence,
        "body": [_proof_to_json(child) for child in proof.body],
    }
    if source is not None:
        result["source"] = source
    return result


def _negative_proof_to_json(neg_proof: NegativeProof) -> dict[str, Any]:
    rel, src, dst = neg_proof.head
    return {
        "answer": False,
        "explanation": {
            "head": [rel, src, dst],
            "reason": neg_proof.reason,
            "body": [_negative_proof_to_json(child) for child in neg_proof.body] if neg_proof.body else [],
        },
    }
