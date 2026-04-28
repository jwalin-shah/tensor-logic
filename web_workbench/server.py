from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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
    source = str(payload.get("source", ""))
    if not source.strip():
        return {"error": "Missing source"}, HTTPStatus.BAD_REQUEST
    relation = str(payload.get("relation", "")).strip()
    arg1 = str(payload.get("arg1", "")).strip()
    arg2 = str(payload.get("arg2", "")).strip()
    recursive = bool(payload.get("recursive", False))

    with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as handle:
        handle.write(source)
        temp_path = Path(handle.name)
    try:
        cmd = [sys.executable, "-m", "tensor_logic"]
        if action == "run":
            cmd += ["run", str(temp_path)]
        elif action in {"query", "prove", "why-not"}:
            if not (relation and arg1 and arg2):
                return {"error": "relation, arg1, and arg2 are required"}, HTTPStatus.BAD_REQUEST
            command = "prove" if action == "why-not" else action
            cmd += [command, str(temp_path), relation, arg1, arg2]
            if recursive:
                cmd.append("--recursive")
            if action == "why-not":
                cmd.append("--why-not")
        else:
            return {"error": f"Unknown action: {action}"}, HTTPStatus.NOT_FOUND
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT.parent), check=False)
    finally:
        temp_path.unlink(missing_ok=True)

    return (
        {
            "action": action,
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
        HTTPStatus.OK if proc.returncode == 0 else HTTPStatus.BAD_REQUEST,
    )


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
