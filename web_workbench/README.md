# tensor_logic web workbench

Minimal shell UI for iterative `.tl` editing and command execution.

## Features

- Left pane: editable source.
- Right pane: command output.
- Toolbar actions: `Run`, `Query`, `Prove`, `Why Not`.
- POSTs to local endpoints (`/api/run`, `/api/query`, `/api/prove`, `/api/why-not`).
- Frontend falls back to a mock response if the backend endpoint is unavailable.

## Run

```bash
python web_workbench/server.py --host 127.0.0.1 --port 8080
```

Then open <http://127.0.0.1:8080>.

The backend wraps the existing CLI (`python -m tensor_logic ...`) by writing current source to a temporary `.tl` file per request.
