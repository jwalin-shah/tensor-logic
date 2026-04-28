# tensor_logic web workbench

Minimal browser shell for editing `.tl` source and running local tensor_logic commands.

## Run

```bash
python web_workbench/server.py --host 127.0.0.1 --port 8080
```

Then open <http://127.0.0.1:8080>.

The server writes the current editor contents to a temporary `.tl` file and invokes `python -m tensor_logic`.
