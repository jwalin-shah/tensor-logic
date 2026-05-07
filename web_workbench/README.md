# tensor_logic web workbench

Browser workbench for editing `.tl` source, running local tensor_logic commands,
and inspecting proof trees.

The repo graph loader turns a Python package into a Tensor Logic dependency
program. Use **Load Imports** with a path such as `tensor_logic` or `.` to
generate `Module`, `imports`, and recursive `depends_on` facts/rules, then run
`Prove` or `Why Not` to inspect dependency paths.

Use **Analyze Impact** on a module symbol to see direct imports, direct
imported-by edges, transitive dependencies, transitive dependents, and example
paths. Ingested repo graphs carry module and source-file metadata through to the
impact JSON and UI tooltips. This is the practical "what might change if I
touch this module?" view.

Use **Repo Overview** to rank hotspots, dependency fanout, entrypoints, leaves,
and import cycles across the loaded graph. Use **Change Brief** on a selected
module to produce a deterministic handoff: read-first files, regression watch,
proof paths, and suggested `depends_on(...)` checks.

Use **Set Baseline** before editing or reloading a graph, then **Compare Graph**
to get added/removed modules and imports, blast-radius deltas, cycle drift, and
clickable proof or why-not checks for changed dependencies.

## Run

```bash
python web_workbench/server.py --host 127.0.0.1 --port 8080
```

Then open <http://127.0.0.1:8080>.

The server writes the current editor contents to a temporary `.tl` file and invokes `python -m tensor_logic`.
