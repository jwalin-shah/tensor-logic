# Tensor Logic Library Survey

Checked on 2026-04-27.

## Packages Found

### `tensorlogic`

- PyPI: https://pypi.org/project/tensorlogic/
- Version inspected: `0.0.4`
- Import name: `tensorlogic`
- Author metadata: Davit Buniatyan
- License: Apache-2.0
- Status classifier: Alpha

This is a lightweight named-index tensor DSL. It supports code like:

```python
from tensorlogic import Domain, Relation

People = Domain(["Alice", "Bob", "Charlie"])
Parent = Relation("Parent", People, People)
Sister = Relation("Sister", People, People)
Aunt = Relation("Aunt", People, People)

Parent["Bob", "Charlie"] = 1
Sister["Alice", "Bob"] = 1
Aunt["x", "z"] = (Sister["x", "y"] * Parent["y", "z"]).step()
```

That example works and returns `Aunt(Alice, Charlie)=1.0`.

Fit for this repo:

- Good fit for named-index syntax and didactic examples.
- Not a drop-in replacement for our closure/provenance experiments.
- No obvious top-k proof-tree provenance.
- No obvious sparse BFS hot path.
- The package name does not conflict with our local `tensor_logic` package because it imports as `tensorlogic`, without the underscore.

### `python-tensorlogic`

- PyPI: https://pypi.org/project/python-tensorlogic/
- Version inspected: `0.3.5`
- Import name: `tensorlogic`
- Repository metadata: https://github.com/Mathews-Tom/TensorLogic
- License: MIT
- Status classifier: Alpha

This is a broader reasoning framework with NumPy/MLX/CUDA backends, fuzzy strategies, quantifiers, RAG integrations, and a `reason(...)` API.

Important caveat: it uses the same import name as `tensorlogic`, so the two packages conflict in one environment.

Observed behavior:

```python
reason(
    "P(x) and Q(x)",
    predicates={"P": [1.0, 0.9, 0.1], "Q": [1.0, 0.8, 0.9]},
    temperature=0.0,
)
```

The README says this should return `[1, 1, 0]`, but the inspected package returned `[1, 1, 1]` with the NumPy backend. That suggests its "hard" semantics are positivity-based in this path, not threshold-0.5 boolean semantics.

Fit for this repo:

- Interesting for fuzzy-logic / temperature API ideas.
- Not reliable enough to replace the local substrate without a deeper audit.
- Import-name conflict with `tensorlogic` makes dependency hygiene awkward.

### `pytensorlogic`

- PyPI: https://pypi.org/project/pytensorlogic/
- Version inspected: `0.1.0a1`
- Import name: `pytensorlogic`
- Repository metadata: https://github.com/cool-japan/tensorlogic
- License: Apache-2.0
- Status classifier: Alpha

This is a Rust-backed package with Python bindings. It advertises expression DSLs, compilation to einsum graphs, backend selection, and provenance metadata.

Local import failed because the wheel expects:

```text
/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib
```

and that library was not installed in the current environment. The bundled examples and stubs show a more compiler-oriented API:

```python
import pytensorlogic as tl

x = tl.var("x")
y = tl.var("y")
knows = tl.pred("knows", [x, y])
graph = tl.compile(knows)
result = tl.execute(graph, {"knows": knows_matrix})
```

Fit for this repo:

- Potentially worth revisiting if we want a Rust-backed compiler layer.
- Not usable as-is in this environment without OpenBLAS.
- Its provenance appears to track source/RDF/tensor metadata, not the proof-tree derivations we need for OpenHuman-style answers.

## Recommendation

Keep the local `tensor_logic/` package as the research substrate.

Use external packages only as references:

- Borrow syntax/API ideas from `tensorlogic` for named-index equations.
- Borrow backend/strategy ideas from `python-tensorlogic` cautiously.
- Revisit `pytensorlogic` only if we need a compiled Rust execution layer.

Do not replace local code yet. The local package is already aligned with the experiment arc: transitive closure, sparse BFS reachability, `<tl_rule>` parsing, stratified negation, proof-tree provenance, and GF(2) controls.
