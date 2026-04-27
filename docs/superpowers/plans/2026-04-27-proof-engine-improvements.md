# Proof Engine Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four improvements to the tensor_logic proof engine: Proof serialization round-trips, an interactive REPL, multi-file `.tl` includes, and tabled recursive proof.

**Architecture:** Each task is independent and targets a specific layer — serialization (proofs.py), CLI (\_\_main\_\_.py), file format (file_format.py), and proof search (proofs.py). Tests live in `tests/test_tensor_logic_core.py`.

**Tech Stack:** Python 3.11+, dataclasses, standard library only (readline, json, pathlib)

---

## File Map

| File | Change |
|---|---|
| `tensor_logic/proofs.py` | Add `Proof.from_json`, `NegativeProof.from_json`, tabled recursive proof |
| `tensor_logic/__main__.py` | Add `repl` subcommand; update `_proof_to_json` to include confidence |
| `tensor_logic/file_format.py` | Add `include "path.tl"` statement parsing |
| `tests/test_tensor_logic_core.py` | New tests for each task |

---

## Task 1: Proof.from_json() / NegativeProof.from_json()

The existing `_proof_to_json` in `__main__.py` serializes proofs but there is no deserializer. We also need to carry `confidence` through the JSON so round-trips are lossless (source locations are omitted intentionally — they reference files, not data).

**Files:**
- Modify: `tensor_logic/proofs.py`
- Modify: `tensor_logic/__main__.py` (add `confidence` to serialized output)
- Modify: `tests/test_tensor_logic_core.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestTensorLogicCore` in `tests/test_tensor_logic_core.py`:

```python
def test_proof_json_roundtrip(self):
    from tensor_logic.proofs import Proof
    original = Proof(
        head=("path", "a", "c"),
        body=(
            Proof(head=("edge", "a", "b"), confidence=0.9),
            Proof(head=("edge", "b", "c"), confidence=0.8),
        ),
        confidence=0.72,
    )
    from tensor_logic.__main__ import _proof_to_json
    d = _proof_to_json(original)
    restored = Proof.from_json(d)
    assert restored.head == original.head
    assert abs(restored.confidence - original.confidence) < 1e-6
    assert len(restored.body) == 2
    assert restored.body[0].head == ("edge", "a", "b")
    assert abs(restored.body[0].confidence - 0.9) < 1e-6

def test_negative_proof_json_roundtrip(self):
    from tensor_logic.proofs import NegativeProof
    original = NegativeProof(
        head=("edge", "a", "z"),
        reason="no_rules",
    )
    from tensor_logic.__main__ import _negative_proof_to_json
    d = _negative_proof_to_json(original)
    restored = NegativeProof.from_json(d)
    assert restored.head == original.head
    assert restored.reason == original.reason
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/jwalinshah/projects/tensor
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_proof_json_roundtrip tests/test_tensor_logic_core.py::TestTensorLogicCore::test_negative_proof_json_roundtrip -v
```

Expected: FAIL with `AttributeError: type object 'Proof' has no attribute 'from_json'`

- [ ] **Step 3: Add `confidence` to `_proof_to_json` in `__main__.py`**

In `tensor_logic/__main__.py`, replace `_proof_to_json`:

```python
def _proof_to_json(proof: Proof) -> dict:
    rel, src, dst = proof.head
    return {
        "head": [rel, src, dst],
        "confidence": proof.confidence,
        "body": [_proof_to_json(child) for child in proof.body],
    }
```

- [ ] **Step 4: Add `Proof.from_json` and `NegativeProof.from_json` to `proofs.py`**

In `tensor_logic/proofs.py`, add these classmethods inside the dataclass definitions. Since dataclasses are frozen, add them after the `@dataclass` block as regular methods. The cleanest approach is to add them as `@classmethod` methods right after the dataclass body.

Replace the `Proof` dataclass definition:

```python
@dataclass(frozen=True)
class Proof:
    head: tuple[str, str, str]
    body: tuple["Proof", ...] = ()
    source: object = None
    confidence: float = 1.0

    @classmethod
    def from_json(cls, d: dict) -> "Proof":
        head = tuple(d["head"])
        body = tuple(cls.from_json(child) for child in d.get("body", []))
        confidence = d.get("confidence", 1.0)
        return cls(head=head, body=body, confidence=confidence)
```

Replace the `NegativeProof` dataclass definition:

```python
@dataclass(frozen=True)
class NegativeProof:
    head: tuple[str, str, str]
    body: tuple["NegativeProof", ...] = ()
    reason: str = ""

    @classmethod
    def from_json(cls, d: dict) -> "NegativeProof":
        explanation = d.get("explanation", d)
        head = tuple(explanation["head"])
        reason = explanation.get("reason", "")
        body = tuple(cls.from_json({"explanation": child["explanation"] if "explanation" in child else child})
                     for child in explanation.get("body", []))
        return cls(head=head, body=body, reason=reason)
```

- [ ] **Step 5: Run tests to verify they pass**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_proof_json_roundtrip tests/test_tensor_logic_core.py::TestTensorLogicCore::test_negative_proof_json_roundtrip -v
```

Expected: PASS

- [ ] **Step 6: Run full suite to check no regressions**

```
python -m pytest tests/ -v
```

Expected: all previously-passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add tensor_logic/proofs.py tensor_logic/__main__.py tests/test_tensor_logic_core.py
git commit -m "feat: add Proof.from_json and NegativeProof.from_json for JSON round-trips"
```

---

## Task 2: Interactive REPL

Add a `repl` subcommand: `python3 -m tensor_logic repl`. The REPL starts with an empty `Program` and accepts `.tl` statements one at a time (`domain`, `relation`, `fact`, `rule`, `prove`, `query`). Results print immediately. `exit` or Ctrl-D quits.

**Files:**
- Modify: `tensor_logic/__main__.py`
- Modify: `tests/test_tensor_logic_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_repl_parse_and_execute(self):
    from tensor_logic.__main__ import _repl_eval
    from tensor_logic.program import Program
    import io

    program = Program()
    out = io.StringIO()
    _repl_eval(program, "domain Node { a, b, c }", out)
    _repl_eval(program, "relation edge(Node, Node)", out)
    _repl_eval(program, "fact edge(a, b)", out)
    _repl_eval(program, "fact edge(b, c)", out)
    _repl_eval(program, "rule path(x,z) := edge(x,y) * edge(y,z)", out)  # wait — rule syntax
    # query
    _repl_eval(program, "query edge(a, b)", out)
    result = out.getvalue()
    assert "True" in result
```

Note: The rule syntax used in `.tl` files is `rule path(x,z) := (edge(x,y) * edge(y,z)).step()`. The REPL should accept the same syntax as `.tl` files.

Revised test:

```python
def test_repl_parse_and_execute(self):
    from tensor_logic.__main__ import _repl_eval
    from tensor_logic.program import Program
    import io

    program = Program()
    out = io.StringIO()
    _repl_eval(program, "domain Node { a, b, c }", out)
    _repl_eval(program, "relation edge(Node, Node)", out)
    _repl_eval(program, "fact edge(a, b)", out)
    _repl_eval(program, "fact edge(b, c)", out)
    _repl_eval(program, "query edge(a, b)", out)
    result = out.getvalue()
    assert "True" in result

def test_repl_prove_command(self):
    from tensor_logic.__main__ import _repl_eval
    from tensor_logic.program import Program
    import io

    program = Program()
    out = io.StringIO()
    _repl_eval(program, "domain P { alice, bob }", out)
    _repl_eval(program, "relation knows(P, P)", out)
    _repl_eval(program, "fact knows(alice, bob)", out)
    _repl_eval(program, "prove knows(alice, bob)", out)
    result = out.getvalue()
    assert "knows(alice, bob)" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_repl_parse_and_execute tests/test_tensor_logic_core.py::TestTensorLogicCore::test_repl_prove_command -v
```

Expected: FAIL with `ImportError: cannot import name '_repl_eval'`

- [ ] **Step 3: Implement `_repl_eval` and `repl` subcommand in `__main__.py`**

Add these functions to `tensor_logic/__main__.py` (before `main()`):

```python
import sys

def _repl_eval(program, line: str, out=None) -> None:
    if out is None:
        out = sys.stdout
    line = line.strip()
    if not line or line.startswith("#"):
        return
    try:
        command = _parse_repl_line(program, line)
    except ValueError as exc:
        print(f"Error: {exc}", file=out)
        return
    if command is not None:
        _execute_command(program, command, "tree", out=out)


def _parse_repl_line(program, line: str):
    from .file_format import _parse_line
    return _parse_line(program, line)


def _run_repl() -> None:
    from .program import Program
    program = Program()
    print("tensor-logic REPL. Type 'exit' or Ctrl-D to quit.")
    try:
        import readline  # noqa: F401 — enables line editing on supported platforms
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
```

Also update `_execute_command` signature to accept an optional `out` parameter so it can write to a StringIO in tests:

```python
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
```

Add `repl` subparser and dispatch in `main()`:

```python
# in main(), after prove_p is defined:
_repl_p = sub.add_parser("repl")

# in main(), after the prove dispatch:
if args.cmd == "repl":
    _run_repl()
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_repl_parse_and_execute tests/test_tensor_logic_core.py::TestTensorLogicCore::test_repl_prove_command -v
```

Expected: PASS

- [ ] **Step 5: Smoke-test the REPL manually**

```
echo -e "domain P { alice, bob }\nrelation knows(P, P)\nfact knows(alice, bob)\nquery knows(alice, bob)\nexit" | python -m tensor_logic repl
```

Expected output includes `knows(alice, bob) = True`

- [ ] **Step 6: Run full suite**

```
python -m pytest tests/ -v
```

- [ ] **Step 7: Commit**

```bash
git add tensor_logic/__main__.py tests/test_tensor_logic_core.py
git commit -m "feat: add interactive REPL (python -m tensor_logic repl)"
```

---

## Task 3: Multi-file `.tl` includes

Add `include "other.tl"` as a valid statement in `.tl` files. Paths are resolved relative to the including file's directory. Cycles (A includes B includes A) raise a clear error.

**Files:**
- Modify: `tensor_logic/file_format.py`
- Modify: `tests/test_tensor_logic_core.py`

- [ ] **Step 1: Write the failing test**

```python
def test_include_directive(self):
    import tempfile, os
    from tensor_logic.file_format import load_tl

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base.tl")
        included_path = os.path.join(tmpdir, "nodes.tl")

        with open(included_path, "w") as f:
            f.write('domain Node { alice, bob }\n')
            f.write('relation knows(Node, Node)\n')
            f.write('fact knows(alice, bob)\n')

        with open(base_path, "w") as f:
            f.write('include "nodes.tl"\n')
            f.write('query knows(alice, bob)\n')

        loaded = load_tl(base_path)
        assert "knows" in loaded.program.relations
        assert len(loaded.commands) == 1
        assert loaded.commands[0].kind == "query"

def test_include_cycle_raises(self):
    import tempfile, os
    from tensor_logic.file_format import load_tl

    with tempfile.TemporaryDirectory() as tmpdir:
        a_path = os.path.join(tmpdir, "a.tl")
        b_path = os.path.join(tmpdir, "b.tl")

        with open(a_path, "w") as f:
            f.write('include "b.tl"\n')
        with open(b_path, "w") as f:
            f.write('include "a.tl"\n')

        with self.assertRaises(ValueError, msg="include cycle"):
            load_tl(a_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_include_directive tests/test_tensor_logic_core.py::TestTensorLogicCore::test_include_cycle_raises -v
```

Expected: FAIL with `ValueError: unrecognized statement: 'include "nodes.tl"'`

- [ ] **Step 3: Add include support to `file_format.py`**

Add at the top of `file_format.py`:

```python
import os
INCLUDE_RE = re.compile(r'^include\s+"(?P<path>[^"]+)"\s*$')
```

Replace `load_tl` to accept an optional `_seen` set for cycle detection:

```python
def load_tl(path: str, _seen: set[str] | None = None) -> LoadedProgram:
    path = os.path.realpath(path)
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ValueError(f"include cycle detected: {path}")
    _seen = _seen | {path}

    program = Program()
    commands: list[Command] = []
    _load_into(path, program, commands, _seen)
    return LoadedProgram(program, commands)


def _load_into(path: str, program: Program, commands: list[Command], seen: set[str]) -> None:
    base_dir = os.path.dirname(path)
    for lineno, line in _logical_lines(path):
        if match := INCLUDE_RE.match(line):
            included = os.path.realpath(os.path.join(base_dir, match.group("path")))
            if included in seen:
                raise ValueError(f"{path}:{lineno}: include cycle detected involving {included}")
            seen = seen | {included}
            _load_into(included, program, commands, seen)
            continue
        try:
            command = _parse_line(program, line, path=path, lineno=lineno)
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if command is not None:
            commands.append(command)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_include_directive tests/test_tensor_logic_core.py::TestTensorLogicCore::test_include_cycle_raises -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```
python -m pytest tests/ -v
```

- [ ] **Step 6: Commit**

```bash
git add tensor_logic/file_format.py tests/test_tensor_logic_core.py
git commit -m "feat: add include directive for multi-file .tl programs"
```

---

## Task 4: Tabled Recursive Proof

Currently `prove()` calls `_prove_recursive_chain` as a BFS fallback that doesn't use rule structure and can loop forever on recursive rules. Replace it with proper SLD resolution with tabling (memoization): mark a goal as `IN_PROGRESS` before recursing, and if we see it again, treat it as "not yet known" (returns `None` for that branch). This is a safe approximation — it finds proofs that don't require the recursive case to bootstrap itself.

The key insight: tabling turns a potentially infinite search into a finite one by memoizing `(relation, src, dst)` → `Proof | None`. A sentinel `_PENDING` marks goals currently being proved. If we see `_PENDING`, we return `None` (no proof via that path without the recursive fact already established). After all rules are tried, the memo is set to the result.

**Files:**
- Modify: `tensor_logic/proofs.py`
- Modify: `tests/test_tensor_logic_core.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_tabled_recursive_proof_direct(self):
    # ancestor(x,y) :- parent(x,y)
    # ancestor(x,y) :- parent(x,z), ancestor(z,y)
    from tensor_logic.program import Program
    from tensor_logic.proofs import prove

    program = Program()
    program.domain("Person", ["alice", "bob", "carol", "dave"])
    program.relation("parent", "Person", "Person")
    program.relation("ancestor", "Person", "Person")
    program.fact("parent", "alice", "bob")
    program.fact("parent", "bob", "carol")
    program.fact("parent", "carol", "dave")
    program.rule("ancestor(x,y) := parent(x,y).step()")
    program.rule("ancestor(x,y) := parent(x,z) * ancestor(z,y).step()")

    # Direct parent — should prove via rule 1
    proof = prove(program, "ancestor", "alice", "bob")
    assert proof is not None
    assert proof.head == ("ancestor", "alice", "bob")

def test_tabled_recursive_proof_deep(self):
    from tensor_logic.program import Program
    from tensor_logic.proofs import prove

    program = Program()
    program.domain("Person", ["alice", "bob", "carol", "dave"])
    program.relation("parent", "Person", "Person")
    program.relation("ancestor", "Person", "Person")
    program.fact("parent", "alice", "bob")
    program.fact("parent", "bob", "carol")
    program.fact("parent", "carol", "dave")
    program.rule("ancestor(x,y) := parent(x,y).step()")
    program.rule("ancestor(x,y) := parent(x,z) * ancestor(z,y).step()")

    # 3 hops — needs recursive rule twice
    proof = prove(program, "ancestor", "alice", "dave")
    assert proof is not None
    assert proof.head == ("ancestor", "alice", "dave")

def test_tabled_recursive_proof_negative(self):
    from tensor_logic.program import Program
    from tensor_logic.proofs import prove

    program = Program()
    program.domain("Person", ["alice", "bob", "carol", "dave"])
    program.relation("parent", "Person", "Person")
    program.relation("ancestor", "Person", "Person")
    program.fact("parent", "alice", "bob")
    program.fact("parent", "bob", "carol")
    program.rule("ancestor(x,y) := parent(x,y).step()")
    program.rule("ancestor(x,y) := parent(x,z) * ancestor(z,y).step()")

    # dave is not an ancestor of alice — should return None without infinite loop
    proof = prove(program, "ancestor", "dave", "alice")
    assert proof is None

def test_tabled_proof_shows_rule_structure(self):
    from tensor_logic.program import Program
    from tensor_logic.proofs import prove, fmt_proof_tree

    program = Program()
    program.domain("Person", ["alice", "bob", "carol"])
    program.relation("parent", "Person", "Person")
    program.relation("ancestor", "Person", "Person")
    program.fact("parent", "alice", "bob")
    program.fact("parent", "bob", "carol")
    program.rule("ancestor(x,y) := parent(x,y).step()")
    program.rule("ancestor(x,y) := parent(x,z) * ancestor(z,y).step()")

    proof = prove(program, "ancestor", "alice", "carol")
    assert proof is not None
    # The proof tree should show the intermediate steps, not just a BFS path
    tree = fmt_proof_tree(proof)
    assert "parent(alice" in tree
    assert "ancestor" in tree
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_recursive_proof_direct tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_recursive_proof_deep tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_recursive_proof_negative tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_proof_shows_rule_structure -v
```

Expected: The direct case may pass (it doesn't need recursion), but deep/negative cases may loop or fail.

- [ ] **Step 3: Add tabling to `proofs.py`**

Add the sentinel and a tabled prove function. The key changes:
1. `prove()` acquires or creates a thread-local table, then delegates to `_prove_tabled()`
2. `_prove_tabled()` checks the table before recursing

In `tensor_logic/proofs.py`, add after the imports:

```python
_PENDING = object()  # sentinel: goal is currently being proved
```

Replace `prove` with a tabled version:

```python
def prove(program: Program, relation_name: str, src: str, dst: str,
          recursive: bool = False, _table: dict | None = None) -> Proof | None:
    if relation_name not in program.relations:
        raise ValueError(f"relation '{relation_name}' not defined")
    relation = program.relations[relation_name]
    _check_symbols(relation, relation_name, src, dst)

    # Check direct fact first
    val = relation.data[relation.domains[0].id(src), relation.domains[1].id(dst)].item()
    if val > 0:
        source = program.sources.get((relation_name, src, dst))
        return Proof((relation_name, src, dst), source=source, confidence=val)

    if relation_name not in program.rules:
        return None

    # Tabled SLD resolution
    if _table is None:
        _table = {}
    key = (relation_name, src, dst)
    if key in _table:
        entry = _table[key]
        if entry is _PENDING:
            return None  # cycle — no proof via this path without the base case
        return entry

    _table[key] = _PENDING
    result = None
    for rule in program.rules[relation_name]:
        proof = _prove_from_rule(program, relation_name, rule, src, dst, _table=_table)
        if proof is not None:
            result = proof
            break
    _table[key] = result
    return result
```

Update `_prove_from_rule` and `_try_prove_body_atoms` to pass `_table` through:

```python
def _prove_from_rule(program: Program, relation_name: str, rule: Rule, src: str, dst: str,
                     _table: dict | None = None) -> Proof | None:
    bindings = _bind_variables(rule.head, src, dst)
    if bindings is None:
        return None
    body_proofs = _prove_body_atoms(program, rule.body, bindings, _table=_table)
    if body_proofs is None:
        return None
    confidence = math.prod(p.confidence for p in body_proofs)
    return Proof((relation_name, src, dst), tuple(body_proofs), confidence=confidence)


def _prove_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str],
                      _table: dict | None = None) -> list[Proof] | None:
    unbound_vars = _find_unbound_vars(atoms, bindings)
    if not unbound_vars:
        return _try_prove_body_atoms(program, atoms, bindings, _table=_table)
    var = unbound_vars.pop()
    for symbol in _get_witness_domain(program, atoms, bindings, var):
        extended_bindings = {**bindings, var: symbol}
        proofs = _try_prove_body_atoms(program, atoms, extended_bindings, _table=_table)
        if proofs is not None:
            return proofs
    return None


def _try_prove_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str],
                          _table: dict | None = None) -> list[Proof] | None:
    proofs = []
    for atom in atoms:
        if len(atom.args) != 2:
            return None
        bound_src = bindings.get(atom.args[0], atom.args[0])
        bound_dst = bindings.get(atom.args[1], atom.args[1])
        atom_proof = prove(program, atom.relation, bound_src, bound_dst, _table=_table)
        if atom_proof is None:
            return None
        proofs.append(atom_proof)
    return proofs
```

Also remove the now-unused `_prove_recursive_chain` function (it's replaced by tabling) and remove the `recursive` parameter usage in the body (the `if recursive:` branch that called it). Keep `recursive` parameter for backwards compatibility but it is now a no-op since tabling handles all cases.

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_recursive_proof_direct tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_recursive_proof_deep tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_recursive_proof_negative tests/test_tensor_logic_core.py::TestTensorLogicCore::test_tabled_proof_shows_rule_structure -v
```

Expected: all 4 PASS

- [ ] **Step 5: Run full suite**

```
python -m pytest tests/ -v
```

Expected: all tests pass. If `prove_negative` tests break, it's because `_try_prove_body_atoms` inside `prove_negative` now needs to call the tabled `prove` too — check that `prove_negative`'s internal calls to `prove()` pass `_table=None` (fresh table per negative query is correct).

- [ ] **Step 6: Commit**

```bash
git add tensor_logic/proofs.py tests/test_tensor_logic_core.py
git commit -m "feat: replace BFS recursive fallback with tabled SLD resolution"
```
