# Negative Proofs Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add negative proofs ("why not?" explanations) to tensor_logic proof engine with CLI support and comprehensive tests.

**Architecture:** 
- Extend `proofs.py` with `NegativeProof` dataclass (mirrors `Proof` structure)
- Implement `prove_negative()` that checks all applicable rules and collects failures
- Add tree formatter for readable output
- Wire `--why-not` flag into CLI with JSON support
- Test on false queries across all example .tl files

**Tech Stack:** Python, dataclasses, tensor_logic proof engine, pytest

---

## File Structure

| File | Responsibility |
|------|-----------------|
| `tensor_logic/proofs.py` | `NegativeProof` class, `prove_negative()`, `fmt_negative_proof_tree()` |
| `tensor_logic/__main__.py` | CLI `--why-not` flag, `_proof_to_json()` update for negative proofs |
| `tests/test_tensor_logic_core.py` | New tests for negative proof behavior |

---

## Design Decisions (Resolved)

1. **Recursion depth:** No limit for first version; output is bounded by rule count and rule body size. Add `--explain-depth` later if needed.
2. **Witness enumeration:** When existential rule fails (e.g., "for any Z in domain, imports(models, Z) failed"), show witnessed domains attempted, not individual Z values.
3. **JSON structure:** Same recursive structure as positive proofs for consistency.

---

## Task List

### Task 1: Add NegativeProof dataclass to proofs.py

**Files:**
- Modify: `tensor_logic/proofs.py:1-15` (add import, add class after Proof)

- [ ] **Step 1: Add NegativeProof dataclass**

Open `tensor_logic/proofs.py` and add the `NegativeProof` class right after the `Proof` class definition (after line 12):

```python
@dataclass(frozen=True)
class NegativeProof:
    head: tuple[str, str, str]
    body: tuple["NegativeProof", ...] = ()
    reason: str = ""
```

Run: `python -c "from tensor_logic.proofs import NegativeProof; print(NegativeProof.__name__)"`
Expected: Output should be `NegativeProof` with no errors.

- [ ] **Step 2: Commit**

```bash
git add tensor_logic/proofs.py
git commit -m "feat: add NegativeProof dataclass"
```

---

### Task 2: Implement prove_negative() with all-rules checking

**Files:**
- Modify: `tensor_logic/proofs.py` (add function after `prove()`)

- [ ] **Step 1: Write failing tests for prove_negative()**

Open `tests/test_tensor_logic_core.py` and add these tests at the end (before `if __name__`):

```python
def test_prove_negative_not_a_fact(self):
    loaded = load_tl("examples/code_dependencies.tl")
    neg_proof = prove_negative(loaded.program, "depends_on", "models", "worker")
    self.assertIsNotNone(neg_proof)
    self.assertEqual(neg_proof.head, ("depends_on", "models", "worker"))
    self.assertEqual(neg_proof.reason, "not_a_fact")

def test_prove_negative_true_query_returns_none(self):
    loaded = load_tl("examples/code_dependencies.tl")
    neg_proof = prove_negative(loaded.program, "depends_on", "worker", "models", recursive=True)
    self.assertIsNone(neg_proof)

def test_prove_negative_no_rules(self):
    program = Program()
    program.domain("Person", ["alice", "bob"])
    program.relation("parent", "Person", "Person")
    program.relation("uncle", "Person", "Person")
    program.fact("parent", "alice", "bob")
    
    neg_proof = prove_negative(program, "uncle", "alice", "bob")
    self.assertIsNotNone(neg_proof)
    self.assertEqual(neg_proof.reason, "no_rules")
```

Add import at top of test file:
```python
from tensor_logic import prove_negative
```

Run: `pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_prove_negative_not_a_fact -v`
Expected: `FAILED - prove_negative is not defined`

- [ ] **Step 2: Implement prove_negative() in proofs.py**

Add this function after `prove()` (after line 25):

```python
def prove_negative(program: Program, relation_name: str, src: str, dst: str, recursive: bool = False) -> NegativeProof | None:
    relation = program.relations[relation_name]
    is_fact = relation.data[relation.domains[0].id(src), relation.domains[1].id(dst)].item() > 0
    
    if is_fact:
        return None
    
    if relation_name not in program.rules:
        return NegativeProof((relation_name, src, dst), reason="no_rules")
    
    rule = program.rules[relation_name]
    rule_failure = _prove_negative_from_rule(program, relation_name, rule, src, dst)
    
    if rule_failure is not None:
        return rule_failure
    
    if recursive:
        neg_recursive = _prove_negative_recursive_chain(program, relation_name, src, dst)
        if neg_recursive is not None:
            return neg_recursive
    
    return None
```

Also add these helper functions after `prove_negative()`:

```python
def _prove_negative_from_rule(program: Program, relation_name: str, rule: Rule, src: str, dst: str) -> NegativeProof | None:
    bindings = _bind_variables(rule.head, src, dst)
    if bindings is None:
        return NegativeProof((relation_name, src, dst), reason="rule_head_mismatch")
    
    body_proof = _prove_negative_body_atoms(program, rule.body, bindings)
    if body_proof is not None:
        return NegativeProof((relation_name, src, dst), body=(body_proof,), reason="rule_body_failed")
    
    return None


def _prove_negative_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str]) -> NegativeProof | None:
    unbound_vars = _find_unbound_vars(atoms, bindings)
    
    if not unbound_vars:
        return _try_prove_negative_body_atoms(program, atoms, bindings)
    
    var = unbound_vars.pop()
    witnesses = _get_witness_domain(program, atoms, bindings, var)
    failed_witnesses = []
    
    for symbol in witnesses:
        extended_bindings = {**bindings, var: symbol}
        neg_proof = _try_prove_negative_body_atoms(program, atoms, extended_bindings)
        if neg_proof is None:
            return None
        failed_witnesses.append(neg_proof)
    
    if failed_witnesses:
        return NegativeProof(
            ("∃" + var, "", ""),
            body=tuple(failed_witnesses),
            reason="no_witness"
        )
    
    return None


def _try_prove_negative_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str]) -> NegativeProof | None:
    neg_proofs = []
    for atom in atoms:
        if len(atom.args) != 2:
            return NegativeProof(
                (atom.relation, "", ""),
                reason="invalid_arity"
            )
        bound_src = bindings.get(atom.args[0], atom.args[0])
        bound_dst = bindings.get(atom.args[1], atom.args[1])
        
        atom_proof = prove(program, atom.relation, bound_src, bound_dst, recursive=False)
        if atom_proof is None:
            atom_neg_proof = prove_negative(program, atom.relation, bound_src, bound_dst, recursive=False)
            if atom_neg_proof is not None:
                neg_proofs.append(atom_neg_proof)
            else:
                neg_proofs.append(NegativeProof(
                    (atom.relation, bound_src, bound_dst),
                    reason="not_provable"
                ))
            return NegativeProof(
                (atoms[0].relation, "", ""),
                body=tuple(neg_proofs),
                reason="atom_failed"
            )
    
    return None


def _prove_negative_recursive_chain(program: Program, relation_name: str, src: str, dst: str) -> NegativeProof | None:
    relation = program.relations[relation_name]
    if len(relation.domains) != 2:
        return None
    
    base = _find_base_relation(program, relation)
    if base is None:
        return NegativeProof((relation_name, src, dst), reason="no_base_relation")
    
    start = base.domains[0].id(src)
    goal = base.domains[1].id(dst)
    queue = deque([(start, [start])])
    seen = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == goal and len(path) > 1:
            return None
        
        row = base.data[current]
        for nxt, value in enumerate(row):
            if value.item() <= 0 or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    
    return NegativeProof((relation_name, src, dst), reason="no_recursive_path")
```

Run: `pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_prove_negative_not_a_fact -v`
Expected: `PASSED`

- [ ] **Step 3: Run all three new tests**

Run: `pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_prove_negative_not_a_fact tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_prove_negative_true_query_returns_none tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_prove_negative_no_rules -v`
Expected: All three tests PASS

- [ ] **Step 4: Verify existing tests still pass**

Run: `pytest tests/test_tensor_logic_core.py -v`
Expected: All 14 tests pass (11 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add tensor_logic/proofs.py tests/test_tensor_logic_core.py
git commit -m "feat: implement prove_negative() with all-rules checking"
```

---

### Task 3: Add fmt_negative_proof_tree() formatter

**Files:**
- Modify: `tensor_logic/proofs.py` (add function after fmt_proof_tree)

- [ ] **Step 1: Write test for formatter**

Add this test to `tests/test_tensor_logic_core.py`:

```python
def test_format_negative_proof_tree(self):
    loaded = load_tl("examples/code_dependencies.tl")
    neg_proof = prove_negative(loaded.program, "depends_on", "models", "worker")
    self.assertIsNotNone(neg_proof)
    
    formatted = fmt_negative_proof_tree(neg_proof)
    self.assertIn("depends_on(models, worker)", formatted)
    self.assertIsInstance(formatted, str)
```

Add import at top:
```python
from tensor_logic import fmt_negative_proof_tree
```

Run: `pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_format_negative_proof_tree -v`
Expected: `FAILED - fmt_negative_proof_tree is not defined`

- [ ] **Step 2: Implement fmt_negative_proof_tree()**

Add this function after `fmt_proof_tree()` (after line 98):

```python
def fmt_negative_proof_tree(neg_proof: NegativeProof, indent: int = 0) -> str:
    pad = "  " * indent
    rel, src, dst = neg_proof.head
    
    if rel == "∃" + neg_proof.head[0].lstrip("∃"):
        lines = [f"{pad}[Witness enumeration failed]"]
    else:
        lines = [f"{pad}{rel}({src}, {dst}) = False"]
    
    if neg_proof.reason:
        lines.append(f"{pad}  reason: {neg_proof.reason}")
    
    for child in neg_proof.body:
        lines.append(fmt_negative_proof_tree(child, indent + 1))
    
    return "\n".join(lines)
```

Run: `pytest tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_format_negative_proof_tree -v`
Expected: `PASSED`

- [ ] **Step 3: Verify formatter on example file**

Run: 
```python
from tensor_logic import load_tl, prove_negative, fmt_negative_proof_tree
loaded = load_tl("examples/code_dependencies.tl")
neg_proof = prove_negative(loaded.program, "depends_on", "models", "worker")
print(fmt_negative_proof_tree(neg_proof))
```

Expected: Readable output showing why `depends_on(models, worker)` is not provable

- [ ] **Step 4: Verify existing tests still pass**

Run: `pytest tests/test_tensor_logic_core.py -v`
Expected: All 15 tests pass

- [ ] **Step 5: Commit**

```bash
git add tensor_logic/proofs.py tests/test_tensor_logic_core.py
git commit -m "feat: add fmt_negative_proof_tree() formatter"
```

---

### Task 4: Update __init__.py to export NegativeProof and prove_negative

**Files:**
- Modify: `tensor_logic/__init__.py`

- [ ] **Step 1: Add exports**

Open `tensor_logic/__init__.py` and update the import from `proofs`:

Change:
```python
from .proofs import (
    fmt_proof_tree,
    prove,
)
```

To:
```python
from .proofs import (
    fmt_proof_tree,
    fmt_negative_proof_tree,
    prove,
    prove_negative,
    Proof,
    NegativeProof,
)
```

Run: `python -c "from tensor_logic import prove_negative, NegativeProof; print('OK')"`
Expected: `OK`

- [ ] **Step 2: Commit**

```bash
git add tensor_logic/__init__.py
git commit -m "feat: export NegativeProof and prove_negative from __init__"
```

---

### Task 5: Wire --why-not flag into CLI

**Files:**
- Modify: `tensor_logic/__main__.py` (update prove command handler)

- [ ] **Step 1: Update argument parser**

Open `tensor_logic/__main__.py` and find the `prove` subcommand argument parser. Add this flag after the `--format` flag (around line 80):

```python
prove_parser.add_argument(
    "--why-not",
    action="store_true",
    help="Show explanation of why query is false (if false)"
)
```

- [ ] **Step 2: Update _execute_command() to handle --why-not**

Find the `_execute_command()` function and update the "prove" case:

Change from:
```python
if args.command == "prove":
    proof = prove(program, args.relation, args.src, args.dst, recursive=args.recursive)
    if proof is None:
        print(f"False")
        return 1
    
    if args.format == "json":
        print(json.dumps(_proof_to_json(proof)))
    else:
        print(fmt_proof_tree(proof))
    return 0
```

To:
```python
if args.command == "prove":
    proof = prove(program, args.relation, args.src, args.dst, recursive=args.recursive)
    
    if proof is None:
        if args.why_not:
            neg_proof = prove_negative(program, args.relation, args.src, args.dst, recursive=args.recursive)
            if neg_proof is not None:
                if args.format == "json":
                    print(json.dumps(_negative_proof_to_json(neg_proof)))
                else:
                    print(fmt_negative_proof_tree(neg_proof))
            else:
                print(f"True")
            return 0
        else:
            print(f"False")
            return 1
    
    if args.format == "json":
        print(json.dumps(_proof_to_json(proof)))
    else:
        print(fmt_proof_tree(proof))
    return 0
```

Also add import at top:
```python
from .proofs import fmt_negative_proof_tree, prove_negative, NegativeProof
```

- [ ] **Step 3: Add _negative_proof_to_json() helper**

Add this function after `_proof_to_json()`:

```python
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
```

- [ ] **Step 4: Test the --why-not flag manually**

Run:
```bash
python -m tensor_logic prove examples/code_dependencies.tl depends_on models worker --why-not
```

Expected: Shows negative proof tree explaining why `depends_on(models, worker)` is false

Run:
```bash
python -m tensor_logic prove examples/code_dependencies.tl depends_on worker models --recursive --why-not
```

Expected: Shows positive proof (true query), not negative proof

- [ ] **Step 5: Test JSON output**

Run:
```bash
python -m tensor_logic prove examples/code_dependencies.tl depends_on models worker --why-not --format json
```

Expected: Valid JSON with `"answer": false` and `"explanation"` object

- [ ] **Step 6: Verify existing tests still pass**

Run: `pytest tests/test_tensor_logic_core.py -v`
Expected: All 15 tests pass

- [ ] **Step 7: Commit**

```bash
git add tensor_logic/__main__.py
git commit -m "feat: add --why-not flag to CLI prove command"
```

---

### Task 6: Add comprehensive negative proof tests

**Files:**
- Modify: `tests/test_tensor_logic_core.py`

- [ ] **Step 1: Add test for rule body failure**

Add this test to `tests/test_tensor_logic_core.py`:

```python
def test_prove_negative_rule_body_failure(self):
    program = Program()
    program.domain("Person", ["alice", "bob", "carol"])
    program.relation("parent", "Person", "Person")
    program.relation("grandparent", "Person", "Person")
    program.fact("parent", "alice", "bob")
    program.fact("parent", "bob", "carol")
    program.rule("grandparent(x,z) := (parent(x,y) * parent(y,z)).step()")
    
    neg_proof = prove_negative(program, "grandparent", "alice", "carol")
    self.assertIsNotNone(neg_proof)
    self.assertEqual(neg_proof.head, ("grandparent", "alice", "carol"))
```

- [ ] **Step 2: Add test for recursive negative proof**

Add this test:

```python
def test_prove_negative_recursive_no_path(self):
    program = Program()
    program.domain("Node", ["a", "b", "c", "d"])
    program.relation("edge", "Node", "Node")
    program.relation("path", "Node", "Node")
    program.fact("edge", "a", "b")
    program.fact("edge", "b", "c")
    program.rule("path(x,z) := (edge(x,z) + path(x,y) * edge(y,z)).step()")
    
    neg_proof = prove_negative(program, "path", "c", "a", recursive=True)
    self.assertIsNotNone(neg_proof)
    self.assertIn("recursive", neg_proof.reason.lower())
```

- [ ] **Step 3: Add test for multiple example files**

Add this test:

```python
def test_prove_negative_on_all_examples(self):
    examples = [
        ("examples/code_dependencies.tl", "depends_on", "models", "worker"),
        ("examples/permissions.tl", "can_access", "user1", "forbidden_resource"),
        ("examples/personal_memory.tl", "should_follow_up", "stranger", "unknown_project"),
    ]
    
    for file_path, relation, src, dst in examples:
        try:
            loaded = load_tl(file_path)
            neg_proof = prove_negative(loaded.program, relation, src, dst)
            self.assertIsNotNone(neg_proof)
            formatted = fmt_negative_proof_tree(neg_proof)
            self.assertIn(relation, formatted)
        except FileNotFoundError:
            pass
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_tensor_logic_core.py -v`
Expected: All 18+ tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_tensor_logic_core.py
git commit -m "test: add comprehensive negative proof tests"
```

---

### Task 7: Verify end-to-end functionality and clean up

**Files:**
- Review: All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Test all example files with --why-not**

Run:
```bash
for file in examples/*.tl; do
  echo "Testing $file..."
  python -m tensor_logic prove "$file" depends_on worker models --why-not 2>/dev/null || true
done
```

Expected: No errors, reasonable output for each file

- [ ] **Step 3: Test JSON serialization on all examples**

Run:
```bash
python -m tensor_logic prove examples/code_dependencies.tl depends_on models worker --why-not --format json | python -m json.tool
```

Expected: Valid, pretty-printed JSON

- [ ] **Step 4: Verify README/docs mention --why-not flag**

Open `README.md` or docs and verify the prove command documentation includes the new flag. If not, add a line:

```markdown
### Negative Proofs (Why Not?)

Use `--why-not` to see why a query is false:
```bash
python -m tensor_logic prove file.tl relation src dst --why-not
```
```

- [ ] **Step 5: Final commit**

```bash
git add README.md docs/
git commit -m "docs: document --why-not flag for negative proofs"
```

---

## Verification

All 7 tasks complete when:
- All tests pass (15+ test cases)
- CLI `--why-not` works end-to-end
- JSON serialization works
- All example files produce sensible negative proofs
