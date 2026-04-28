# Code Index Design

**Date:** 2026-04-28  
**Status:** Approved  
**Scope:** `tensor_logic/` package — API surface for planner injection and interactive lookup

---

## Problem

Planning sessions get burned by stale API knowledge. The planner writes implementation steps using constructor signatures, function args, and return types it remembers from conversation history — not from the live code. When the code has moved (e.g. `Schema.__init__` args changed), the plan is wrong before a line is written.

memjuice answers "what did we decide." Nothing answers "what does the code actually say right now."

---

## Solution

A lightweight code index over `tensor_logic/` that:
1. Extracts current function/class signatures via Python `ast` (no deps)
2. Auto-rebuilds lazily whenever source files are newer than the index
3. Is queried explicitly at plan time via CLI — not injected wholesale into every session

---

## Architecture

```
tools/code_index.py    ← extractor + lazy-rebuild logic + CLI
tools/index.json       ← generated snapshot (gitignored)
CLAUDE.md              ← one instruction: lookup before planning
```

### `tools/code_index.py`

Single script, stdlib only. On every invocation:

1. Check mtime of `tools/index.json` vs all `tensor_logic/*.py` files
2. If any source file is newer (or index missing), rebuild and write `tools/index.json`
3. Execute the requested command (`--rebuild`, `--lookup <symbol>`, `--dump`)

### `tools/index.json`

Gitignored. Schema:

```json
{
  "_meta": {
    "built_at": "<iso timestamp>",
    "note": "AST-extracted; runtime behavior (kwargs, dynamic dispatch) not captured"
  },
  "tensor_logic.rules": {
    "Schema": {
      "kind": "class",
      "init_args": ["name: str", "relations: list[Relation]"],
      "methods": ["add_rule", "validate"]
    },
    "make_rule": {
      "kind": "function",
      "args": ["head: str", "body: list[str]"],
      "returns": "Rule"
    }
  }
}
```

Keys: `<module>.<SymbolName>`. Each entry has `kind` (class/function/constant), args with type annotations where present, return type, and for classes the public method list and `__init__` args. Private methods (leading `_`) excluded.

### `CLAUDE.md` (new, project root)

The tensor repo has no `CLAUDE.md` yet — create one. It gets one planning instruction:

> Before writing any implementation plan that touches `tensor_logic/`, run `python tools/code_index.py --lookup <RelevantSymbol>` to get current signatures.

`AGENTS.md` is memjuice-managed and must not be edited. The new `CLAUDE.md` is the correct home for project-level Claude Code instructions.

No full index injected at session start. Pull what you need at plan time.

---

## CLI Interface

```bash
# Rebuild index unconditionally
python tools/code_index.py --rebuild

# Lookup a symbol (auto-rebuilds if stale)
python tools/code_index.py --lookup Schema
python tools/code_index.py --lookup load_tl

# Dump full index as compact text
python tools/code_index.py --dump

# Check staleness without rebuilding
python tools/code_index.py --status
```

Output of `--lookup Schema`:
```
tensor_logic.rules.Schema [class]
  __init__(name: str, relations: list[Relation])
  methods: add_rule, validate
```

---

## Freshness Model

**Lazy rebuild at read time** — no pre-commit hook required, no daemon, no file watcher.

Every `code_index.py` invocation checks:
```python
index_mtime = os.path.getmtime("tools/index.json")
source_mtime = max(os.path.getmtime(p) for p in glob("tensor_logic/*.py"))
if source_mtime > index_mtime:
    rebuild()
```

If `index.json` is missing (fresh clone), the first lookup triggers a rebuild automatically.

Pre-commit hook is optional bonus — can be added later if the team wants a hard gate.

---

## Cross-Module Lookup (2-hop)

The flat schema doesn't encode type relationships. The planner handles this with two explicit lookups:

```bash
python tools/code_index.py --lookup load_tl
# → returns: LoadedProgram

python tools/code_index.py --lookup LoadedProgram
# → returns: __init__ args, methods
```

No graph traversal needed. Two CLI calls in a plan-writing session is acceptable friction.

---

## Known Limitations

- **Runtime truth gap:** `**kwargs`, dynamic dispatch, and `TypeVar`-typed args not captured. Noted in `_meta.note`.
- **No cross-module relationships:** Flat keying by symbol name. Type chasing requires manual 2-hop lookup.
- **`tensor_logic/` only:** Experiments and tests excluded intentionally — their APIs are throw-away.

---

## What This Is Not

- Not a vector search / embedding index — that's a future concern if queries become fuzzy
- Not a replacement for reading source — for deep understanding, read the file
- Not wired to memjuice — code facts and episodic facts have different freshness models and should not share a store

---

## Success Criteria

1. `python tools/code_index.py --lookup Schema` returns current `__init__` args in <1s
2. Editing `tensor_logic/rules.py` and immediately running `--lookup` returns updated signature (lazy rebuild fires)
3. Fresh clone with no `index.json` works on first lookup without manual setup
4. CLAUDE.md instruction causes planner to run a lookup before drafting steps that touch `tensor_logic/`
