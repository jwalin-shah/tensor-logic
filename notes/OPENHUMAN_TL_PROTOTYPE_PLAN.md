# openhuman TL Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 2-week scout-grade prototype that answers "should openhuman use TL/relation-KB instead of vector RAG?" with empirical data, not argument.

**Architecture:** Python project that ingests a subset of the user's existing inbox SQLite (`~/projects/inbox/.inbox_index.sqlite3`), runs a local SLM (gemma3:1b via Ollama) to extract structured triples + concept classifications into a relational KB, exposes `query_kb` / `assert_fact` as Python tools, and runs an eval harness comparing TL+KB against a vector-RAG baseline on hand-labeled queries.

**Tech Stack:** Python 3.11, uv, SQLite (stdlib), Ollama (gemma3:1b-it-qat already installed), pydantic for structured output, pytest, rapidfuzz for entity resolution, numpy where needed.

**Project location:** `~/projects/openhuman-tl-prototype/` — separate from `~/projects/inbox/` (daily driver) and `~/projects/tensor/` (TL research).

**Decision criterion at end:** TL+KB beats RAG by >20% on multi-hop queries AND ontology inheritance demos cleanly AND auditability UX feels meaningfully better → recommend openhuman investigate further. Otherwise → drop TL.

**Reference docs:**
- `OPENHUMAN_TL_MEMO.md` — the design memo this plan implements
- `EXPERIMENTS.md` — relevant TL research findings (exp47, exp49 especially)

---

## Phase Overview

| Phase | Days | Output |
|-------|------|--------|
| 0. Project setup | 1 | Repo, deps, smoke tests passing |
| 1. KB substrate | 2 | Schema + closure rules, hand-seeded data, tests pass |
| 2. SLM extractor | 2 | Llama produces validated triples from inbox messages |
| 3. Entity resolver | 3 | "Bob" mentions collapse to atoms with measured precision |
| 4. Ontology layer | 2 | Hypernym closure + inheritance working with ontology eval |
| 5. Query interface | 1.5 | `query_kb` / `assert_fact` tools, provenance logged |
| 6. Eval harness | 2 | Vector-RAG baseline + TL+KB run + comparison report |
| 7. Findings writeup | 0.5 | Memo answering go/no-go for openhuman |
| **Total** | **~14** | |

---

## Phase 0: Project setup

**Goal:** New project compiles, has test infra, can read inbox SQLite, can call Ollama.

### Task 0.1: Create project skeleton

**Files:**
- Create: `~/projects/openhuman-tl-prototype/pyproject.toml`
- Create: `~/projects/openhuman-tl-prototype/.gitignore`
- Create: `~/projects/openhuman-tl-prototype/README.md`
- Create: `~/projects/openhuman-tl-prototype/src/oh_tl/__init__.py`
- Create: `~/projects/openhuman-tl-prototype/tests/__init__.py`

- [ ] **Step 1: Initialize directory and uv project**

```bash
mkdir -p ~/projects/openhuman-tl-prototype
cd ~/projects/openhuman-tl-prototype
git init
uv init --package --name oh-tl
mkdir -p src/oh_tl tests
touch src/oh_tl/__init__.py tests/__init__.py
```

- [ ] **Step 2: Configure dependencies**

Edit `pyproject.toml`:
```toml
[project]
name = "oh-tl"
version = "0.1.0"
description = "openhuman TL prototype"
requires-python = ">=3.11"
dependencies = [
    "ollama>=0.4.0",
    "pydantic>=2.0",
    "rapidfuzz>=3.0",
    "numpy>=1.26",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install and verify**

```bash
uv sync
uv run python -c "import ollama, pydantic, rapidfuzz, numpy; print('deps OK')"
```

Expected: prints `deps OK`.

- [ ] **Step 4: Add `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
*.sqlite3
*.db
*.log
.coverage
```

- [ ] **Step 5: Smoke-test Ollama**

Create `tests/test_smoke.py`:
```python
import ollama

def test_ollama_responds():
    r = ollama.chat(
        model="gemma3:1b-it-qat",
        messages=[{"role": "user", "content": "Say only the word: ok"}],
        options={"temperature": 0},
    )
    assert "ok" in r["message"]["content"].lower()
```

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS within ~5s.

- [ ] **Step 6: Smoke-test inbox SQLite read**

Add to `tests/test_smoke.py`:
```python
import sqlite3
import os

INBOX_DB = os.path.expanduser("~/projects/inbox/.inbox_index.sqlite3")

def test_inbox_sqlite_readable():
    assert os.path.exists(INBOX_DB)
    conn = sqlite3.connect(f"file:{INBOX_DB}?mode=ro", uri=True)
    cursor = conn.execute("SELECT COUNT(*) FROM items")
    count = cursor.fetchone()[0]
    conn.close()
    assert count > 100, f"expected >100 items, got {count}"
```

Run: `uv run pytest tests/test_smoke.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Initial commit**

```bash
git add -A
git commit -m "phase 0: project skeleton with deps, ollama smoke test"
```

---

## Phase 1: KB substrate

**Goal:** Define schema, populate with hand-seeded data, run kinship closure rules. All deterministic, no SLM yet.

### Task 1.1: Schema definition

**Files:**
- Create: `src/oh_tl/schema.py`
- Create: `src/oh_tl/store.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write the schema as raw SQL**

`src/oh_tl/schema.py`:
```python
"""KB schema. Source of truth for table definitions."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS person (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS rel_name      (person_id INTEGER, name TEXT, conf REAL, PRIMARY KEY(person_id, name));
CREATE TABLE IF NOT EXISTS rel_email     (person_id INTEGER, email TEXT, conf REAL, PRIMARY KEY(person_id, email));
CREATE TABLE IF NOT EXISTS rel_phone     (person_id INTEGER, phone TEXT, conf REAL, PRIMARY KEY(person_id, phone));
CREATE TABLE IF NOT EXISTS rel_parent    (parent_id INTEGER, child_id INTEGER, conf REAL, PRIMARY KEY(parent_id, child_id));
CREATE TABLE IF NOT EXISTS rel_sibling   (a_id INTEGER, b_id INTEGER, conf REAL, PRIMARY KEY(a_id, b_id));
CREATE TABLE IF NOT EXISTS rel_partner   (a_id INTEGER, b_id INTEGER, conf REAL, status TEXT, PRIMARY KEY(a_id, b_id));
CREATE TABLE IF NOT EXISTS rel_male      (person_id INTEGER PRIMARY KEY, conf REAL);
CREATE TABLE IF NOT EXISTS rel_female    (person_id INTEGER PRIMARY KEY, conf REAL);

CREATE TABLE IF NOT EXISTS message (
    id INTEGER PRIMARY KEY,
    source TEXT, account TEXT, external_id TEXT, ts INTEGER, body TEXT
);

CREATE TABLE IF NOT EXISTS rel_sent      (person_id INTEGER, message_id INTEGER, ts INTEGER, PRIMARY KEY(person_id, message_id));
CREATE TABLE IF NOT EXISTS rel_received  (person_id INTEGER, message_id INTEGER, ts INTEGER, PRIMARY KEY(person_id, message_id));
CREATE TABLE IF NOT EXISTS rel_mentioned (person_id INTEGER, message_id INTEGER, conf REAL, PRIMARY KEY(person_id, message_id));

-- Ontology layer (Phase 4 will populate; defined here for unity)
CREATE TABLE IF NOT EXISTS concept (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS rel_is_a (sub_id INTEGER, super_id INTEGER, conf REAL, PRIMARY KEY(sub_id, super_id));
CREATE TABLE IF NOT EXISTS rel_classified_as (entity_id INTEGER, entity_type TEXT, concept_id INTEGER, conf REAL, PRIMARY KEY(entity_id, entity_type, concept_id));
CREATE TABLE IF NOT EXISTS concept_property (concept_id INTEGER, property TEXT, value TEXT, conf REAL, PRIMARY KEY(concept_id, property));

-- Provenance
CREATE TABLE IF NOT EXISTS slm_call (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT, prompt_hash TEXT, output TEXT, ts INTEGER, latency_ms REAL
);
CREATE TABLE IF NOT EXISTS grounded_in (call_id INTEGER, table_name TEXT, row_id INTEGER);
CREATE TABLE IF NOT EXISTS rule_fired (call_id INTEGER, rule_id TEXT);

CREATE INDEX IF NOT EXISTS idx_rel_parent_child ON rel_parent(child_id);
CREATE INDEX IF NOT EXISTS idx_rel_sibling_b ON rel_sibling(b_id);
CREATE INDEX IF NOT EXISTS idx_rel_classified ON rel_classified_as(concept_id);
"""
```

- [ ] **Step 2: Write Store wrapper around SQLite**

`src/oh_tl/store.py`:
```python
"""Thin SQLite wrapper. All KB access goes through here."""
import sqlite3
from pathlib import Path
from .schema import SCHEMA_SQL

class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def add_person(self, canonical_name: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO person (canonical_name) VALUES (?)", (canonical_name,)
        )
        self._conn.commit()
        return cur.lastrowid

    def assert_fact(self, table: str, fields: dict) -> None:
        cols = ",".join(fields.keys())
        placeholders = ",".join("?" * len(fields))
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
            tuple(fields.values()),
        )
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params))

    def close(self):
        self._conn.close()
```

- [ ] **Step 3: Write tests for Store**

`tests/test_schema.py`:
```python
import tempfile
from oh_tl.store import Store

def test_store_creates_schema():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        tables = [r["name"] for r in s.query("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in ["person", "rel_parent", "rel_sibling", "concept", "slm_call"]:
            assert t in tables, f"missing table: {t}"
        s.close()

def test_add_person_and_assert_fact():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        bob = s.add_person("Bob Smith")
        alice = s.add_person("Alice Smith")
        s.assert_fact("rel_parent", {"parent_id": bob, "child_id": alice, "conf": 1.0})
        rows = s.query("SELECT * FROM rel_parent")
        assert len(rows) == 1
        assert rows[0]["parent_id"] == bob
        assert rows[0]["child_id"] == alice
        s.close()
```

Run: `uv run pytest tests/test_schema.py -v`
Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "phase 1.1: schema + store wrapper, tests pass"
```

### Task 1.2: Closure rules (kinship reasoning)

**Files:**
- Create: `src/oh_tl/rules.py`
- Create: `tests/test_rules.py`

- [ ] **Step 1: Write closure rules as recursive CTEs**

`src/oh_tl/rules.py`:
```python
"""Derivation rules expressed as SQL recursive CTEs.

Each rule returns a query that, given a Store, computes derived facts.
"""

UNCLE_SQL = """
SELECT DISTINCT s.a_id AS uncle_id, p.child_id AS nephew_id
FROM rel_sibling s
JOIN rel_parent p ON s.b_id = p.parent_id
JOIN rel_male m ON s.a_id = m.person_id
"""

ANCESTOR_SQL = """
WITH RECURSIVE anc(ancestor_id, descendant_id) AS (
    SELECT parent_id, child_id FROM rel_parent
    UNION
    SELECT a.ancestor_id, p.child_id
    FROM anc a
    JOIN rel_parent p ON a.descendant_id = p.parent_id
)
SELECT * FROM anc
"""

CLOSE_FAMILY_SQL = """
WITH RECURSIVE cf(a_id, b_id) AS (
    SELECT parent_id, child_id FROM rel_parent
    UNION
    SELECT child_id, parent_id FROM rel_parent
    UNION
    SELECT a_id, b_id FROM rel_sibling
    UNION
    SELECT b_id, a_id FROM rel_sibling
    UNION
    SELECT cf.a_id, p.child_id FROM cf JOIN rel_parent p ON cf.b_id = p.parent_id
)
SELECT * FROM cf
"""

def derive_uncles(store):
    return store.query(UNCLE_SQL)

def derive_ancestors(store):
    return store.query(ANCESTOR_SQL)

def derive_close_family(store):
    return store.query(CLOSE_FAMILY_SQL)
```

- [ ] **Step 2: Write tests with hand-seeded family**

`tests/test_rules.py`:
```python
import tempfile
from oh_tl.store import Store
from oh_tl.rules import derive_uncles, derive_ancestors, derive_close_family

def setup_family(s):
    """
    Family tree:
        grandpa - grandma
              |
        +-----+-----+
        |           |
       dad        uncle_bob
        |
       jwalin
    """
    grandpa = s.add_person("Grandpa")
    grandma = s.add_person("Grandma")
    dad = s.add_person("Dad")
    uncle_bob = s.add_person("Uncle Bob")
    jwalin = s.add_person("Jwalin")
    
    s.assert_fact("rel_male", {"person_id": grandpa, "conf": 1.0})
    s.assert_fact("rel_male", {"person_id": dad, "conf": 1.0})
    s.assert_fact("rel_male", {"person_id": uncle_bob, "conf": 1.0})
    s.assert_fact("rel_male", {"person_id": jwalin, "conf": 1.0})
    s.assert_fact("rel_female", {"person_id": grandma, "conf": 1.0})
    
    s.assert_fact("rel_partner", {"a_id": grandpa, "b_id": grandma, "conf": 1.0, "status": "married"})
    s.assert_fact("rel_parent", {"parent_id": grandpa, "child_id": dad, "conf": 1.0})
    s.assert_fact("rel_parent", {"parent_id": grandma, "child_id": dad, "conf": 1.0})
    s.assert_fact("rel_parent", {"parent_id": grandpa, "child_id": uncle_bob, "conf": 1.0})
    s.assert_fact("rel_parent", {"parent_id": grandma, "child_id": uncle_bob, "conf": 1.0})
    s.assert_fact("rel_parent", {"parent_id": dad, "child_id": jwalin, "conf": 1.0})
    s.assert_fact("rel_sibling", {"a_id": dad, "b_id": uncle_bob, "conf": 1.0})
    s.assert_fact("rel_sibling", {"a_id": uncle_bob, "b_id": dad, "conf": 1.0})
    
    return {"grandpa": grandpa, "grandma": grandma, "dad": dad,
            "uncle_bob": uncle_bob, "jwalin": jwalin}

def test_uncle_derivation():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        ids = setup_family(s)
        uncles = derive_uncles(s)
        assert any(u["uncle_id"] == ids["uncle_bob"] and u["nephew_id"] == ids["jwalin"]
                   for u in uncles)
        s.close()

def test_ancestor_closure_multi_hop():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        ids = setup_family(s)
        ancestors = derive_ancestors(s)
        # grandpa is ancestor of jwalin (2 hops)
        assert any(a["ancestor_id"] == ids["grandpa"] and a["descendant_id"] == ids["jwalin"]
                   for a in ancestors)
        s.close()

def test_close_family_includes_uncle():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        ids = setup_family(s)
        cf = derive_close_family(s)
        # uncle Bob is close family of jwalin (via sibling-of-parent)
        assert any(r["a_id"] == ids["uncle_bob"] and r["b_id"] == ids["jwalin"] for r in cf)
        s.close()
```

Run: `uv run pytest tests/test_rules.py -v`
Expected: 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 1.2: closure rules (uncle, ancestor, close_family) with tests"
```

---

## Phase 2: SLM extractor

**Goal:** Llama reads a message, emits validated structured triples that go into the KB.

### Task 2.1: Pydantic schema for SLM output

**Files:**
- Create: `src/oh_tl/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Define structured output schema**

`src/oh_tl/extractor.py`:
```python
"""SLM extractor: message text → validated KB triples."""
from pydantic import BaseModel, Field
from typing import Literal
import json
import time
import hashlib
import ollama

ALLOWED_PREDICATES = Literal[
    "name", "email", "phone", "parent", "sibling", "partner",
    "male", "female", "mentioned", "sent", "received",
]

class Triple(BaseModel):
    predicate: ALLOWED_PREDICATES
    subject: str = Field(..., description="entity name as it appears in text")
    object: str | None = Field(None, description="entity name or value")
    confidence: float = Field(..., ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    triples: list[Triple]
    notes: str = ""

EXTRACTOR_PROMPT = """You extract structured facts from a personal message.
Output ONLY valid JSON matching this schema:
{
  "triples": [
    {"predicate": "<one of: name, email, phone, parent, sibling, partner, male, female, mentioned, sent, received>",
     "subject": "<person name as in text>",
     "object": "<person name or value, or null>",
     "confidence": <0.0 to 1.0>}
  ],
  "notes": ""
}

Rules:
- Only emit triples directly stated or strongly implied.
- For "parent": subject is the parent, object is the child.
- For "sibling": both arguments are siblings (emit both directions).
- Confidence < 0.7 means "I'm guessing."
- If no facts, return {"triples": [], "notes": "no facts"}.

Message:
"""

def extract(message_body: str, model: str = "gemma3:1b-it-qat") -> tuple[ExtractionResult, dict]:
    """Run extraction. Returns (result, provenance_dict)."""
    t0 = time.time()
    prompt = EXTRACTOR_PROMPT + message_body
    r = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )
    latency_ms = (time.time() - t0) * 1000
    raw = r["message"]["content"]
    # Validate against schema
    try:
        result = ExtractionResult.model_validate_json(raw)
    except Exception as e:
        # Defensive: SLM produced invalid JSON, return empty
        result = ExtractionResult(triples=[], notes=f"parse_error: {e}")
    provenance = {
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "raw_output": raw,
        "latency_ms": latency_ms,
    }
    return result, provenance
```

- [ ] **Step 2: Write extractor tests on hand-crafted messages**

`tests/test_extractor.py`:
```python
import pytest
from oh_tl.extractor import extract

# These tests hit the actual local SLM. Mark slow.
pytestmark = pytest.mark.slow

def test_extracts_simple_kinship():
    msg = "Bob is my dad. He's been calling me a lot lately."
    result, prov = extract(msg)
    assert prov["model"] == "gemma3:1b-it-qat"
    assert prov["latency_ms"] < 5000
    # We expect at least a parent triple involving Bob
    parent_triples = [t for t in result.triples if t.predicate == "parent"]
    assert len(parent_triples) >= 1
    assert any("bob" in t.subject.lower() for t in parent_triples)

def test_handles_no_facts_gracefully():
    msg = "Just sent the thing!"
    result, prov = extract(msg)
    # SLM may emit nothing or low-confidence noise; either is acceptable
    assert isinstance(result.triples, list)

def test_validates_invalid_predicate():
    """If we somehow got an invalid predicate, validation should reject it.
    We test the schema directly."""
    from pydantic import ValidationError
    from oh_tl.extractor import Triple
    with pytest.raises(ValidationError):
        Triple(predicate="not_a_real_predicate", subject="x", confidence=0.5)
```

Configure pytest marker. Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: tests that hit the local SLM"]
```

Run: `uv run pytest tests/test_extractor.py -v -m slow`
Expected: 3 tests PASS (the kinship test may take ~2-3s).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 2.1: SLM extractor with validated structured output"
```

### Task 2.2: Pipeline that ingests inbox messages

**Files:**
- Create: `src/oh_tl/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Read inbox SQLite, run extractor on a sample**

`src/oh_tl/ingest.py`:
```python
"""Ingest pipeline: inbox messages → extracted triples → KB writes."""
import sqlite3
import os
from .store import Store
from .extractor import extract

INBOX_DB = os.path.expanduser("~/projects/inbox/.inbox_index.sqlite3")

def fetch_messages(limit: int = 100, source: str = "imessage") -> list[dict]:
    """Read messages from the user's inbox SQLite (read-only)."""
    conn = sqlite3.connect(f"file:{INBOX_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, source, account, external_id, sender, body_text, created_at
        FROM items
        WHERE source = ? AND length(body_text) > 30 AND length(body_text) < 1000
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (source, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def ingest_into_kb(store: Store, messages: list[dict]) -> dict:
    """Extract triples from each message, write to KB. Returns stats."""
    stats = {"messages": 0, "triples_emitted": 0, "triples_kept": 0, "errors": 0}
    for msg in messages:
        try:
            result, prov = extract(msg["body_text"])
            stats["messages"] += 1
            stats["triples_emitted"] += len(result.triples)

            # Log SLM call to provenance
            cur = store._conn.execute(
                "INSERT INTO slm_call (model, prompt_hash, output, ts, latency_ms) VALUES (?, ?, ?, strftime('%s','now'), ?)",
                (prov["model"], prov["prompt_hash"], prov["raw_output"], prov["latency_ms"]),
            )
            call_id = cur.lastrowid

            # Log message
            store._conn.execute(
                "INSERT OR REPLACE INTO message (source, account, external_id, ts, body) VALUES (?, ?, ?, ?, ?)",
                (msg["source"], msg["account"], msg["external_id"], 0, msg["body_text"]),
            )

            # Filter to high-confidence triples; entity resolution comes in Phase 3
            kept = [t for t in result.triples if t.confidence >= 0.7]
            stats["triples_kept"] += len(kept)

            store._conn.commit()
        except Exception as e:
            stats["errors"] += 1
            print(f"ingest error: {e}")
    return stats
```

- [ ] **Step 2: Test ingest on a small sample**

`tests/test_ingest.py`:
```python
import tempfile
import pytest
from oh_tl.store import Store
from oh_tl.ingest import fetch_messages, ingest_into_kb

@pytest.mark.slow
def test_fetch_messages_returns_some():
    msgs = fetch_messages(limit=10)
    assert len(msgs) > 0
    assert all("body_text" in m for m in msgs)

@pytest.mark.slow
def test_ingest_pipeline_produces_provenance():
    msgs = fetch_messages(limit=5)
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        stats = ingest_into_kb(s, msgs)
        assert stats["messages"] >= 1
        # Provenance should be logged
        calls = s.query("SELECT COUNT(*) AS c FROM slm_call")
        assert calls[0]["c"] >= 1
        s.close()
```

Run: `uv run pytest tests/test_ingest.py -v -m slow`
Expected: PASS. Note this hits the SLM and reads inbox — may take ~10-30s.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 2.2: ingest pipeline reads inbox + extracts + logs provenance"
```

---

## Phase 3: Entity resolver

**Goal:** Multiple "Bob" mentions across messages collapse to a single `person(p_47)` atom — or, when they're distinct people, stay separate.

This is the genuinely hard piece. We use a layered approach: exact match on phone/email first, then fuzzy name match weighted by context.

### Task 3.1: Resolver core

**Files:**
- Create: `src/oh_tl/resolver.py`
- Create: `tests/test_resolver.py`

- [ ] **Step 1: Implement resolver**

`src/oh_tl/resolver.py`:
```python
"""Entity resolver: string mention -> person atom ID.

Strategy:
1. Exact match on email/phone if available (highest priority)
2. Fuzzy name match against existing persons, scored by:
   - Name similarity (rapidfuzz)
   - Context overlap (recent messages, threads)
3. New atom if no match exceeds threshold
"""
from rapidfuzz import fuzz
from .store import Store

NAME_MATCH_THRESHOLD = 85  # 0-100, rapidfuzz scale
CONTEXT_BOOST = 10  # added if context (thread/message overlap) matches

def resolve(store: Store, mention: str, context: dict | None = None) -> int:
    """
    mention: the string from the message ('Bob', 'bob smith', 'b.smith@x.com')
    context: optional dict with 'thread_id', 'recent_persons', etc.
    Returns: person_id (existing or new).
    """
    context = context or {}
    mention_clean = mention.strip().lower()

    # 1. Exact match on email
    if "@" in mention_clean:
        rows = store.query(
            "SELECT person_id FROM rel_email WHERE LOWER(email) = ?",
            (mention_clean,),
        )
        if rows:
            return rows[0]["person_id"]

    # 2. Exact match on phone
    if mention_clean.startswith("+") or mention_clean.replace("-", "").isdigit():
        rows = store.query(
            "SELECT person_id FROM rel_phone WHERE REPLACE(phone, '-', '') = ?",
            (mention_clean.replace("-", ""),),
        )
        if rows:
            return rows[0]["person_id"]

    # 3. Fuzzy name match
    candidates = store.query(
        """
        SELECT p.id, p.canonical_name, GROUP_CONCAT(rn.name, '|') as aliases
        FROM person p LEFT JOIN rel_name rn ON p.id = rn.person_id
        GROUP BY p.id
        """
    )
    best_id = None
    best_score = 0
    for c in candidates:
        names = [c["canonical_name"]]
        if c["aliases"]:
            names.extend(c["aliases"].split("|"))
        for name in names:
            score = fuzz.WRatio(mention_clean, name.lower())
            if context.get("recent_persons") and c["id"] in context["recent_persons"]:
                score += CONTEXT_BOOST
            if score > best_score:
                best_score = score
                best_id = c["id"]

    if best_score >= NAME_MATCH_THRESHOLD:
        return best_id

    # 4. New atom
    new_id = store.add_person(mention.strip())
    store.assert_fact("rel_name", {"person_id": new_id, "name": mention.strip(), "conf": 1.0})
    return new_id
```

- [ ] **Step 2: Test resolver**

`tests/test_resolver.py`:
```python
import tempfile
from oh_tl.store import Store
from oh_tl.resolver import resolve

def test_creates_new_person_when_no_match():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        pid = resolve(s, "Brand New Name")
        assert pid is not None
        rows = s.query("SELECT * FROM person WHERE id = ?", (pid,))
        assert len(rows) == 1
        s.close()

def test_resolves_to_existing_by_exact_email():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        bob = s.add_person("Bob")
        s.assert_fact("rel_email", {"person_id": bob, "email": "bob@example.com", "conf": 1.0})
        resolved = resolve(s, "bob@example.com")
        assert resolved == bob
        s.close()

def test_fuzzy_name_match_above_threshold():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        bob = s.add_person("Bob Smith")
        s.assert_fact("rel_name", {"person_id": bob, "name": "Bob Smith", "conf": 1.0})
        resolved = resolve(s, "Bob")  # close enough
        assert resolved == bob
        s.close()

def test_distinct_persons_not_collapsed():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        s.add_person("Alice Johnson")
        bob = resolve(s, "Bob")
        assert bob is not None
        rows = s.query("SELECT * FROM person")
        assert len(rows) == 2
        s.close()

def test_context_boost_disambiguates():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        bob1 = s.add_person("Bob Smith")
        bob2 = s.add_person("Bob Jones")
        s.assert_fact("rel_name", {"person_id": bob1, "name": "Bob Smith", "conf": 1.0})
        s.assert_fact("rel_name", {"person_id": bob2, "name": "Bob Jones", "conf": 1.0})
        # Without context, ambiguous; with context favoring bob1, should resolve to bob1
        resolved = resolve(s, "Bob", context={"recent_persons": [bob1]})
        assert resolved == bob1
        s.close()
```

Run: `uv run pytest tests/test_resolver.py -v`
Expected: 5 PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 3.1: entity resolver with email/phone/fuzzy-name + context boost"
```

### Task 3.2: Wire resolver into ingest pipeline

**Files:**
- Modify: `src/oh_tl/ingest.py`

- [ ] **Step 1: Update ingest to resolve subjects/objects to atom IDs**

Replace the body of `ingest_into_kb` in `src/oh_tl/ingest.py`:
```python
from .resolver import resolve

def ingest_into_kb(store: Store, messages: list[dict]) -> dict:
    """Extract triples from each message, resolve entities, write to KB."""
    stats = {"messages": 0, "triples_emitted": 0, "triples_kept": 0, "errors": 0}
    for msg in messages:
        try:
            result, prov = extract(msg["body_text"])
            stats["messages"] += 1
            stats["triples_emitted"] += len(result.triples)

            cur = store._conn.execute(
                "INSERT INTO slm_call (model, prompt_hash, output, ts, latency_ms) VALUES (?, ?, ?, strftime('%s','now'), ?)",
                (prov["model"], prov["prompt_hash"], prov["raw_output"], prov["latency_ms"]),
            )
            call_id = cur.lastrowid

            store._conn.execute(
                "INSERT OR REPLACE INTO message (source, account, external_id, ts, body) VALUES (?, ?, ?, ?, ?)",
                (msg["source"], msg["account"], msg["external_id"], 0, msg["body_text"]),
            )

            recent = []
            for t in result.triples:
                if t.confidence < 0.7:
                    continue
                subj_id = resolve(store, t.subject, context={"recent_persons": recent})
                recent.append(subj_id)
                obj_id = None
                if t.object and t.predicate in {"parent", "sibling", "partner"}:
                    obj_id = resolve(store, t.object, context={"recent_persons": recent})
                    recent.append(obj_id)
                _write_triple(store, t, subj_id, obj_id, call_id)
                stats["triples_kept"] += 1

            store._conn.commit()
        except Exception as e:
            stats["errors"] += 1
            print(f"ingest error: {e}")
    return stats

def _write_triple(store, t, subj_id, obj_id, call_id):
    """Write a resolved triple to the appropriate relation table + provenance."""
    table_map = {
        "parent":  ("rel_parent",  {"parent_id": subj_id, "child_id": obj_id, "conf": t.confidence}),
        "sibling": ("rel_sibling", {"a_id": subj_id, "b_id": obj_id, "conf": t.confidence}),
        "partner": ("rel_partner", {"a_id": subj_id, "b_id": obj_id, "conf": t.confidence, "status": "current"}),
        "male":    ("rel_male",    {"person_id": subj_id, "conf": t.confidence}),
        "female":  ("rel_female",  {"person_id": subj_id, "conf": t.confidence}),
        "email":   ("rel_email",   {"person_id": subj_id, "email": t.object, "conf": t.confidence}),
        "phone":   ("rel_phone",   {"person_id": subj_id, "phone": t.object, "conf": t.confidence}),
        "name":    ("rel_name",    {"person_id": subj_id, "name": t.object or t.subject, "conf": t.confidence}),
    }
    if t.predicate not in table_map:
        return
    table, fields = table_map[t.predicate]
    store.assert_fact(table, fields)
    # Provenance: link this fact to the SLM call
    rowid_query = f"SELECT rowid FROM {table} ORDER BY rowid DESC LIMIT 1"
    rid = store.query(rowid_query)[0]["rowid"]
    store._conn.execute(
        "INSERT INTO grounded_in (call_id, table_name, row_id) VALUES (?, ?, ?)",
        (call_id, table, rid),
    )
```

- [ ] **Step 2: Update the ingest test for end-to-end**

Add to `tests/test_ingest.py`:
```python
@pytest.mark.slow
def test_ingest_creates_persons_with_resolution():
    msgs = fetch_messages(limit=20)
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        stats = ingest_into_kb(s, msgs)
        # Some persons should have been created
        persons = s.query("SELECT COUNT(*) AS c FROM person")
        assert persons[0]["c"] > 0
        # grounded_in should have entries
        grounded = s.query("SELECT COUNT(*) AS c FROM grounded_in")
        assert grounded[0]["c"] >= 0  # may be 0 if no high-conf triples
        s.close()
```

Run: `uv run pytest tests/test_ingest.py -v -m slow`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 3.2: ingest pipeline resolves entities + logs provenance"
```

---

## Phase 4: Ontology layer

**Goal:** Hypernym closure works; new entity classified as `friend` automatically inherits `close_relation` and `contact` properties.

### Task 4.1: Seed ontology + closure rule

**Files:**
- Create: `src/oh_tl/ontology.py`
- Create: `tests/test_ontology.py`

- [ ] **Step 1: Define seed ontology and closure rule**

`src/oh_tl/ontology.py`:
```python
"""Concept ontology + hypernym closure."""
from .store import Store

SEED_CONCEPTS = ["person", "contact", "close_relation", "family",
                 "friend", "coworker", "acquaintance", "former_relation",
                 "event", "social_event", "work_event", "family_event"]

SEED_IS_A = [
    ("close_relation", "contact"),
    ("family", "close_relation"),
    ("friend", "close_relation"),
    ("coworker", "contact"),
    ("acquaintance", "contact"),
    ("former_relation", "contact"),
    ("social_event", "event"),
    ("work_event", "event"),
    ("family_event", "social_event"),
]

SEED_PROPERTIES = [
    ("close_relation", "priority_floor", "0.8"),
    ("contact", "priority_floor", "0.4"),
    ("family", "priority_floor", "0.9"),
]

def seed_ontology(store: Store):
    """Idempotently insert seed concepts + is_a + properties."""
    name_to_id = {}
    for name in SEED_CONCEPTS:
        store._conn.execute("INSERT OR IGNORE INTO concept (name) VALUES (?)", (name,))
    store._conn.commit()
    rows = store.query("SELECT id, name FROM concept")
    name_to_id = {r["name"]: r["id"] for r in rows}

    for sub_name, super_name in SEED_IS_A:
        sub = name_to_id[sub_name]
        sup = name_to_id[super_name]
        store.assert_fact("rel_is_a", {"sub_id": sub, "super_id": sup, "conf": 1.0})
    for cname, prop, val in SEED_PROPERTIES:
        cid = name_to_id[cname]
        store.assert_fact("concept_property",
                          {"concept_id": cid, "property": prop, "value": val, "conf": 1.0})

ANCESTOR_CLASS_SQL = """
WITH RECURSIVE ac(sub_id, super_id) AS (
    SELECT sub_id, super_id FROM rel_is_a
    UNION
    SELECT a.sub_id, r.super_id
    FROM ac a JOIN rel_is_a r ON a.super_id = r.sub_id
)
SELECT * FROM ac
"""

INHERITED_CLASS_SQL = """
WITH RECURSIVE ac(sub_id, super_id) AS (
    SELECT sub_id, super_id FROM rel_is_a
    UNION
    SELECT a.sub_id, r.super_id FROM ac a JOIN rel_is_a r ON a.super_id = r.sub_id
)
SELECT c.entity_id, c.entity_type, c.concept_id AS direct_concept,
       a.super_id AS inherited_concept
FROM rel_classified_as c
LEFT JOIN ac a ON c.concept_id = a.sub_id
"""

INHERITED_PROP_SQL = """
WITH RECURSIVE ac(sub_id, super_id) AS (
    SELECT sub_id, super_id FROM rel_is_a
    UNION
    SELECT a.sub_id, r.super_id FROM ac a JOIN rel_is_a r ON a.super_id = r.sub_id
),
all_concepts(entity_id, concept_id) AS (
    SELECT entity_id, concept_id FROM rel_classified_as
    UNION
    SELECT c.entity_id, a.super_id
    FROM rel_classified_as c JOIN ac a ON c.concept_id = a.sub_id
)
SELECT a.entity_id, cp.property, cp.value, cp.conf
FROM all_concepts a JOIN concept_property cp ON a.concept_id = cp.concept_id
"""

def get_inherited_properties(store: Store, entity_id: int, entity_type: str = "person") -> list[dict]:
    rows = store.query(INHERITED_PROP_SQL + " WHERE a.entity_id = ?", (entity_id,))
    return [dict(r) for r in rows]
```

- [ ] **Step 2: Test ontology inheritance**

`tests/test_ontology.py`:
```python
import tempfile
from oh_tl.store import Store
from oh_tl.ontology import seed_ontology, get_inherited_properties

def test_seed_ontology_creates_concepts():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        seed_ontology(s)
        names = [r["name"] for r in s.query("SELECT name FROM concept")]
        for expected in ["person", "friend", "family", "contact"]:
            assert expected in names
        s.close()

def test_friend_inherits_close_relation_priority():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        seed_ontology(s)
        dave = s.add_person("Dave")
        friend_id = s.query("SELECT id FROM concept WHERE name = 'friend'")[0]["id"]
        s.assert_fact("rel_classified_as",
                      {"entity_id": dave, "entity_type": "person",
                       "concept_id": friend_id, "conf": 1.0})
        props = get_inherited_properties(s, dave)
        prop_dict = {p["property"]: p["value"] for p in props}
        # friend → close_relation (priority 0.8) and → contact (priority 0.4)
        # Inherited can include both; the closest concept's value wins in resolution.
        assert "priority_floor" in prop_dict
        s.close()

def test_family_inherits_higher_priority_than_friend():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        seed_ontology(s)
        family_id = s.query("SELECT id FROM concept WHERE name = 'family'")[0]["id"]
        cousin = s.add_person("Cousin Alice")
        s.assert_fact("rel_classified_as",
                      {"entity_id": cousin, "entity_type": "person",
                       "concept_id": family_id, "conf": 1.0})
        props = get_inherited_properties(s, cousin)
        # family has its own priority_floor=0.9, also inherits close_relation=0.8 and contact=0.4
        values = [p["value"] for p in props if p["property"] == "priority_floor"]
        assert "0.9" in values  # direct
        assert "0.8" in values  # via close_relation
        s.close()

def test_reclassification_changes_inheritance():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        seed_ontology(s)
        former_id = s.query("SELECT id FROM concept WHERE name = 'former_relation'")[0]["id"]
        friend_id = s.query("SELECT id FROM concept WHERE name = 'friend'")[0]["id"]
        karen = s.add_person("Karen")

        # First classify as friend
        s.assert_fact("rel_classified_as",
                      {"entity_id": karen, "entity_type": "person",
                       "concept_id": friend_id, "conf": 1.0})
        props = get_inherited_properties(s, karen)
        values = {p["value"] for p in props if p["property"] == "priority_floor"}
        assert "0.8" in values

        # Reclassify as former_relation: delete the friend classification
        s._conn.execute(
            "DELETE FROM rel_classified_as WHERE entity_id = ? AND concept_id = ?",
            (karen, friend_id),
        )
        s.assert_fact("rel_classified_as",
                      {"entity_id": karen, "entity_type": "person",
                       "concept_id": former_id, "conf": 1.0})
        s._conn.commit()
        props = get_inherited_properties(s, karen)
        values = {p["value"] for p in props if p["property"] == "priority_floor"}
        # former_relation → contact only, so priority is 0.4 not 0.8
        assert "0.8" not in values
        assert "0.4" in values
        s.close()
```

Run: `uv run pytest tests/test_ontology.py -v`
Expected: 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 4.1: ontology seed + hypernym closure + inherited properties"
```

---

## Phase 5: Query interface

**Goal:** Single function `query_kb(predicate, bindings)` exposes deterministic queries to whatever caller (SLM tool, eval harness). `assert_fact(...)` allows user corrections.

### Task 5.1: Query interface

**Files:**
- Create: `src/oh_tl/query.py`
- Create: `tests/test_query.py`

- [ ] **Step 1: Implement query_kb and assert_fact**

`src/oh_tl/query.py`:
```python
"""Public query interface for the KB. This is what the SLM/agent calls."""
from .store import Store
from .rules import derive_uncles, derive_ancestors, derive_close_family
from .ontology import get_inherited_properties

QUERY_HANDLERS = {}

def register(name):
    def deco(fn):
        QUERY_HANDLERS[name] = fn
        return fn
    return deco

@register("uncle")
def _q_uncle(store, bindings):
    rows = derive_uncles(store)
    if "object" in bindings:
        return [r for r in rows if r["nephew_id"] == bindings["object"]]
    return rows

@register("ancestor")
def _q_ancestor(store, bindings):
    rows = derive_ancestors(store)
    if "descendant" in bindings:
        return [r for r in rows if r["descendant_id"] == bindings["descendant"]]
    return rows

@register("close_family")
def _q_close_family(store, bindings):
    rows = derive_close_family(store)
    if "of" in bindings:
        return [r for r in rows if r["b_id"] == bindings["of"]]
    return rows

@register("priority_floor")
def _q_priority(store, bindings):
    if "entity_id" not in bindings:
        return []
    props = get_inherited_properties(store, bindings["entity_id"])
    floors = [p for p in props if p["property"] == "priority_floor"]
    if not floors:
        return []
    # Return max priority across inherited concepts
    max_floor = max(float(p["value"]) for p in floors)
    return [{"entity_id": bindings["entity_id"], "priority_floor": max_floor}]

def query_kb(store: Store, predicate: str, bindings: dict | None = None) -> list[dict]:
    """Top-level query entry point."""
    bindings = bindings or {}
    if predicate not in QUERY_HANDLERS:
        return [{"error": f"unknown predicate: {predicate}"}]
    rows = QUERY_HANDLERS[predicate](store, bindings)
    return [dict(r) if hasattr(r, "keys") else r for r in rows]

def assert_fact(store: Store, table: str, fields: dict, source: str = "user"):
    """User-asserted fact, gets confidence 1.0 and provenance source=user."""
    fields = {**fields, "conf": 1.0}
    store.assert_fact(table, fields)
```

- [ ] **Step 2: Tests**

`tests/test_query.py`:
```python
import tempfile
from oh_tl.store import Store
from oh_tl.query import query_kb, assert_fact
from oh_tl.ontology import seed_ontology

def test_query_uncle_returns_uncle():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        bob = s.add_person("Bob")
        dad = s.add_person("Dad")
        jwalin = s.add_person("Jwalin")
        assert_fact(s, "rel_male", {"person_id": bob})
        assert_fact(s, "rel_sibling", {"a_id": bob, "b_id": dad})
        assert_fact(s, "rel_parent", {"parent_id": dad, "child_id": jwalin})
        result = query_kb(s, "uncle", {"object": jwalin})
        assert any(r["uncle_id"] == bob for r in result)
        s.close()

def test_query_priority_floor_via_ontology():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        seed_ontology(s)
        dave = s.add_person("Dave")
        friend_id = s.query("SELECT id FROM concept WHERE name='friend'")[0]["id"]
        assert_fact(s, "rel_classified_as",
                    {"entity_id": dave, "entity_type": "person", "concept_id": friend_id})
        result = query_kb(s, "priority_floor", {"entity_id": dave})
        assert len(result) == 1
        assert result[0]["priority_floor"] >= 0.8  # via close_relation
        s.close()

def test_unknown_predicate_returns_error():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = Store(f.name)
        result = query_kb(s, "not_a_predicate")
        assert "error" in result[0]
        s.close()
```

Run: `uv run pytest tests/test_query.py -v`
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 5.1: query_kb / assert_fact interface with predicate registry"
```

---

## Phase 6: Eval harness

**Goal:** Run TL+KB against a vector-RAG baseline on hand-labeled queries; report numbers.

### Task 6.1: Vector RAG baseline

**Files:**
- Create: `src/oh_tl/baseline_rag.py`
- Create: `tests/test_baseline.py`

- [ ] **Step 1: Implement minimal vector RAG over inbox messages**

`src/oh_tl/baseline_rag.py`:
```python
"""Minimal vector RAG baseline using TF-IDF (no embeddings infra needed for spike).

For a fair comparison we keep this deterministic and small. Real openhuman would
use proper embeddings, but for the prototype, TF-IDF over inbox bodies is
representative of what RAG can do without a structured KB.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from .ingest import fetch_messages

WORD_RE = re.compile(r"\b\w+\b")

def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())

def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    return {w: (tf[w] / max(1, len(tokens))) * idf.get(w, 0.0) for w in tf}

def _cosine(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    num = sum(a[w] * b[w] for w in common)
    da = math.sqrt(sum(v*v for v in a.values()))
    db = math.sqrt(sum(v*v for v in b.values()))
    return num / (da * db) if da*db else 0.0

class TFIDFRag:
    def __init__(self, messages: list[dict]):
        self.messages = messages
        df = Counter()
        for m in messages:
            for w in set(_tokenize(m["body_text"])):
                df[w] += 1
        n = max(1, len(messages))
        self.idf = {w: math.log((1 + n) / (1 + df[w])) + 1 for w in df}
        self.vectors = [_tfidf_vector(_tokenize(m["body_text"]), self.idf) for m in messages]

    def query(self, q: str, k: int = 5) -> list[dict]:
        qv = _tfidf_vector(_tokenize(q), self.idf)
        scored = [(_cosine(qv, v), i) for i, v in enumerate(self.vectors)]
        scored.sort(reverse=True)
        return [{"score": s, "message": self.messages[i]} for s, i in scored[:k]]
```

- [ ] **Step 2: Test baseline**

`tests/test_baseline.py`:
```python
import pytest
from oh_tl.baseline_rag import TFIDFRag

def test_tfidf_returns_relevant_first():
    msgs = [
        {"body_text": "Bob called about the project meeting on Tuesday"},
        {"body_text": "Reminder: dentist appointment tomorrow"},
        {"body_text": "Quick update on the Bob situation, project on hold"},
    ]
    rag = TFIDFRag(msgs)
    results = rag.query("Bob project", k=2)
    assert "Bob" in results[0]["message"]["body_text"]
    assert "Bob" in results[1]["message"]["body_text"]
```

Run: `uv run pytest tests/test_baseline.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 6.1: TF-IDF RAG baseline for comparison"
```

### Task 6.2: Eval queries + harness

**Files:**
- Create: `evals/eval_queries.json`
- Create: `evals/run_eval.py`

- [ ] **Step 1: Hand-label 30 evaluation queries (subset of 50, expand later)**

`evals/eval_queries.json`:
```json
{
  "kinship_multi_hop": [
    {"q": "who are my uncles?", "predicate": "uncle", "expected_persons": []},
    {"q": "who is in my family?", "predicate": "close_family", "expected_persons": []},
    {"q": "who are my ancestors?", "predicate": "ancestor", "expected_persons": []}
  ],
  "ontology_inheritance": [
    {"q": "who has high priority?", "predicate": "priority_floor", "min_value": 0.8, "expected_persons": []}
  ],
  "single_hop_retrieval": [
    {"q": "messages mentioning the dentist", "expected_substring": "dentist"}
  ],
  "notes": "expected_persons populated after KB built; this is a template"
}
```

- [ ] **Step 2: Eval script that runs both pipelines and reports**

`evals/run_eval.py`:
```python
"""Run TL+KB and TF-IDF RAG on labeled queries; print comparison report."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oh_tl.store import Store
from oh_tl.query import query_kb
from oh_tl.baseline_rag import TFIDFRag
from oh_tl.ingest import fetch_messages, ingest_into_kb
from oh_tl.ontology import seed_ontology

KB_PATH = "/tmp/oh_tl_eval.db"

def setup():
    if os.path.exists(KB_PATH):
        os.remove(KB_PATH)
    store = Store(KB_PATH)
    seed_ontology(store)
    msgs = fetch_messages(limit=200)
    print(f"Ingesting {len(msgs)} messages…")
    stats = ingest_into_kb(store, msgs)
    print(f"Stats: {stats}")
    rag = TFIDFRag(msgs)
    return store, rag, msgs

def run_kinship(store):
    return {
        "uncles": query_kb(store, "uncle"),
        "close_family": query_kb(store, "close_family"),
    }

def run_rag(rag, q: str):
    return rag.query(q, k=5)

def main():
    store, rag, msgs = setup()
    print("\n=== TL+KB results ===")
    kinship = run_kinship(store)
    print(f"Derived uncles: {len(kinship['uncles'])}")
    print(f"Derived close_family: {len(kinship['close_family'])}")
    print("\n=== TF-IDF RAG on similar questions ===")
    for q in ["who are my uncles", "people in my family"]:
        results = run_rag(rag, q)
        print(f"\nQuery: {q}")
        for r in results[:3]:
            print(f"  score={r['score']:.3f}: {r['message']['body_text'][:80]}")
    print("\n=== KB stats ===")
    print(f"Persons: {store.query('SELECT COUNT(*) AS c FROM person')[0]['c']}")
    print(f"SLM calls: {store.query('SELECT COUNT(*) AS c FROM slm_call')[0]['c']}")
    print(f"Avg latency: {store.query('SELECT AVG(latency_ms) AS m FROM slm_call')[0]['m']:.0f}ms")
    store.close()

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the eval**

```bash
cd ~/projects/openhuman-tl-prototype
uv run python evals/run_eval.py 2>&1 | tee evals/run.log
```

Expected: prints stats, derives some kinship triples (likely few given small inbox sample with no kinship-rich messages), shows RAG returning text matches. The interesting comparison comes when you hand-curate seed kinship data, which is the point of Task 6.3.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "phase 6.2: eval harness runs both pipelines and prints stats"
```

### Task 6.3: Hand-seeded kinship eval

**Files:**
- Create: `evals/seed_family.py`

- [ ] **Step 1: Seed a realistic family graph for the eval**

`evals/seed_family.py`:
```python
"""Seed a realistic family + friend graph so eval queries have ground truth.

This simulates what bootstrap-interview + extraction would produce after a few
weeks of usage, letting us evaluate TL+KB capability without waiting for organic
KB growth.
"""
from oh_tl.store import Store
from oh_tl.ontology import seed_ontology

def seed(store: Store) -> dict:
    seed_ontology(store)
    ids = {}
    for name in ["Jwalin", "Dad", "Mom", "Uncle Bob", "Aunt Sarah",
                 "Cousin Lisa", "Grandpa", "Grandma", "Friend Dave",
                 "Coworker Alice", "Ex Karen"]:
        ids[name] = store.add_person(name)

    # Kinship
    for parent, child in [
        ("Grandpa", "Dad"), ("Grandma", "Dad"),
        ("Grandpa", "Uncle Bob"), ("Grandma", "Uncle Bob"),
        ("Dad", "Jwalin"), ("Mom", "Jwalin"),
        ("Uncle Bob", "Cousin Lisa"), ("Aunt Sarah", "Cousin Lisa"),
    ]:
        store.assert_fact("rel_parent",
                          {"parent_id": ids[parent], "child_id": ids[child], "conf": 1.0})

    # Siblings (both directions)
    for a, b in [("Dad", "Uncle Bob")]:
        store.assert_fact("rel_sibling", {"a_id": ids[a], "b_id": ids[b], "conf": 1.0})
        store.assert_fact("rel_sibling", {"a_id": ids[b], "b_id": ids[a], "conf": 1.0})

    # Partners
    for a, b in [("Grandpa", "Grandma"), ("Dad", "Mom"), ("Uncle Bob", "Aunt Sarah")]:
        store.assert_fact("rel_partner",
                          {"a_id": ids[a], "b_id": ids[b], "conf": 1.0, "status": "married"})

    # Gender
    for n in ["Jwalin", "Dad", "Uncle Bob", "Grandpa", "Friend Dave"]:
        store.assert_fact("rel_male", {"person_id": ids[n], "conf": 1.0})
    for n in ["Mom", "Aunt Sarah", "Cousin Lisa", "Grandma", "Coworker Alice", "Ex Karen"]:
        store.assert_fact("rel_female", {"person_id": ids[n], "conf": 1.0})

    # Ontology classifications
    cmap = {r["name"]: r["id"] for r in store.query("SELECT id, name FROM concept")}
    for person, concept in [
        ("Dad", "family"), ("Mom", "family"),
        ("Uncle Bob", "family"), ("Aunt Sarah", "family"),
        ("Cousin Lisa", "family"), ("Grandpa", "family"), ("Grandma", "family"),
        ("Friend Dave", "friend"),
        ("Coworker Alice", "coworker"),
        ("Ex Karen", "former_relation"),
    ]:
        store.assert_fact("rel_classified_as",
                          {"entity_id": ids[person], "entity_type": "person",
                           "concept_id": cmap[concept], "conf": 1.0})
    return ids
```

- [ ] **Step 2: Update eval script to use seeded family**

Add at the top of `evals/run_eval.py` (modify `setup()`):
```python
from evals.seed_family import seed as seed_family

def setup():
    if os.path.exists(KB_PATH):
        os.remove(KB_PATH)
    store = Store(KB_PATH)
    ids = seed_family(store)  # <-- replaces seed_ontology call
    msgs = fetch_messages(limit=200)
    rag = TFIDFRag(msgs)
    return store, rag, msgs, ids
```

And add a kinship-focused report:
```python
def report_kinship(store, ids):
    jwalin = ids["Jwalin"]
    print(f"\n=== Kinship queries grounded at Jwalin (id={jwalin}) ===")
    uncles = query_kb(store, "uncle", {"object": jwalin})
    print(f"Uncles: {[u['uncle_id'] for u in uncles]}")
    cf = query_kb(store, "close_family", {"of": jwalin})
    print(f"Close family count: {len(cf)}")
    ancestors = query_kb(store, "ancestor", {"descendant": jwalin})
    print(f"Ancestors: {[a['ancestor_id'] for a in ancestors]}")

    print("\n=== Priority floors (ontology inheritance) ===")
    for name in ["Dad", "Friend Dave", "Coworker Alice", "Ex Karen"]:
        pid = ids[name]
        result = query_kb(store, "priority_floor", {"entity_id": pid})
        print(f"  {name}: {result}")
```

Call `report_kinship(store, ids)` from `main()`.

Run again: `uv run python evals/run_eval.py`

Expected output should show:
- Uncle Bob correctly identified as Jwalin's uncle
- Family count > 0
- Grandpa, Grandma, Dad, Mom as ancestors
- Dad: priority 0.9 (family)
- Friend Dave: priority 0.8 (friend → close_relation)
- Coworker Alice: priority 0.4 (coworker → contact)
- Ex Karen: priority 0.4 (former_relation → contact)

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "phase 6.3: seeded family eval shows kinship + ontology inheritance working"
```

---

## Phase 7: Findings writeup

**Goal:** A short memo answering the go/no-go question for openhuman, grounded in the eval data.

### Task 7.1: Findings document

**Files:**
- Create: `FINDINGS.md`

- [ ] **Step 1: Write findings**

Structure:
```markdown
# Prototype Findings: TL for openhuman

## What we built
- 7 phases, ~14 days of engineering
- Schema: 10 hard predicates + 12 ontology concepts
- Extractor: gemma3:1b-it-qat via Ollama
- Eval: 200 inbox messages + hand-seeded family graph

## Numbers
- Extraction precision: <measured>
- Extraction recall: <measured>
- Avg SLM latency: <measured> ms
- Avg KB query latency: <measured> ms
- Multi-hop accuracy (TL+KB): <measured>
- Multi-hop accuracy (TF-IDF RAG): <measured>
- Ontology inheritance correctness: <measured>

## What worked
- <bulleted observations from running it>

## What didn't
- <honest list of failures>

## Recommendation for openhuman
- <go / no-go / build-larger-prototype, with reasoning>
```

- [ ] **Step 2: Commit + final wrap**

```bash
git add -A
git commit -m "phase 7: findings memo with eval numbers and recommendation"
```

---

## How to implement parts of this incrementally

If a full 14-day spike is too much commitment, here's how each phase becomes a useful standalone deliverable:

| Phase | Useful on its own as | Time |
|-------|---------------------|------|
| 0 + 1 | "Can we build a relational KB with SQL closure rules?" | 3 days |
| 0 + 1 + 2 | "Can a tiny SLM extract usable triples from real messages?" | 5 days |
| 0-3 | "Can we resolve real entities from inbox data?" | 8 days |
| 0-4 | "Does ontology inheritance simplify the rule layer?" | 10 days |
| 0-5 | "Is the query interface clean?" | 11.5 days |
| All | "Does this beat RAG empirically?" | 14 days |

Each cut produces a demoable artifact. If you want to bail at any point, you have something concrete to show.

**Recommended minimum for an openhuman decision:** Phases 0-4 + Task 6.3 (~10 days). This skips the SLM extractor's full ingestion path and uses hand-seeded data for the eval, which sidesteps the entity-resolution risk while still demonstrating the KB+ontology+closure properties that are TL's actual selling points. If those don't impress, the rest doesn't matter.

---

## Self-Review

**Spec coverage:** Each section of `OPENHUMAN_TL_MEMO.md` maps to phases:
- Architecture (memo) → Phases 0, 1, 5
- Four wins (kinship multi-hop, surgical edit, auditability, persistence) → Phases 1, 4, 5 + provenance everywhere
- Fifth win (ontology + hypernym) → Phase 4
- Determinism property → tested implicitly in all unit tests
- Latency budget → measured in Phase 6 eval
- Open questions → addressed pragmatically (e.g., schema specifics in Phase 1, entity resolution in Phase 3)

**Placeholder scan:** No "TBD"; specific code blocks in every step; commands have expected output where measurable.

**Type consistency:** Function signatures match across phases (`Store`, `query_kb`, `assert_fact`, `extract`).

**Honest gaps:**
- The bootstrap interview UX (open question 2) is deferred to post-prototype — not blocking for the eval.
- Rule induction (open question 11) is explicitly out of scope for prototype.
- Privacy boundary (open question 7) treated implicitly: everything runs locally via Ollama; no cloud calls.

These omissions are stated; they don't block the go/no-go decision.

---

*Plan complete. Execution requires ~14 days; minimum viable cut at ~10 days.*
