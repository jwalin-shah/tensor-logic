# FAFSA Engine — New Repo Design

**Date:** 2026-04-28  
**Status:** Approved  
**Repo name:** `fafsa-engine` (working name — confirm before creating)

---

## Problem

Every FAFSA calculator gives you a number. You trust it or you don't. There's no way to check it. LLMs can narrate a result but can hallucinate the rules that produced it.

This repo demonstrates a different pattern: **LLM handles language, proof engine handles correctness.** The proof is the contract between them — machine-checkable, citable, auditable.

---

## What It Is

A end-to-end neurosymbolic pipeline for FAFSA SAI computation:

1. User asks in plain English
2. LLM extracts structured facts
3. `tensor_logic` proof engine computes SAI deterministically with full derivation trace
4. LLM narrates the proof in plain English
5. Verification layer confirms engine output matches known-correct answers

The engine is the source of truth. The LLM is the language layer. Swapping LLM backends changes presentation, never correctness.

---

## Repo Structure

```
fafsa-engine/
├── README.md
├── demo.py                  ← LLM-in-the-loop pipeline: natural language → facts → proof → narration
├── wizard.py                ← structured terminal Q&A (no LLM required, just asks questions)
├── fafsa/
│   ├── __init__.py
│   ├── kb.py                ← TL rules from federal guidelines (from exp80_fafsa_kb.py)
│   ├── wizard.py            ← structured terminal Q&A (ask questions one-by-one, no LLM needed)
│   └── validate.py          ← 1015-family synthetic validation
├── llm/
│   ├── __init__.py
│   ├── base.py              ← LLMBackend abstract interface
│   ├── ollama_backend.py    ← default: Ollama (Qwen3.5:4b)
│   ├── claude_backend.py    ← optional: Claude API
│   └── openai_backend.py    ← optional: OpenAI API
├── tensor_logic/            ← engine copied as-is from tensor repo
├── examples/
│   └── counterfactual.py    ← sweep parent income $40k→$200k, plot SAI curve
├── tests/
│   ├── test_fafsa_kb.py     ← smoke tests for rule engine
│   └── test_llm_backends.py ← mock-based backend tests
└── pyproject.toml
```

---

## System Architecture

```
User (natural language)
    ↓
LLMBackend.extract_facts(query) → dict of structured facts
    ↓
fafsa.kb.prove_sai(facts) → SAITrace (proof tree + value)
    ↓
LLMBackend.narrate_proof(trace) → plain-English explanation
    ↓
validate.verify(trace) → ✅ or ❌ with discrepancy details
    ↓
Output: facts + proof tree + narration + verification status
```

The `LLMBackend` interface has two methods: `extract_facts(query: str) -> dict` and `narrate_proof(trace: SAITrace) -> str`. Everything else is engine.

---

## LLM Backend Configuration

```bash
# Default — Ollama with Qwen3.5:4b (no API key, runs on CPU)
python demo.py "My parents make $80k, family of 4"

# Swap local model
FAFSA_LLM_MODEL=gemma4:4b python demo.py "..."
FAFSA_LLM_MODEL=qwen3.5:9b python demo.py "..."

# Use Claude API
FAFSA_LLM=claude ANTHROPIC_API_KEY=sk-... python demo.py "..."

# Use OpenAI API
FAFSA_LLM=openai OPENAI_API_KEY=sk-... python demo.py "..."
```

Backend resolution order: `FAFSA_LLM` env var → default Ollama. Model resolution: `FAFSA_LLM_MODEL` env var → `qwen3.5:4b`.

The Ollama backend requires `ollama` running locally (`ollama serve`) with the model pulled (`ollama pull qwen3.5:4b`). README documents this as a one-time setup step.

---

## README Structure

Opens with: *"An LLM will tell you your SAI. This will prove it."*

Immediately followed by a terminal output block showing the full pipeline output — facts extracted, proof tree, narration, verification tick. No explanation before it.

**Sections:**

1. **Quick start** — four commands: clone, install deps, `ollama pull qwen3.5:4b`, run demo. Works in under 2 minutes on CPU, no API key required.
2. **What you see** — annotated breakdown of the output block: which part is LLM, which part is engine, what the verification tick means.
3. **How it works** — three sentences: rules extracted from federal guidelines → encoded as Tensor Logic facts → proof engine derives SAI with full derivation trace. LLM extracts your facts and narrates the result; the engine guarantees correctness regardless of which model you use.
4. **Swap the LLM** — one-line table: Ollama (default), Claude, OpenAI. One env var to switch.
5. **Beyond FAFSA** — four bullets: Medicaid eligibility, tax compliance, clinical guidelines, visa eligibility. Same engine, different rule file. One paragraph, no overclaiming.
6. **Engine** — two sentences pointing to `tensor_logic/`, link to Domingos (2025).
7. **License** — MIT.

Under 500 words. No "what's next" section, no long reference list.

---

## What Gets Copied / Adapted

| Source (tensor repo) | Destination | Changes |
|---|---|---|
| `experiments/exp80_fafsa_kb.py` | `fafsa/kb.py` | Refactor into importable module, remove exp scaffolding |
| `experiments/exp80_fafsa_wizard.py` | `fafsa/wizard.py` | Refactor into callable functions |
| `experiments/exp80_validate_synthetic.py` | `fafsa/validate.py` | Expose `verify(trace)` function |
| `tensor_logic/` | `tensor_logic/` | Source files only — tests NOT copied (they reference exp79/exp81 which don't exist in new repo) |
| `tests/test_tensor_logic_core.py` | `tests/test_fafsa_kb.py` | Scoped to FAFSA smoke tests only |

`demo.py` is new — wires LLM backend + fafsa package end-to-end.

---

## Dependencies

```toml
[project]
name = "fafsa-engine"
requires-python = ">=3.11"
dependencies = [
    "torch",          # tensor_logic substrate
    "requests",       # ollama HTTP API
]

[project.optional-dependencies]
claude = ["anthropic"]
openai = ["openai"]
```

No OpenFisca dependency. MIT license, no AGPL exposure.

---

## Verification Layer

`validate.verify(trace: SAITrace) -> VerificationResult` runs on every query. It checks the computed SAI against the pre-validated 1015-family synthetic dataset: given the same input facts, does the engine produce the same output it did during validation?

If it matches: `✅ verified (matches validated dataset)`.  
If it doesn't match or facts are outside the validation set: `⚠️ unverified (novel input — engine result not cross-checked)`.

Never silently hides a discrepancy.

---

## Success Criteria

1. `python demo.py "My parents make $80k, family of 4"` runs end-to-end with Qwen3.5:4b and prints facts + proof tree + narration + ✅ in under 30 seconds on CPU
2. Swapping `FAFSA_LLM=claude` produces the same proof tree, different narration
3. `pytest tests/ -v` passes with no external API calls (backends mocked)
4. README under 500 words, derivation trace visible without scrolling on a 1080p screen

---

## What This Is Not

- Not a financial advisor — disclaimer in README
- Not a replacement for the official FAFSA4caster
- Not a general-purpose LLM wrapper — the engine is the source of truth
- Not dependent on OpenFisca (MIT, no AGPL)
