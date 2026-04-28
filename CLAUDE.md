# tensor — Project Instructions

## Planning

Before writing any implementation plan that touches `tensor_logic/`, run:

```bash
python tools/code_index.py --lookup <RelevantSymbol>
```

to get current constructor signatures and function args. This prevents plan/code divergence from stale API knowledge. Use `--dump` to see all indexed symbols.

## Test suite

```bash
pytest tests/ -v
```

## Notes

- `AGENTS.md` is memjuice-managed — do not edit it manually.
- `.cocoindex_code/` contains a vector embedding index (semantic search) — separate from `tools/index.json` (structured API signatures).
