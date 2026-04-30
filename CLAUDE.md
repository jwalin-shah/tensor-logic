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

## LM Backend Support

`lm_prune()` in `experiments/exp78_rule_induction.py` now supports both MLX and transformers backends:
- Detects MLX-compatible models by naming convention (`"mlx-community"` in model name, `-mlx` suffix, or `-MLX` in name)
- Loads MLX models via `mlx_lm.load()` when available
- Falls back to transformers + torch (`AutoModelForCausalLM`) if MLX loading fails
- Transformers models loaded with `torch.float16` for memory efficiency
- Backend state cached in `_LM_CACHE` per model

**Known limitation**: VLM models (e.g., `mlx-community/Qwen3.5-0.8B-MLX-4bit`) are incompatible with `mlx_lm.load()` due to missing video processor support. Use text-only models or transformers backend instead.

## Batch Experiment Runner

`experiments/exp81_sweep.py` runs multiple rule induction tasks in sequence, useful for testing model/schema combinations across the experiment grid. Supports difficulty modes (`normal`, `hard_v2`, `extra_hard`) and includes step-level logging with failure mode classification for diagnostics.

## Notes

- `AGENTS.md` is memjuice-managed — do not edit it manually.
- `.cocoindex_code/` contains a vector embedding index (semantic search) — separate from `tools/index.json` (structured API signatures).
