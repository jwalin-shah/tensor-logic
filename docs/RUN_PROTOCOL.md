# Run protocol — don't lose weights, don't waste compute

The exp76c adapter was lost because Kaggle's `/kaggle/working/` is wiped on
session end unless you commit a notebook version. This protocol exists so that
doesn't happen again.

## Before every Kaggle run

1. Pull latest: `cd tensor-logic && git pull origin main`
2. Set output dir to `/kaggle/working/<expN>_<tag>/` — never bare `/kaggle/working/`.
3. Set `--dump-failures` to a path inside the same dir.
4. Plan to commit a notebook version — not just an interactive run.

## During the run

The script must write three files into the output dir:

```
/kaggle/working/<expN>_<tag>/
├── adapter/                    ← LoRA weights (or full ckpt)
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── manifest.json               ← config + metrics + git sha + timestamp
└── failures.jsonl              ← raw responses for offline analysis
```

`manifest.json` schema:
```json
{
  "experiment": "exp76c",
  "git_sha": "2767bdf",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "train_file": "train_paraphrased.jsonl",
  "eval_file": "eval_hard.jsonl",
  "epochs": 3,
  "batch_size": 12,
  "n_eval": 200,
  "metrics": {
    "A_base": 0.455,
    "B_sft_no_tool": 0.46,
    "C_sft_tool": 0.83,
    "tool_validity": 0.80
  },
  "started_at": "2026-04-27T18:57:00Z",
  "finished_at": "2026-04-27T19:16:40Z"
}
```

Add a small helper to `exp60d_sft.py` (and future scripts) that writes this
manifest at end of `main()`. One block, ~15 lines.

## After the run — Kaggle persistence

**The only reliable way to keep `/kaggle/working/` is "Save Version → Save & Run All (Commit)".** Quick-save / interactive saves are not enough.

Verify in this order:

1. Open the saved notebook version → Output tab → confirm `<expN>_<tag>/` is
   there with non-zero `adapter_model.safetensors`.
2. Click "New Dataset from Notebook Output" → name it `tensorlogic-<expN>-<tag>`
   → make it the canonical adapter source.
3. Future eval-only runs: attach that dataset as input, point `--out` at
   `/kaggle/input/tensorlogic-<expN>-<tag>/adapter/`.

## After the run — local mirror

Don't commit weights to git (LoRA adapters are 4–8 MB, full ckpts are huge).
Instead, commit a *pointer*:

```
experiments/expN_data/
├── results.json     ← copy of manifest.json (committed)
└── ADAPTER.md       ← Kaggle dataset URL + git sha at training time
```

`ADAPTER.md` template:
```
Adapter: kaggle.com/datasets/<user>/tensorlogic-<expN>-<tag>
Trained from git: <sha>
Trained on: <date>
Metrics: see results.json
```

This way the repo always has provenance and metrics, even if the Kaggle dataset
is later deleted you know exactly what config to reproduce from.

## Repo structure invariants

```
experiments/
  expN_<tag>.py           ← runnable script
  expN_data/              ← committed data + results pointers (no weights)
    train.jsonl
    eval.jsonl
    results.json
    ADAPTER.md
docs/
  expN_<tag>_spec.md      ← spec written BEFORE the run, with falsification criterion
  RUN_PROTOCOL.md         ← this file
notes/
  RESEARCH_NOTES.md       ← findings across experiments
  IDEAS.md                ← unrun candidates
```

## Compute discipline

- **Spec before code.** No experiment without a `docs/expN_*_spec.md` containing
  a falsification criterion. If you can't state one, you don't know what you're
  testing.
- **Eval-only loops over re-training.** If the adapter is saved, `--eval-only`
  is 90 seconds vs 16 minutes. Use it when iterating on eval logic, prompts, or
  failure dumps.
- **Don't retrain to fix eval bugs.** Patch eval, point `--out` at the saved
  adapter, run `--eval-only`.
- **Commit notebook on every successful run.** Even if the run failed, commit
  it — the failure logs are themselves data.
