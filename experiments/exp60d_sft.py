"""
exp60d (step 4/4 of the TL-as-tool integration line):
SFT a small instruct LM to emit <tl_closure ...> tags, route the tags
through exp60b's harness, and compare three conditions on the eval set:

  (A) base LM, zero-shot, no tool         — the "LM alone" lower bound
  (B) SFT'd LM, no tool                   — does fine-tuning alone fix it?
  (C) SFT'd LM + TL tool harness          — the integration claim

Falsification (from notes/IDEAS.md):
  - (C) accuracy at hop ≥ 3 must be ≥ 1.5× (A) accuracy, OR the SFT-only
    accuracy claim falsifies the substrate's value.
  - Tool-call syntactic validity rate must be ≥ 95% on (C).

Defaults are tuned to run end-to-end on a Mac (MPS) or a single Colab T4
in well under an hour:
  - Model: Qwen/Qwen2.5-0.5B-Instruct  (overridable via --model)
  - LoRA r=8, 1 epoch, lr=2e-4
  - 1000 train traces (exp60a output), 200 eval

Run order:
  1. python3 exp60a_kinship_traces.py     # generate exp60_data/{train,eval}.jsonl
  2. python3 exp60b_tl_tool_harness.py    # smoke test
  3. python3 exp60c_rule_baseline.py      # 100% on substrate side
  4. python3 exp60d_sft.py                # this file

Dependencies (install once):
  pip install torch transformers peft datasets accelerate
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
DATA = HERE / "exp60_data"

# Import the harness as a sibling module
_spec = importlib.util.spec_from_file_location(
    "exp60b_harness", HERE / "exp60b_tl_tool_harness.py"
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

# Optional: exp65's rule-chain evaluator, for routing <tl_rule> tags from
# exp76 (multi-relation rule SFT). Imported lazily so exp60d still works
# in isolation.
_RULE_SPEC = importlib.util.spec_from_file_location(
    "exp65_rules", HERE / "exp65_rule_chain_joins.py"
)
rule_harness = importlib.util.module_from_spec(_RULE_SPEC)
_RULE_SPEC.loader.exec_module(rule_harness)

# Strict superset of exp65's RULE_RE — adds optional subject/object attrs
# so the LM can emit a self-contained tag (rule + concrete query).
RULE_RE_FULL = re.compile(
    r'<tl_rule\s+head="(?P<head>[^"]+)"\s+body="(?P<body>[^"]+)"'
    r'\s+subject="(?P<subj>[^"]+)"\s+object="(?P<obj>[^"]+)"\s*></tl_rule>'
)


def route_response(graph: dict, resp: str):
    """Try <tl_closure> first (exp60d shape), then <tl_rule> (exp76 shape).

    Returns: (answer "yes"/"no"/None, valid: bool).
    valid=True iff at least one well-formed tag was parsed.
    """
    closure = harness.evaluate_string(graph, resp)
    if closure:
        return closure[0]["result"]["answer"], True
    m = RULE_RE_FULL.search(resp)
    if m:
        head_atom = rule_harness.ATOM_RE.search(m.group("head"))
        body_atoms = list(rule_harness.ATOM_RE.finditer(m.group("body")))
        if head_atom and body_atoms:
            head = (head_atom.group("rel"), head_atom.group("a"), head_atom.group("b"))
            body = [(a.group("rel"), a.group("a"), a.group("b")) for a in body_atoms]
            head_result, err = rule_harness.evaluate_rule(graph, (head, body))
            if head_result is not None:
                related = rule_harness.query_relation(
                    head_result, m.group("subj"), m.group("obj"),
                )
                return ("yes" if related else "no"), True
    return None, False


SYSTEM_PROMPT = (
    "You answer kinship questions. When the user asks a reachability "
    "question over a `parent` relation, emit a single tool call of the form "
    '`<tl_closure relation="parent" query="<ancestor|descendant>" '
    'subject="<name>" object="<name>"></tl_closure>` and nothing else. '
    "The tool will return the answer."
)


def format_graph(graph: dict) -> str:
    """Pretty-print the graph as 'parent(a, b)' facts for the prompt."""
    lines = []
    for rel, pairs in graph.items():
        for src, dst in pairs:
            lines.append(f"{rel}({src}, {dst})")
    return "\n".join(lines)


def render_user_msg(trace: dict) -> str:
    return f"Facts:\n{format_graph(trace['graph'])}\n\nQuestion: {trace['query']}"


def load_traces(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------- Training ----------

def train_lora(model_name: str, train_traces, out_dir: Path, epochs: int, lr: float,
               batch_size: int = 12, grad_ckpt: bool = True):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_chat(trace):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_msg(trace)},
            {"role": "assistant", "content": trace["gold_tool_call"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False)
        return {"text": text}

    ds = Dataset.from_list([to_chat(t) for t in train_traces])

    # Dynamic padding: tokenize without padding here, pad-to-longest in the
    # data collator. Most kinship traces are <256 tokens; padding to 512
    # was wasting ~50% of forward/backward compute.
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=384)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        # T4 (16GB) sits comfortably at bs=12 for 0.5B + LoRA r=8 with
        # dynamic padding to ~256. Bump down if OOM on a smaller GPU.
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=(device == "cuda"),
        fp16=False,
        gradient_checkpointing=grad_ckpt,
    )
    # mlm=False + this collator pads to longest in batch (dynamic padding).
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir


# ---------- Inference ----------

def load_for_inference(model_name: str, adapter_dir: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32 if device == "mps" else torch.bfloat16,
    ).to(device)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir)).to(device)
    model.eval()
    return model, tok, device


def generate(model, tok, device, system: str, user: str, max_new_tokens: int = 96) -> str:
    """Single-example generate, kept for compatibility / debugging."""
    return generate_batch(model, tok, device, system, [user], max_new_tokens)[0]


def generate_batch(model, tok, device, system: str, users: list[str],
                   max_new_tokens: int = 80) -> list[str]:
    """Batched generation with left-padding.

    Causal LMs need LEFT padding so the last real token is at position -1
    of the input — otherwise generation continues from a pad token and
    produces garbage. This is the #1 footgun of batched LM eval.
    """
    import torch
    # Render each example through the chat template, get plain text
    texts = []
    for u in users:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": u}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True))
    # Left-pad as a single batch
    old_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=512)
    tok.padding_side = old_side
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    # Slice off the prompt portion per-row and decode
    prompt_len = input_ids.shape[1]
    return [tok.decode(out[i, prompt_len:], skip_special_tokens=True)
            for i in range(out.shape[0])]


# ---------- Eval modes ----------

YES_RE = re.compile(r"\b(yes|true|is|descendant|ancestor)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|not|isn'?t|false)\b", re.IGNORECASE)


def parse_freeform_yesno(text: str) -> str:
    """Best-effort yes/no extraction from a non-tool LM response."""
    if NO_RE.search(text):
        return "no"
    if YES_RE.search(text):
        return "yes"
    return "no"


def eval_condition(name: str, model, tok, device, eval_traces, use_tool: bool, system: str,
                   batch_size: int = 16):
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_): return it
    correct = 0
    well_formed = 0
    by_hop = defaultdict(lambda: [0, 0])

    # Group traces into batches; generate in parallel.
    n = len(eval_traces)
    batches = [eval_traces[i:i + batch_size] for i in range(0, n, batch_size)]
    bar = tqdm(batches, desc=name, leave=False)

    for batch in bar:
        users = [render_user_msg(t) for t in batch]
        resps = generate_batch(model, tok, device, system, users)
        for trace, resp in zip(batch, resps):
            if use_tool:
                ans, valid = route_response(trace["graph"], resp)
                if valid:
                    well_formed += 1
                    pred = ans
                else:
                    pred = parse_freeform_yesno(resp)
            else:
                pred = parse_freeform_yesno(resp)
            ok = (pred == trace["gold_answer"])
            correct += ok
            h = trace.get("hops", trace.get("rule_type", "all"))
            by_hop[h][0] += ok
            by_hop[h][1] += 1
    n = len(eval_traces)
    print(f"\n  [{name}] accuracy: {correct}/{n} = {100*correct/n:.1f}%")
    if use_tool:
        print(f"  [{name}] well-formed tool calls: {well_formed}/{n} = {100*well_formed/n:.1f}%")
    print(f"  [{name}] by hop:")
    for h in sorted(by_hop):
        r, t = by_hop[h]
        print(f"    hops={h}: {r}/{t} = {100*r/t:.1f}%")
    return {"acc": correct / n, "well_formed": well_formed / n if use_tool else None,
            "by_hop": {h: by_hop[h] for h in by_hop}}


# ---------- Driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=str(HERE / "exp60_data" / "lora_adapter"))
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; eval base LM only (condition A).")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training AND skip (A); load saved adapter from --out "
                         "and run only (B) + (C). For tight eval-iteration loops.")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--eval-file", default="eval.jsonl",
                    help="Filename for the eval set within --data-dir.")
    ap.add_argument("--train-file", default="train.jsonl",
                    help="Filename for the train set within --data-dir.")
    ap.add_argument("--data-dir", default=str(DATA),
                    help="Directory containing train/eval JSONL files. "
                         "Use experiments/exp76_data/ for the rule-chain SFT.")
    ap.add_argument("--batch-size", type=int, default=12,
                    help="Per-device train batch size. Bump up on bigger GPUs.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable gradient checkpointing (faster if VRAM allows).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train = load_traces(data_dir / args.train_file)
    evald = load_traces(data_dir / args.eval_file)[: args.n_eval]
    print(f"exp60d: SFT + tool-routed eval ({len(train)} train, {len(evald)} eval)")

    out_dir = Path(args.out)
    if not (args.skip_train or args.eval_only):
        print("\n[1/2] LoRA SFT...")
        train_lora(args.model, train, out_dir, args.epochs, args.lr,
                   batch_size=args.batch_size, grad_ckpt=not args.no_grad_ckpt)

    print("\n[2/2] Evaluation...")

    if args.eval_only:
        if not out_dir.exists():
            raise SystemExit(f"--eval-only set but no adapter at {out_dir}; run a training pass first.")
        A = None
    else:
        print("\n--- (A) base LM, no tool ---")
        base_model, tok, device = load_for_inference(args.model, adapter_dir=None)
        A = eval_condition("A:base", base_model, tok, device, evald,
                           use_tool=False, system="Answer yes or no.")
        del base_model

    if args.skip_train and not args.eval_only:
        return

    print("\n--- (B) SFT'd LM, no tool (forced freeform) ---")
    sft_model, tok, device = load_for_inference(args.model, adapter_dir=out_dir)
    B = eval_condition("B:sft-no-tool", sft_model, tok, device, evald,
                       use_tool=False, system="Answer yes or no.")

    print("\n--- (C) SFT'd LM + TL tool harness ---")
    C = eval_condition("C:sft+tool", sft_model, tok, device, evald,
                       use_tool=True, system=SYSTEM_PROMPT)

    print("\n=== Summary ===")
    if A is not None:
        print(f"  (A) base, no tool       : {100*A['acc']:.1f}%")
    print(f"  (B) SFT, no tool        : {100*B['acc']:.1f}%")
    print(f"  (C) SFT + tool          : {100*C['acc']:.1f}%")
    print(f"  tool-call validity (C)  : {100*C['well_formed']:.1f}%")

    # Falsification check (per IDEAS.md). Skips ratio gate without (A).
    deep_C = sum(C["by_hop"].get(h, [0, 0])[0] for h in [3, 4, 5])
    deep_C_n = sum(C["by_hop"].get(h, [0, 0])[1] for h in [3, 4, 5]) or 1
    c_acc = deep_C / deep_C_n
    print(f"\n  deep-hop (3-5) tool   : {100*c_acc:.1f}%")
    if A is not None:
        deep_A = sum(A["by_hop"].get(h, [0, 0])[0] for h in [3, 4, 5])
        deep_A_n = sum(A["by_hop"].get(h, [0, 0])[1] for h in [3, 4, 5]) or 1
        a_acc = deep_A / deep_A_n
        ratio = c_acc / max(a_acc, 1e-6)
        print(f"  deep-hop (3-5) base   : {100*a_acc:.1f}%")
        print(f"  ratio C/A             : {ratio:.2f}× "
              f"({'PASS ≥1.5×' if ratio >= 1.5 else 'FAIL — falsified'})")
    print(f"  tool-call validity    : "
          f"{'PASS ≥95%' if C['well_formed'] >= 0.95 else 'FAIL — falsified'}")


if __name__ == "__main__":
    main()
