"""
exp77: zero-shot schema-aware rule construction.

Question: can a base 0.5B instruct LM construct a valid TL rule for a
question in a domain it's never seen (not kinship), given only the relation
schema in the prompt?

This is the baseline before any rule-construction SFT. Three outcomes:
  A) Model emits a valid, correct rule  → latent capability, SFT not needed
  B) Model emits a valid but wrong rule → it understands TL syntax but not semantics
  C) Model emits no rule / malformed    → SFT on rule construction is required

KB: small professional graph (works_at, manages, has_skill, collaborates_on).
Questions require 2-3 hop joins over relations NOT in the kinship training set.

Usage:
  python exp77_schema_rule_construction.py                  # base model
  python exp77_schema_rule_construction.py --model <path>   # any HF model
"""

import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensor_logic import parse_rule, evaluate_rule, query_relation

# ---------------------------------------------------------------------------
# Knowledge base: professional graph
# ---------------------------------------------------------------------------

KB = {
    "works_at": [
        ["alice", "acme"],
        ["bob", "acme"],
        ["carol", "acme"],
        ["dave", "initech"],
        ["eve", "initech"],
        ["frank", "acme"],
    ],
    "manages": [
        ["alice", "bob"],
        ["alice", "carol"],
        ["alice", "frank"],
        ["dave", "eve"],
    ],
    "has_skill": [
        ["bob", "ml"],
        ["carol", "ml"],
        ["carol", "infra"],
        ["frank", "infra"],
        ["eve", "ml"],
        ["alice", "strategy"],
    ],
    "collaborates_on": [
        ["alice", "bob"],
        ["bob", "carol"],
        ["alice", "carol"],
        ["dave", "eve"],
    ],
}

SCHEMA_DESC = """\
Available relations (all binary: first arg → second arg):
  works_at(Person, Company)       — person is employed at company
  manages(Person, Person)         — first person directly manages second
  has_skill(Person, Skill)        — person has this skill
  collaborates_on(Person, Person) — two people work together on a project

People: alice, bob, carol, dave, eve, frank
Companies: acme, initech
Skills: ml, infra, strategy
"""

# ---------------------------------------------------------------------------
# Test cases: (question, ground truth checks)
# Each check is (subject, object, expected: bool)
# ---------------------------------------------------------------------------

CASES = [
    {
        "question": "Who does Alice manage that has machine learning skills?",
        "hint": "You need to find people Alice manages who also have the skill 'ml'.",
        "example_rule_head": "managed_with_skill(X, Y)",
        "example_rule_body": "manages(X, Z), has_skill(Z, Y)",
        # checks: alice manages bob+carol, both have ml; frank has infra not ml
        "checks": [
            ("alice", "ml", True),   # alice manages someone with ml
        ],
        "rule_head_rel": "managed_with_skill",
    },
    {
        "question": "Do Alice and Carol work at the same company?",
        "hint": "You need to check if two people share a company.",
        "example_rule_head": "same_company(X, Y)",
        "example_rule_body": "works_at(X, Z), works_at(Y, Z)",
        "checks": [
            ("alice", "carol", True),
            ("alice", "dave", False),
        ],
        "rule_head_rel": "same_company",
    },
    {
        "question": "Is there a manager at Acme who has infrastructure skills?",
        "hint": "Find people who manage someone and also have the 'infra' skill, at acme.",
        "example_rule_head": "acme_manager_with_skill(X, Y)",
        "example_rule_body": "works_at(X, acme), manages(X, Z), has_skill(X, Y)",
        "checks": [
            ("alice", "strategy", True),  # alice works at acme, manages people, has strategy
            ("frank", "infra", False),    # frank has infra but doesn't manage anyone — excluded
            ("bob", "ml", False),         # bob has ml but doesn't manage anyone — excluded
        ],
        "rule_head_rel": "acme_manager_with_skill",
    },
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = """\
You are a reasoning assistant with access to a Tensor Logic (TL) substrate.
The substrate executes relational rules expressed as Datalog-style TL tags.

To answer a question about the knowledge base, emit a single TL rule tag:
  <tl_rule head="relation_name(X, Y)" body="rel1(X, Z), rel2(Z, Y)"></tl_rule>

Rules:
- Variables are uppercase (X, Y, Z, P, Q, ...)
- Constants (specific names) are lowercase and quoted if needed
- The head defines a NEW derived relation
- The body chains existing relations from the schema
- Emit ONLY the tag, nothing else
"""

def make_prompt(case: dict) -> str:
    return f"""{SCHEMA_DESC}
Question: {case["question"]}
Hint: {case["hint"]}

Emit a <tl_rule> tag to derive the answer. Use only relations from the schema above."""


# ---------------------------------------------------------------------------
# Rule extraction + evaluation
# ---------------------------------------------------------------------------

RULE_RE = re.compile(
    r'<tl_rule\s+head="([^"]+)"\s+body="([^"]+)"\s*></tl_rule>'
)

def run_rule_against_kb(tag_text: str, kb: dict) -> dict:
    rule = parse_rule(tag_text)
    if rule is None:
        return {"valid": False, "error": "parse failed", "rule": None, "result": None}
    result, err = evaluate_rule(kb, rule)
    if err:
        return {"valid": False, "error": err, "rule": rule, "result": None}
    return {"valid": True, "error": None, "rule": rule, "result": result}


def run_checks(result_dict: dict, checks: list) -> list:
    outcomes = []
    if not result_dict["valid"] or result_dict["result"] is None:
        return [{"check": c, "got": None, "pass": False} for c in checks]
    result = result_dict["result"]
    for subj, obj, expected in checks:
        got = query_relation(result, subj, obj)
        outcomes.append({"check": (subj, obj, expected), "got": got, "pass": got == expected})
    return outcomes


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_model(model_name: str, prompt: str, system: str) -> str:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("transformers not available — returning empty string")
        return ""

    print(f"  Loading {model_name} ...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    print("done")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )
    generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip model, use the gold rule to verify harness works")
    args = ap.parse_args()

    print("=" * 72)
    print(f"exp77: zero-shot schema-aware rule construction")
    print(f"model : {args.model}")
    print(f"mode  : {'dry-run (gold rules)' if args.dry_run else 'live inference'}")
    print("=" * 72)
    print()

    n_valid = 0
    n_checks_pass = 0
    n_checks_total = 0

    for i, case in enumerate(CASES, 1):
        print(f"--- Case {i}: {case['question']}")

        if args.dry_run:
            # Use gold rule to verify the harness itself is wired correctly
            tag = f'<tl_rule head="{case["example_rule_head"]}" body="{case["example_rule_body"]}"></tl_rule>'
            raw = tag
        else:
            prompt = make_prompt(case)
            print(f"  Prompting model...")
            raw = run_model(args.model, prompt, SYSTEM)

        print(f"  Model output: {repr(raw[:200])}")

        # Extract tag if buried in prose
        m = RULE_RE.search(raw)
        tag_text = m.group(0) if m else raw

        result_dict = run_rule_against_kb(tag_text, KB)

        if not result_dict["valid"]:
            print(f"  Rule: INVALID — {result_dict['error']}")
        else:
            n_valid += 1
            print(f"  Rule: valid — {result_dict['rule']}")

        checks = run_checks(result_dict, case["checks"])
        for c in checks:
            subj, obj, expected = c["check"]
            status = "PASS" if c["pass"] else "FAIL"
            print(f"  Check {subj}→{obj} (expected {expected}): {status}")
            n_checks_total += 1
            if c["pass"]:
                n_checks_pass += 1

        print()

    print("=" * 72)
    print(f"Rules valid  : {n_valid}/{len(CASES)}")
    print(f"Checks pass  : {n_checks_pass}/{n_checks_total}")
    print()
    if n_valid == 0:
        print("Outcome C: model emits no valid rules → SFT on rule construction required")
    elif n_checks_pass == 0:
        print("Outcome B: valid syntax but wrong semantics → model knows TL format, not composition")
    elif n_checks_pass == n_checks_total:
        print("Outcome A: all checks pass → latent rule-construction capability exists")
    else:
        print(f"Outcome B/A: partial — {n_checks_pass}/{n_checks_total} correct")


if __name__ == "__main__":
    main()
