# Roadmap: From fafsa-engine local repo to credible artifact

**Date:** 2026-04-28
**Goal:** Get from "I built a thing" to "a stranger can use it, a technical reviewer takes it seriously, and the pattern is instantiated more than once."

---

## Honest current state (updated 2026-04-29)

**Built and working:**
- `~/projects/fafsa-engine/` exists as a clean repo, pushed to `github.com/jwalin-shah/fafsa-engine`
- Unit tests pass; `demo.py` runs end-to-end with Ollama
- LLM backends: Ollama (default), Claude, OpenAI, MLX — interface clean, tests mocked
- **Verifier validates against 42 real ED test ISIR records** (`fafsa/isir.py` parses `data/IDSA25OP-20240308.txt` from `usedgov/fafsa-test-isirs-2024-25`). Component-level checks: parent contribution schedule, SAI summation integrity, IPA table membership.
- **`verify()` and README are honest** about what's validated and what isn't — explicit caveat that arbitrary inputs aren't independently checked against ED, only the engine's components are.
- `tests/test_isir_validation.py` enforces all 42 records pass; engine fails closed.

**Phase 0 is essentially DONE.** The original roadmap was written assuming the verifier was a tautology. That's no longer the case. The honest claim — "validated against 42 ED-published test ISIR records, component-level" — is in the code, the verifier output, and the README.

**Remaining gaps:**
1. **End-to-end input → SAI worked examples not yet validated.** `isir.py` header openly calls this "future work." Could be closed by encoding 3-5 EFC Formula Guide PDF worked examples as test cases. Nice-to-have, not deploy-blocking.
2. **Default model isn't pulled locally.** `ollama pull qwen3.5:4b` is a one-time user action.
3. **No public URL.** Phase 2 still undone.
4. **No post.** Phase 3 still undone.
5. **Single domain.** Phase 4 still undone.

**Recommended order from 2026-04-29 forward:** Phase 1 → Phase 2 → Phase 3 → (optionally close Phase 0 gap with EFC Formula Guide examples in parallel) → Phase 4.

---

## Phases

### Phase 0 — Fix the verifier ✅ DONE (different path than originally planned)

**What got built:** Instead of EFC Formula Guide PDF worked examples, the engine validates against the **2024-25 ED test ISIR file** (`data/IDSA25OP-20240308.txt`, 42 Formula A dependent records). Component-level checks confirm parent contribution schedule, SAI summation, and IPA table all agree with ED's own test data.

**What this proves:** the engine matches federal ground truth on the test data ED publishes for system implementers. That's a verifiable claim and the README states it precisely.

**Phase 0.5 (optional, ~1 day) — Close the end-to-end gap:**

ISIR validation is component-level, not full input → SAI. To strengthen the claim:
1. Pull the 2024-25 EFC Formula Guide PDF from ifap.ed.gov (federal publication, free).
2. Find 3-5 worked examples (full case studies with all input fields and the official SAI value).
3. Encode each as `(DependentFamily, expected_sai)` in `tests/test_ed_published_cases.py`.
4. README claim upgrades to: "validated against 42 ED test ISIRs (component-level) + N EFC Formula Guide worked examples (end-to-end)."

Not deploy-blocking. Do it before posting if the PDF is easy to grab; skip it otherwise.

---

### Phase 1 — Make the demo good locally (~half day)

1. `ollama pull qwen3.5:4b` (one-time, ~3GB download)
2. Re-run `demo.py "My parents make $80k, family of 4"` — narration should be tight, professional, ~3 sentences
3. Record a screen capture or copy the output as the canonical demo block in README

**What this proves:** end-to-end pipeline works at quality level you'd actually show someone.

**Time:** half day, mostly download.

---

### Phase 2 — Deploy to a public URL (~1 day)

Pick one:

| Option | Pros | Cons |
|---|---|---|
| **Modal** | Python-native, supports Ollama-on-GPU or Claude API, auto-scaling | $0.20–$1/hr GPU when used; $0 idle |
| **HF Spaces** | Free, simple Gradio UI, recognized URL | CPU only on free tier — slow narration |
| **Fly.io** | Long-lived service, full control | Most setup |

Recommendation: **Modal with Claude backend** (no GPU needed since LLM is API-side, $0 when idle, ~$0.001/query). Fall back to HF Spaces with Ollama-on-CPU if cost matters.

**Steps:**
1. Add `app.py` with a single function: input string → JSON {facts, proof_steps, narration, verification}
2. Add a thin Gradio UI or just a minimal HTML form
3. Deploy command in README
4. Test with 5 sample queries

**What this proves:** anyone with the URL can use it. This is the actual artifact people share.

**Time:** 1 day.

---

### Phase 3 — Write the post (~half day)

You already have `POST_linkedin.md` and `POST_twitter_thread.md` templates in `tensor/`. Adapt for fafsa-engine specifically.

**Structure:**
- Lede: "An LLM will tell you your SAI. This will prove it."
- 2-sentence problem framing (LLMs hallucinate; FAFSA mistakes are expensive)
- Demo screenshot/GIF (proof tree + ED citations highlighted)
- Link to URL
- Link to GitHub
- One paragraph on the architecture (proof engine = source of truth, LLM = language layer, swap-the-model story)
- One paragraph on what this generalizes to (tax credits, Medicaid, visas)

Ship to LinkedIn + X + HN Show. ~1000 chars total per platform.

**What this proves:** you can communicate the work to a non-expert audience. This is what recruiters/founders actually see.

**Time:** half day.

---

### Phase 4 — Second domain: tax credits (~1 week)

This is what makes "regulatory engine" credible vs. "I built a FAFSA calculator."

**Approach: refactor first, then implement.**

1. **Refactor fafsa-engine** to separate engine from domain (~1 day):
   - `engine/` — generic: trace dataclass, narration interface, validation pattern, LLM backends
   - `domains/fafsa/` — FAFSA-specific: rules, fields, ED citations, published cases
   - `domains/tax_credits/` — new
2. **Pick one tax credit, narrow scope** (~1 day): American Opportunity Tax Credit (AOTC) is small, popular, has clear IRS published examples in Pub 970. Don't try to do all of Pub 970 — one credit, end-to-end.
3. **Build `domains/tax_credits/`** (~3 days):
   - Encode AOTC rules as Python arithmetic with IRS citations
   - Pull 3-5 IRS published worked examples for verification
   - Wire to same demo.py / app.py (now domain-pluggable)
4. **Update deployed app** to support both domains via dropdown (~half day)

**What this proves:** the architecture isn't a one-off. The pattern instantiates on a second federal-regulatory domain in a week, with the engine unchanged. This is the moment hireability turns into "potential founder."

**Time:** 1 week.

**Data-collection side effect (do this while encoding, costs ~0 extra time):**

Every time you encode a domain, log the trace as a training-corpus row:

```jsonl
{
  "domain": "aotc",
  "regulation_source": "IRS Pub 970, §American Opportunity Credit",
  "regulation_text": "<verbatim PDF excerpt of the rule>",
  "worked_examples": [
    {"inputs": {...}, "expected_output": 2500, "source": "Pub 970 Example 1"},
    ...
  ],
  "encoded_rule": "<TL DSL or Python source of the rule>",
  "verified_against_n_cases": 5,
  "encoding_time_hours": 18
}
```

Store in `domains/<name>/induction_corpus.jsonl`. One row per rule, not one per domain — a domain like AOTC may produce 5–10 rows.

This is the **seed dataset for the induction transformer** (Phase 6+). After 3 domains you'll have ~30 rows, after 10 domains ~100. Not enough to train, but enough to:
- Run the existing exp78 LM-proposer against and measure baseline accuracy
- Identify which rule shapes are easy vs. hard for an LM to induce
- Decide whether to invest in a trained proposer or stay LM-only

**Verification:** corpus rows are well-formed JSON; loader script in `tools/load_induction_corpus.py` parses all domains.

---

### Phase 5 — Decide: keep going on applications, or pivot to research bets

After Phase 4 you have evidence about whether the pattern is fast to instantiate or slow. Use that evidence.

**If second domain was easy** (3 days + 2 days fluff):
- Keep going. Third domain (Medicaid eligibility, simpler). Stay on the application/startup track.
- At 3 domains, write a longer post + cold-email YC, applied AI funds, regulatory tech buyers.
- Total time-to-decision: ~2-3 weeks.

**If second domain was hard** (the rules don't fit the pattern, validation is brittle, narration breaks):
- That's important data. The pattern works for FAFSA-shape problems but not all of regulation.
- Pivot to research bets that strengthen the *capability claim*, not the application surface.
- Priority order: exp79 self-play loop → open-world TL-as-tool probe → exp61 Hyperloop.

Either way: **don't start research bets before you have a deployed, verified, posted artifact**. Research bets pay off on a 6-18 month horizon. Hireability/fundability today is what the artifact-and-post path gives you.

---

## Phase 6+ — Scaling out: induction transformer (research bet, ~1-3 months)

**Only attempt after Phase 4 confirms pattern works AND you have ~30+ corpus rows.**

The thesis: manual encoding is 1 human-week per domain. That doesn't scale to 50 domains. An induction transformer trained on `(regulation_text + worked_examples) → encoded_rule` triples turns each new domain from human-week to verification-pass.

**Stages:**

1. **LM proposer baseline** (~3 days): Run exp78's LM-prune stack against the corpus. Input: regulation_text + worked_examples. Output: candidate rule in TL DSL. Verify via the same `verify()` against worked examples. Measure: % of rules a frontier LM can induce zero-shot.
2. **Synthetic corpus expansion** (~1 week): Use exp81's sweep to generate synthetic regulatory-shape rule families (bracketed allowances, threshold formulas, conditional credits). Train data = real corpus + synthetic. AutomataGPT lesson: small models trained over many synthetic worlds learn the rule-space prior.
3. **Train small TL transformer** (~2 weeks, MLX-trainable): 50–200M params, decoder-only, trained on synthetic + real corpus. Output = TL DSL rule. Plug into fafsa-engine as proposer.
4. **Human-in-the-loop deploy** (~1 week): proposer suggests rule → engine verifies against published cases → human reviews → ship. Goal: domain encoding time drops from 1 week to 1 day.

**What this proves:** the regulatory-engine pattern scales. Now "I built infrastructure for encoding any federal/state regulation" is defensible because the encoding cost per domain is bounded.

**Risk:** LM proposer baseline is bad enough that even a trained model can't recover. Mitigation: that's also a finding — means regulatory rules need richer structure than current TL DSL supports, which is itself a research direction (predicate invention, exp79, etc.).

---

## Speeding up research bets (when we get there)

The reason 80+ experiments feels slow isn't compute — it's that each experiment is a one-off file with no shared evaluator. Speed-up moves:

1. **Single eval harness** — one `experiments/eval_harness.py` that takes a config (rules, families/cases, expected outputs) and runs it. New experiments plug in instead of recreating boilerplate.
2. **Shared model loader cache** — already partially done in `_LM_CACHE`; extend to all experiments.
3. **Stop adding `expN_data/` directories** — store inputs and outputs in one `experiments/data/` with a manifest.
4. **Hooks for failure mode classification** — exp81 has step logging + failure mode classification. Pull that into the harness as default.

These are 1-2 day refactors that pay back across the next 5+ experiments.

---

## Time budget summary

| Phase | Time | Cumulative |
|---|---|---|
| Phase 0 — fix verifier | 1-2 days | ~2 days |
| Phase 1 — pull qwen, polish demo | half day | ~2.5 days |
| Phase 2 — deploy Modal/HF | 1 day | ~3.5 days |
| Phase 3 — write post | half day | ~4 days |
| Phase 4 — tax credits domain | 1 week | ~2 weeks |
| Phase 5 — decide research vs. third domain | — | — |

**End of week 2: deployed verified URL + post + pattern instantiated twice.** That's the threshold for "this is real."

---

## Risks and where it could fail

1. **ED Formula Guide PDF doesn't have machine-readable worked examples** — risk: have to hand-extract from prose. Mitigation: a few hours of careful reading; this is the actual real-validation step and is non-skippable.
2. **The encoded rules disagree with ED published cases** — risk: a rule was extracted wrong (bracket boundary, allowance table, rounding). Mitigation: this is exactly what Phase 0 finds. If it happens, fix the rule and the failing case becomes a regression test.
3. **Modal/HF deploy hits CORS/auth/rate-limit issues** — risk: Phase 2 takes 2 days not 1. Mitigation: HF Spaces fallback is well-trodden.
4. **Tax credits don't fit the same shape** — risk: Pub 970 rules involve non-arithmetic conditions that don't translate cleanly. Mitigation: pick AOTC specifically (well-bounded, formula-driven). If it doesn't fit, that's signal — Phase 5 pivots accordingly.

---

## What this plan deliberately does NOT do

- No exp79 self-play loop yet (research, not deliverable)
- No exp61 Hyperloop yet (multi-month research bet)
- No predicate invention work
- No new tensor_logic features
- No additional substrate experiments
- No academic paper draft
- No rewriting tensor_logic for performance

These are all defensible later. They are not what makes the artifact credible today.
