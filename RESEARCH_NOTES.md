# Research Notes

Dated narrative log. Not a polished doc — a journal. Most recent at the top.

Format: each entry has **Date / Session focus / What we tried / What worked / What surprised us / What we'd do next**.

The point is to make session-to-session continuity possible. If you forget what you were doing, this is the first file you read.

---

## 2026-04-25 — Synthesis & infrastructure

**Session focus:** Stop and figure out what we actually know after 28 experiments + 6 outsourced PRs. Set up real research infrastructure.

**What we tried:**
- Reviewed all 28 exps; categorized into confirmed/falsified/superseded.
- Merged PR #2 (M_compose asymmetric composition fix) and cherry-picked `object_permanence.py` from PR #4. Closed #1, #3, #5, #6 as redundant.
- Built `EXPERIMENTS.md`, `IDEAS.md`, `RESEARCH_NOTES.md` (this file).

**What worked:**
- The honest debrief produced four genuinely confirmed findings (tensor-logic-as-primitive, symbolic > bilinear composition, generative replay > EWC, semantic > random embeddings) and four genuinely refuted approaches (sigmoid-floor, T-annealing alone, naive curiosity, half-rules).
- Cherry-picking + closing redundant PRs was the right call — only 6 lines and 1 file out of ~600 lines were keepers from autonomous agents.

**What surprised us:**
- How little of the autonomous agents' work was usable (~1%) — they're great at coding, terrible at not duplicating each other.
- How many of our exps had no clear hypothesis going in (about half). The ones that did produced sharp results; the ones that didn't produced "interesting" but unactionable observations.

**What we'd do next:**
- **exp29 (log-odds tensor logic).** CPU, hours. Highest leverage / lowest cost item on the backlog. Should retroactively fix exp1, exp5, exp6, exp9.
- After exp29: decide between **exp30 (surprise-gated update)** and **exp31 (composed-attention probe in GPT-2)**. The probe is more exciting but needs Colab; the surprise-gated update stays on CPU.

---

## How to add a session entry

Top of this file. Use the same six fields. Be honest in "What surprised us" — that's the most valuable signal for future-you. If nothing surprised you, the session was probably scripted, not exploratory, and you should ask whether you're still doing research or just executing.

## How this connects to the other docs

- `EXPERIMENTS.md` is the table — one row per run, terse.
- `IDEAS.md` is the backlog — one entry per untried direction, with cost/falsifiability.
- `RESEARCH_NOTES.md` (this) is the narrative — what we were thinking, what shifted.

Read order for resuming work: this file (latest entry) → `EXPERIMENTS.md` (most recent rows) → `IDEAS.md` (top of high-leverage section) → start working.
