"""
exp60c (step 3/4 of the TL-as-tool integration line):
deterministic "fake LM" rule-based baseline.

This is the end-to-end sanity check. We replace the LM with a perfect
rule-based emitter that:
  1. Reads the query and graph from a trace,
  2. Always emits a syntactically-correct <tl_closure ...> tag,
  3. Lets exp60b's harness execute the tag against TL substrate,
  4. Returns the substrate's answer as the final output.

If this rule-based "model" doesn't reach 100% accuracy on exp60a's eval
set, the harness has a bug — and we don't want to discover that during
real LM SFT. This is the "make the test pass before training" check.

Once this hits 100%, the only remaining variable in exp60d is the LM's
ability to learn the tool-call protocol via SFT. The substrate is
proven correct.

Run order:
  1. python exp60a_kinship_traces.py     # generate exp60_data/{train,eval}.jsonl
  2. python exp60b_tl_tool_harness.py    # smoke-test the harness
  3. python exp60c_rule_baseline.py      # this file — end-to-end sanity
  4. (deferred) python exp60d_sft.py     # actual LM SFT
"""

import json
import re
from pathlib import Path

# Import the harness (sibling file) without packaging
import importlib.util
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp60b_harness", HERE / "exp60b_tl_tool_harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


QUERY_RE = re.compile(
    r"Is\s+(?P<subj>\w+)\s+(?:an?\s+)?(?P<rel>ancestor|descendant)\s+of\s+(?P<obj>\w+)\?",
    re.IGNORECASE,
)


def rule_based_emit_tool_call(query_text: str) -> str:
    """Parse the query string and emit the corresponding TL tool call.
    Mimics what an SFT'd LM should learn to do.
    """
    m = QUERY_RE.search(query_text)
    if not m:
        return ""
    return (
        f'<tl_closure relation="parent" '
        f'query="{m.group("rel").lower()}" '
        f'subject="{m.group("subj").lower()}" '
        f'object="{m.group("obj").lower()}">'
        f'</tl_closure>'
    )


def main():
    data_path = HERE / "exp60_data" / "eval.jsonl"
    if not data_path.exists():
        raise SystemExit(
            f"Missing {data_path}. Run exp60a_kinship_traces.py first."
        )

    print("exp60c: rule-based fake-LM end-to-end sanity check")
    print()
    print(f"Eval set: {data_path}")

    by_hops = {}  # hops -> [right, total]
    n_total = 0
    n_right = 0
    n_tool_call_ok = 0

    with data_path.open() as f:
        for line in f:
            ex = json.loads(line)
            graph = ex["graph"]
            query = ex["query"]
            gold = ex["gold_answer"]
            hops = ex["hops"]

            # 1. Rule-based "LM" emits tool call from query
            tool_call_str = rule_based_emit_tool_call(query)

            # 2. Verify tool call is well-formed
            if not tool_call_str or not harness.parse_tool_call(tool_call_str):
                pred = "no"  # malformed → say no
            else:
                n_tool_call_ok += 1
                # 3. Harness executes the tool call against the TL substrate
                results = harness.evaluate_string(graph, tool_call_str)
                pred = results[0]["result"]["answer"] if results else "no"

            n_total += 1
            if pred == gold:
                n_right += 1
                by_hops.setdefault(hops, [0, 0])[0] += 1
            by_hops.setdefault(hops, [0, 0])[1] += 1

    print()
    print(f"Tool calls well-formed: {n_tool_call_ok}/{n_total} ({100*n_tool_call_ok/n_total:.1f}%)")
    print(f"Overall accuracy:       {n_right}/{n_total} ({100*n_right/n_total:.1f}%)")
    print()
    print(f"Accuracy by hop count:")
    print(f"  {'hops':<6}{'right':<8}{'total':<8}{'acc':<8}")
    for hops in sorted(by_hops):
        right, total = by_hops[hops]
        print(f"  {hops:<6}{right:<8}{total:<8}{100*right/total:<8.1f}")

    print()
    if n_right == n_total:
        print("100% — substrate + harness are correctly wired. exp60d (real LM SFT)")
        print("can plug in any instruct LM and the only remaining variable is the LM's")
        print("ability to learn the tool-call protocol.")
    else:
        print(f"{n_total - n_right} failures — bug somewhere in harness or trace gen.")
        print("Show first 3 failures:")
        n_shown = 0
        with data_path.open() as f:
            for line in f:
                ex = json.loads(line)
                tool_call_str = rule_based_emit_tool_call(ex["query"])
                if not tool_call_str:
                    pred = "no"
                else:
                    results = harness.evaluate_string(ex["graph"], tool_call_str)
                    pred = results[0]["result"]["answer"] if results else "no"
                if pred != ex["gold_answer"]:
                    print(f"  query={ex['query']!r}  hops={ex['hops']}  gold={ex['gold_answer']}  pred={pred}")
                    print(f"    tool_call={tool_call_str}")
                    n_shown += 1
                    if n_shown >= 3:
                        break


if __name__ == "__main__":
    main()
