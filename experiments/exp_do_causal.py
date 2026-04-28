"""
do() causal operator — Pearl level-2 intervention on TL proof engine.

Spec (IDEAS.md):
  prove(grandparent(alice,charlie)) = true
  do(parent(alice,bob), false)
  Re-prove: NegativeProof because causal chain cut, not just fact retracted.

Key property: do(fact, false) severs rule derivations — the intervention
overrides both the stored tensor value AND any rules that could re-derive
the fact. This is Pearl's "cutting incoming edges" in the causal graph.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor_logic import Program, prove, prove_negative, prove_with_do, fmt_proof_tree, fmt_negative_proof_tree


def build_family_kb() -> Program:
    p = Program()
    people = ["alice", "bob", "carol", "charlie", "diana"]
    p.domain("person", people)
    p.relation("parent", "person", "person")
    p.relation("grandparent", "person", "person")
    p.relation("sibling", "person", "person")

    # Facts
    p.fact("parent", "alice", "bob")
    p.fact("parent", "alice", "carol")
    p.fact("parent", "bob", "charlie")
    p.fact("parent", "carol", "diana")
    p.fact("sibling", "bob", "carol")
    p.fact("sibling", "carol", "bob")

    # Rule: grandparent(x,z) via parent chain
    p.rule("grandparent(x,z) := (parent(x,y) * parent(y,z)).step()")

    return p


def run():
    print("exp_do_causal: Pearl level-2 causal intervention")
    print("=" * 60)
    p = build_family_kb()

    # --- Baseline: no intervention ---
    print("\n[baseline] no intervention")
    cases = [
        ("grandparent", "alice", "charlie"),
        ("grandparent", "alice", "diana"),
        ("parent",      "alice", "bob"),
    ]
    for rel, src, dst in cases:
        proof = prove(p, rel, src, dst)
        if proof:
            print(f"  prove({rel}({src},{dst})) = YES")
            print(fmt_proof_tree(proof, indent=2))
        else:
            print(f"  prove({rel}({src},{dst})) = NO")

    # --- Intervention: do(parent(alice,bob), false) ---
    print("\n[do] parent(alice,bob) := false  (cut alice→bob causal edge)")
    do_map = {("parent", "alice", "bob"): 0.0}

    # grandparent(alice,charlie) goes through alice→bob→charlie — should become NO
    proof = prove_with_do(p, "grandparent", "alice", "charlie", do=do_map)
    status = "YES" if proof else "NO"
    expected = "NO"
    mark = "PASS" if status == expected else "FAIL"
    print(f"  [{mark}] prove_with_do(grandparent(alice,charlie)) = {status}  (expected {expected})")

    # grandparent(alice,diana) goes through alice→carol→diana — should remain YES
    proof = prove_with_do(p, "grandparent", "alice", "diana", do=do_map)
    status = "YES" if proof else "NO"
    expected = "YES"
    mark = "PASS" if status == expected else "FAIL"
    print(f"  [{mark}] prove_with_do(grandparent(alice,diana)) = {status}  (expected {expected})")
    if proof:
        print(fmt_proof_tree(proof, indent=4))

    # parent(alice,bob) itself should be severed — NO (intervention overrides stored fact)
    proof = prove_with_do(p, "parent", "alice", "bob", do=do_map)
    status = "YES" if proof else "NO"
    expected = "NO"
    mark = "PASS" if status == expected else "FAIL"
    print(f"  [{mark}] prove_with_do(parent(alice,bob)) = {status}  (expected {expected})")

    # parent(alice,carol) unaffected
    proof = prove_with_do(p, "parent", "alice", "carol", do=do_map)
    status = "YES" if proof else "NO"
    expected = "YES"
    mark = "PASS" if status == expected else "FAIL"
    print(f"  [{mark}] prove_with_do(parent(alice,carol)) = {status}  (expected {expected})")

    # --- do() with positive override: assert a fact that isn't in KB ---
    print("\n[do] parent(bob,diana) := true  (inject synthetic fact)")
    do_inject = {("parent", "bob", "diana"): 1.0}
    proof = prove_with_do(p, "grandparent", "alice", "diana", do=do_inject)
    # Now grandparent(alice,diana) can be proved two ways: alice→carol→diana (real)
    # AND alice→bob→diana (injected). Either path suffices.
    status = "YES" if proof else "NO"
    expected = "YES"
    mark = "PASS" if status == expected else "FAIL"
    print(f"  [{mark}] prove_with_do(grandparent(alice,diana)) = {status}  (expected {expected})")

    # Injected path specifically: grandparent(alice,charlie) still provable via alice→bob→charlie
    proof = prove_with_do(p, "grandparent", "alice", "charlie", do=do_inject)
    status = "YES" if proof else "NO"
    expected = "YES"
    mark = "PASS" if status == expected else "FAIL"
    print(f"  [{mark}] prove_with_do(grandparent(alice,charlie)) = {status}  (expected {expected})")

    # --- Verify existing prove() unchanged (no regression) ---
    print("\n[regression] prove() without _do still works")
    proof = prove(p, "grandparent", "alice", "charlie")
    status = "YES" if proof else "NO"
    mark = "PASS" if status == "YES" else "FAIL"
    print(f"  [{mark}] prove(grandparent(alice,charlie)) = {status}  (expected YES)")

    proof = prove(p, "parent", "alice", "bob")
    status = "YES" if proof else "NO"
    mark = "PASS" if status == "YES" else "FAIL"
    print(f"  [{mark}] prove(parent(alice,bob)) = {status}  (expected YES)")

    print("\n" + "=" * 60)
    print("Pearl level-2 (intervention) landed:")
    print("  do(fact, false) severs derivation chain — fact + downstream unprovable")
    print("  do(fact, true)  injects synthetic fact — downstream derivable")
    print("  Unrelated paths unaffected by intervention")


if __name__ == "__main__":
    run()
