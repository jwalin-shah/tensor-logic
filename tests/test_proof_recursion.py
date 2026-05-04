import pytest
import torch
from tensor_logic.program import Program, Domain, Relation, Atom, Rule
from tensor_logic.proofs import prove, prove_negative, fmt_proof_tree

def test_3_hop_proof():
    # skip_skip_manager: X -> Y -> Z -> W
    p = Program()
    p.domain("person", ["a", "b", "c", "d"])
    p.relation("manages", "person", "person")
    p.fact("manages", "a", "b")
    p.fact("manages", "b", "c")
    p.fact("manages", "c", "d")
    
    # skip_skip_manager(X, W) :- manages(X, Y), manages(Y, Z), manages(Z, W)
    rule = Rule(
        Atom("skip_skip_manager", ("X", "W")),
        (
            Atom("manages", ("X", "Y")),
            Atom("manages", ("Y", "Z")),
            Atom("manages", ("Z", "W")),
        )
    )
    p.rules.setdefault("skip_skip_manager", []).append(rule)
    # Also need the relation defined for query to work
    p.relation("skip_skip_manager", "person", "person")
    
    # This should find the proof a -> b -> c -> d
    proof = prove(p, "skip_skip_manager", "a", "d")
    assert proof is not None
    print("\nProof found:")
    print(fmt_proof_tree(proof))
    assert len(proof.body) == 3


def test_3_hop_negative_proof_binds_all_intermediates():
    p = Program()
    p.domain("person", ["a", "b", "c", "d", "e"])
    p.relation("manages", "person", "person")
    p.relation("skip_skip_manager", "person", "person")
    p.fact("manages", "a", "b")
    p.fact("manages", "b", "c")
    p.fact("manages", "c", "d")
    p.rule(
        "skip_skip_manager(X,W) := "
        "manages(X,Y) * manages(Y,Z) * manages(Z,W)"
    )

    neg = prove_negative(p, "skip_skip_manager", "a", "e")
    assert neg is not None
    assert neg.reason in {"rule_body_failed", "all_rules_failed"}
    assert neg.body

if __name__ == "__main__":
    pytest.main([__file__])
