from tensor_logic.provenance_datalog import (
    provenanced_transitive_closure,
)
from tensor_logic.provenance_semiring import facts_to_relation


def test_recursive_closure_derives_reachability_with_fact_witnesses():
    edges = facts_to_relation(
        [
            ("a", "b", "e:ab"),
            ("b", "c", "e:bc"),
            ("c", "d", "e:cd"),
        ]
    )
    result = provenanced_transitive_closure(edges)

    assert ("a", "d") in result.relation
    proofs = result.relation[("a", "d")].proofs
    assert frozenset({"e:ab", "e:bc", "e:cd"}) in proofs
    assert result.iterations > 0


def test_alternative_paths_remain_alternative_proofs():
    edges = facts_to_relation(
        [
            ("a", "b", "e:ab"),
            ("b", "d", "e:bd"),
            ("a", "c", "e:ac"),
            ("c", "d", "e:cd"),
        ]
    )
    result = provenanced_transitive_closure(edges)

    assert result.relation[("a", "d")].proofs == frozenset(
        {
            frozenset({"e:ab", "e:bd"}),
            frozenset({"e:ac", "e:cd"}),
        }
    )


def test_cycle_converges_on_finite_proof_support():
    edges = facts_to_relation(
        [
            ("a", "b", "e:ab"),
            ("b", "a", "e:ba"),
        ]
    )
    result = provenanced_transitive_closure(edges)

    assert ("a", "a") in result.relation
    assert ("b", "b") in result.relation
