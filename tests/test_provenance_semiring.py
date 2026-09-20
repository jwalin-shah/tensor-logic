from tensor_logic.provenance_semiring import (
    ProofSemiring,
    compose_relation_with_provenance,
    facts_to_relation,
)


def test_semiring_zero_one_and_idempotent_plus():
    a = ProofSemiring.fact("a")

    assert a.plus(ProofSemiring.zero()) == a
    assert a.times(ProofSemiring.one()) == a
    assert a.plus(a) == a
    assert a.times(ProofSemiring.zero()) == ProofSemiring.zero()


def test_times_combines_joint_premises():
    proof = ProofSemiring.fact("a").times(
        ProofSemiring.fact("b")
    )
    assert proof.proofs == frozenset(
        {frozenset({"a", "b"})}
    )


def test_distributivity():
    a = ProofSemiring.fact("a")
    b = ProofSemiring.fact("b")
    c = ProofSemiring.fact("c")

    left = a.times(b.plus(c))
    right = a.times(b).plus(a.times(c))

    assert left == right


def test_minimal_drops_strictly_redundant_supersets():
    value = ProofSemiring(
        frozenset(
            {
                frozenset({"a"}),
                frozenset({"a", "b"}),
                frozenset({"c", "d"}),
            }
        )
    )
    assert value.minimal().proofs == frozenset(
        {
            frozenset({"a"}),
            frozenset({"c", "d"}),
        }
    )


def test_relational_composition_carries_all_minimal_witness_sets():
    attends = facts_to_relation(
        [
            ("p0", "e0", "calendar:e0"),
            ("p0", "e1", "calendar:e1"),
        ]
    )
    topic = facts_to_relation(
        [
            ("e0", "t0", "topic:e0:t0"),
            ("e1", "t0", "topic:e1:t0"),
        ]
    )

    out = compose_relation_with_provenance(attends, topic)
    proofs = out[("p0", "t0")].proofs

    assert proofs == frozenset(
        {
            frozenset({"calendar:e0", "topic:e0:t0"}),
            frozenset({"calendar:e1", "topic:e1:t0"}),
        }
    )


def test_topk_prefers_lower_cost_proofs():
    value = ProofSemiring(
        frozenset(
            {
                frozenset({"a"}),
                frozenset({"b", "c"}),
                frozenset({"d"}),
            }
        )
    )

    selected = value.topk(
        2,
        fact_costs={"a": 5.0, "b": 1.0, "c": 1.0, "d": 0.5},
    )

    assert selected[0] == frozenset({"d"})
    assert selected[1] == frozenset({"b", "c"})
