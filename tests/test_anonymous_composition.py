from tensor_logic.research.anonymous_composition import (
    run_anonymous_composition_benchmark,
)


def _small_benchmark():
    return run_anonymous_composition_benchmark(
        train_frames=120,
        induction_frames=50,
        heldout_frames=50,
        train_seed=11,
        induction_seed=22,
        heldout_seed=44,
        random_seed=99,
        admission_threshold=0.70,
    )


def test_oracle_recovers_anonymous_two_hop_rule():
    result = _small_benchmark()
    oracle = result["conditions"]["oracle"]

    assert oracle["accepted"] is True
    assert oracle["heldout_f1"] >= 0.95
    assert oracle["candidate_semantics_evaluator_only"] == [
        "left_of",
        "left_of",
    ]
    assert all(
        relation.startswith("z_rel_")
        for relation in oracle["candidate"]
    )


def test_random_predicates_do_not_pass_same_gate():
    result = _small_benchmark()
    random = result["conditions"]["random"]

    assert random["accepted"] is False
    assert random["heldout_f1"] < 0.70


def test_false_rule_is_rejected_with_counterexample():
    result = _small_benchmark()
    false_rule = result["false_rule_control"]

    assert false_rule["accepted"] is False
    assert false_rule["counterexample"] is not None
    assert false_rule["heldout_f1"] < 0.70


def test_learned_conditions_never_expose_semantic_names_to_candidate():
    result = _small_benchmark()

    for name in ("pca", "kmeans"):
        candidate = result["conditions"][name]["candidate"]
        if candidate is None:
            continue
        assert all(
            relation.rstrip("^T").startswith("z_rel_")
            for relation in candidate
        )
        assert result["conditions"][name][
            "candidate_semantics_evaluator_only"
        ] is None


def test_composition_benchmark_is_fixed_seed_reproducible():
    first = _small_benchmark()
    second = _small_benchmark()

    assert first == second
