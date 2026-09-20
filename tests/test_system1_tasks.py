from tensor_logic.system1_tasks import (
    SYSTEM1_QUESTIONS,
    feature_vector,
    generate_cardinality_pairs,
    generate_pairs,
)


def test_pair_generator_is_deterministic():
    left = generate_pairs(seed=7, iid_count=12, ood_per_family=3)
    right = generate_pairs(seed=7, iid_count=12, ood_per_family=3)

    assert left == right


def test_raw_and_structured_pair_share_questions_targets_and_teacher_distributions():
    pair = generate_pairs(
        seed=11,
        iid_count=1,
        ood_per_family=0,
    )[0]

    assert pair.raw.questions == pair.structured.questions
    assert pair.raw.questions == SYSTEM1_QUESTIONS
    assert pair.raw.targets == pair.structured.targets
    assert (
        pair.raw.target_distributions
        == pair.structured.target_distributions
    )
    assert pair.raw.case_id.endswith("::raw")
    assert pair.structured.case_id.endswith("::structured")


def test_feature_vector_contains_all_structured_numeric_decision_signals():
    pair = generate_pairs(
        seed=13,
        iid_count=1,
        ood_per_family=0,
    )[0]
    features = feature_vector(pair.scenario)

    assert len(features) == 11
    assert all(0.0 <= value <= 1.0 for value in features)


def test_ood_families_enforce_intended_distribution_shift():
    pairs = generate_pairs(
        seed=17,
        iid_count=0,
        ood_per_family=5,
    )
    scenarios = [pair.scenario for pair in pairs]

    compositional = [
        row for row in scenarios
        if row.split == "ood_compositional"
    ]
    source_failure = [
        row for row in scenarios
        if row.split == "ood_source_failure"
    ]
    confidence_shift = [
        row for row in scenarios
        if row.split == "ood_confidence_shift"
    ]

    assert all(
        row.complexity >= 0.82
        and row.stakes >= 0.82
        and row.source_freshness <= 0.30
        for row in compositional
    )
    assert all(
        row.authority_available is False
        and row.external_search_allowed is False
        and row.source_freshness <= 0.25
        for row in source_failure
    )
    assert all(
        row.confidence >= 0.82
        and row.contradiction >= 0.82
        for row in confidence_shift
    )


def test_targets_are_legal_and_teacher_distributions_normalized():
    pairs = generate_pairs(
        seed=19,
        iid_count=20,
        ood_per_family=2,
    )

    for pair in pairs:
        case = pair.structured
        for question in case.questions:
            target = case.targets[question.question_id]
            distribution = case.target_distributions[
                question.question_id
            ]

            assert target in question.labels
            assert set(distribution) == set(question.labels)
            assert abs(sum(distribution.values()) - 1.0) < 1e-9
            assert max(
                distribution,
                key=distribution.get,
            ) == target


def test_cardinality_sweep_has_requested_option_counts_and_paired_targets():
    pairs = generate_cardinality_pairs(
        seed=23,
        option_counts=(2, 8, 32),
        per_count=2,
    )

    assert len(pairs) == 6
    assert {pair.option_count for pair in pairs} == {2, 8, 32}

    for pair in pairs:
        raw_q = pair.raw.questions[0]
        structured_q = pair.structured.questions[0]

        assert len(raw_q.labels) == pair.option_count
        assert raw_q == structured_q
        assert pair.raw.targets == pair.structured.targets
        assert (
            pair.raw.target_distributions
            == pair.structured.target_distributions
        )


def test_no_personal_identifiers_in_generated_states():
    pairs = generate_pairs(
        seed=29,
        iid_count=5,
        ood_per_family=1,
    )

    joined = "\n".join(
        str(pair.raw.state) + str(pair.structured.state)
        for pair in pairs
    ).lower()

    for forbidden in (
        "@gmail.com",
        "jwalin",
        "fremont",
        "sunnyvale",
        "linkedin.com/in/",
    ):
        assert forbidden not in joined
