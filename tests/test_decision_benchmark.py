from tensor_logic.decision_benchmark import (
    DecisionCase,
    DecisionQuestion,
    DecisionResult,
    distribution_soft_accuracy,
    evaluate_results,
    option_order_changed,
)


def _cases():
    q = DecisionQuestion(
        question_id="route",
        primitive="choice",
        labels=("fast", "deep"),
    )
    return (
        DecisionCase(
            case_id="c1",
            state={"x": 1},
            questions=(q,),
            targets={"route": "fast"},
            target_distributions={
                "route": {"fast": 0.8, "deep": 0.2}
            },
        ),
        DecisionCase(
            case_id="c2",
            state={"x": 2},
            questions=(q,),
            targets={"route": "deep"},
        ),
        DecisionCase(
            case_id="c3",
            state={"x": 3},
            questions=(q,),
            targets={"route": "deep"},
        ),
    )


def _results():
    return (
        DecisionResult(
            case_id="c1",
            question_id="route",
            primitive="choice",
            labels=("fast", "deep"),
            prediction="fast",
            probabilities={"fast": 0.9, "deep": 0.1},
            confidence=0.9,
            latency_ms=10.0,
            backend="mini",
            model="m",
            model_version="1",
        ),
        DecisionResult(
            case_id="c2",
            question_id="route",
            primitive="choice",
            labels=("fast", "deep"),
            prediction="fast",
            probabilities={"fast": 0.6, "deep": 0.4},
            confidence=0.6,
            latency_ms=20.0,
            backend="mini",
            model="m",
            model_version="1",
        ),
        DecisionResult(
            case_id="c3",
            question_id="route",
            primitive="choice",
            labels=("fast", "deep"),
            prediction="deep",
            probabilities={"fast": 0.2, "deep": 0.8},
            confidence=0.8,
            latency_ms=30.0,
            backend="mini",
            model="m",
            model_version="1",
        ),
    )


def test_common_metrics_and_selective_curve():
    metrics = evaluate_results(
        _cases(),
        _results(),
        ece_bins=2,
        thresholds=(0.5, 0.7, 0.95),
    )

    assert metrics.count == 3
    assert abs(metrics.accuracy - (2 / 3)) < 1e-9
    assert metrics.brier > 0.0
    assert metrics.nll > 0.0
    assert 0.0 <= metrics.ece <= 1.0
    assert metrics.p50_latency_ms == 20.0
    assert metrics.p95_latency_ms > 20.0

    low, mid, high = metrics.selective_curve
    assert low.coverage == 1.0
    assert abs(mid.coverage - (2 / 3)) < 1e-9
    assert mid.accuracy == 1.0
    assert high.coverage == 0.0
    assert high.accuracy is None
    assert high.risk is None


def test_soft_accuracy_uses_probability_overlap():
    case = _cases()[0]
    result = _results()[0]
    value = distribution_soft_accuracy(case, result)

    assert value is not None
    assert abs(value - 0.9) < 1e-9


def test_option_order_robustness_compares_prediction_not_position():
    original = _results()[0]
    same = DecisionResult(
        case_id="c1",
        question_id="route",
        primitive="choice",
        labels=("deep", "fast"),
        prediction="fast",
        probabilities={"deep": 0.1, "fast": 0.9},
        confidence=0.9,
        latency_ms=10.0,
        backend="mini",
        model="m",
        model_version="1",
    )
    changed = DecisionResult(
        case_id="c1",
        question_id="route",
        primitive="choice",
        labels=("deep", "fast"),
        prediction="deep",
        probabilities={"deep": 0.55, "fast": 0.45},
        confidence=0.55,
        latency_ms=10.0,
        backend="mini",
        model="m",
        model_version="1",
    )

    assert option_order_changed(original, same) is False
    assert option_order_changed(original, changed) is True


def test_result_schema_fails_closed_on_missing_probabilities():
    try:
        DecisionResult(
            case_id="c1",
            question_id="route",
            primitive="choice",
            labels=("fast", "deep"),
            prediction="fast",
            probabilities={"fast": 1.0},
            confidence=1.0,
            latency_ms=1.0,
            backend="x",
            model="x",
            model_version="1",
        )
    except ValueError as exc:
        assert "cover exactly" in str(exc)
    else:
        raise AssertionError("missing probability label should fail")


def test_noul_question_has_fixed_binary_labels():
    try:
        DecisionQuestion(
            question_id="q",
            primitive="noul",
            labels=("maybe", "yes"),
        )
    except ValueError as exc:
        assert "noul labels" in str(exc)
    else:
        raise AssertionError("invalid noul labels should fail")
