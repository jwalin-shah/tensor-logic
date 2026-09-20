from tensor_logic.decision_benchmark import DecisionResult
from tensor_logic.paired_bootstrap import paired_bootstrap_report
from tensor_logic.system1_tasks import generate_pairs


def _results_for_pairs(pairs, *, structured_better: bool):
    raw = []
    structured = []

    for pair in pairs:
        for raw_q, structured_q in zip(
            pair.raw.questions,
            pair.structured.questions,
        ):
            target = pair.raw.targets[raw_q.question_id]

            def make(case_id, question, good):
                labels = question.labels
                if len(labels) == 2:
                    wrong = next(label for label in labels if label != target)
                else:
                    wrong = next(label for label in labels if label != target)

                prediction = target if good else wrong
                probabilities = {}
                for label in labels:
                    if good:
                        probabilities[label] = (
                            0.8
                            if label == target
                            else 0.2 / (len(labels) - 1)
                        )
                    else:
                        probabilities[label] = (
                            0.2
                            if label == target
                            else (
                                0.8
                                if label == wrong
                                else 0.0
                            )
                        )
                total = sum(probabilities.values())
                probabilities = {
                    label: value / total
                    for label, value in probabilities.items()
                }
                return DecisionResult(
                    case_id=case_id,
                    question_id=question.question_id,
                    primitive=question.primitive,
                    labels=labels,
                    prediction=prediction,
                    probabilities=probabilities,
                    confidence=probabilities[prediction],
                    latency_ms=1.0,
                    backend="test",
                    model="test",
                    model_version="1",
                )

            raw.append(
                make(
                    pair.raw.case_id,
                    raw_q,
                    good=not structured_better,
                )
            )
            structured.append(
                make(
                    pair.structured.case_id,
                    structured_q,
                    good=True,
                )
            )

    return tuple(raw), tuple(structured)


def test_bootstrap_detects_consistent_structured_advantage():
    pairs = tuple(
        pair
        for pair in generate_pairs(
            seed=7,
            iid_count=40,
            ood_per_family=0,
        )
        if pair.scenario.split == "test"
    )
    raw, structured = _results_for_pairs(
        pairs,
        structured_better=True,
    )

    report = paired_bootstrap_report(
        pairs,
        raw,
        structured,
        split="test",
        iterations=500,
        seed=3,
    )

    by_metric = {
        row.metric: row
        for row in report.right_minus_left
    }

    assert by_metric["accuracy"].mean_delta > 0
    assert by_metric["accuracy"].lower > 0

    assert by_metric["nll"].mean_delta < 0
    assert by_metric["nll"].upper < 0

    assert by_metric["brier"].mean_delta < 0
    assert by_metric["brier"].upper < 0


def test_bootstrap_is_deterministic_for_seed():
    pairs = tuple(
        pair
        for pair in generate_pairs(
            seed=11,
            iid_count=30,
            ood_per_family=0,
        )
        if pair.scenario.split == "test"
    )
    raw, structured = _results_for_pairs(
        pairs,
        structured_better=True,
    )

    left = paired_bootstrap_report(
        pairs,
        raw,
        structured,
        split="test",
        iterations=100,
        seed=9,
    )
    right = paired_bootstrap_report(
        pairs,
        raw,
        structured,
        split="test",
        iterations=100,
        seed=9,
    )

    assert left == right


def test_missing_paired_result_fails_closed():
    pairs = tuple(
        pair
        for pair in generate_pairs(
            seed=13,
            iid_count=20,
            ood_per_family=0,
        )
        if pair.scenario.split == "test"
    )
    raw, structured = _results_for_pairs(
        pairs,
        structured_better=True,
    )

    try:
        paired_bootstrap_report(
            pairs,
            raw[:-1],
            structured,
            split="test",
            iterations=10,
        )
    except ValueError as exc:
        assert "missing paired result" in str(exc)
    else:
        raise AssertionError("missing paired observation should fail")
