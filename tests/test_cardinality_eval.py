from tensor_logic.cardinality_eval import (
    cardinality_degradation,
    evaluate_by_option_count,
)
from tensor_logic.decision_benchmark import DecisionResult
from tensor_logic.system1_tasks import generate_cardinality_pairs


def _perfect_results(cases):
    rows = []
    for case in cases:
        question = case.questions[0]
        target = case.targets[question.question_id]
        probabilities = {
            label: (1.0 if label == target else 0.0)
            for label in question.labels
        }
        rows.append(
            DecisionResult(
                case_id=case.case_id,
                question_id=question.question_id,
                primitive="choice",
                labels=question.labels,
                prediction=target,
                probabilities=probabilities,
                confidence=1.0,
                latency_ms=1.0,
                backend="oracle",
                model="oracle",
                model_version="1",
            )
        )
    return rows


def test_cardinality_metrics_group_exact_option_counts():
    pairs = generate_cardinality_pairs(
        seed=61,
        option_counts=(2, 8, 32),
        per_count=3,
    )
    cases = [pair.structured for pair in pairs]
    results = _perfect_results(cases)

    grouped = evaluate_by_option_count(cases, results)

    assert tuple(row.option_count for row in grouped) == (2, 8, 32)
    assert all(row.cases == 3 for row in grouped)
    assert all(row.metrics.accuracy == 1.0 for row in grouped)


def test_cardinality_degradation_is_relative_to_smallest_choice_space():
    pairs = generate_cardinality_pairs(
        seed=67,
        option_counts=(2, 4),
        per_count=2,
    )
    cases = [pair.structured for pair in pairs]
    results = _perfect_results(cases)

    # Force one 4-way result wrong.
    wrong = results[-1]
    target = cases[-1].targets["source_choice"]
    alternate = next(label for label in wrong.labels if label != target)
    probabilities = {
        label: (1.0 if label == alternate else 0.0)
        for label in wrong.labels
    }
    results[-1] = DecisionResult(
        case_id=wrong.case_id,
        question_id=wrong.question_id,
        primitive=wrong.primitive,
        labels=wrong.labels,
        prediction=alternate,
        probabilities=probabilities,
        confidence=1.0,
        latency_ms=1.0,
        backend="oracle",
        model="damaged",
        model_version="1",
    )

    grouped = evaluate_by_option_count(cases, results)
    delta = cardinality_degradation(grouped)

    assert delta[2] == 0.0
    assert delta[4] == -0.5


def test_nonchoice_result_is_rejected():
    pairs = generate_cardinality_pairs(
        seed=71,
        option_counts=(2,),
        per_count=1,
    )
    case = pairs[0].structured
    question = case.questions[0]
    result = DecisionResult(
        case_id=case.case_id,
        question_id=question.question_id,
        primitive="score",
        labels=question.labels,
        prediction=case.targets[question.question_id],
        probabilities=case.target_distributions[question.question_id],
        confidence=0.8,
        latency_ms=1.0,
        backend="bad",
        model="bad",
        model_version="1",
    )

    try:
        evaluate_by_option_count([case], [result])
    except ValueError as exc:
        assert "choice results" in str(exc)
    else:
        raise AssertionError("non-choice result should fail")
