import math

from tensor_logic.decision_benchmark import (
    DecisionCase,
    DecisionQuestion,
    DecisionResult,
    evaluate_results,
)
from tensor_logic.system1_calibration import (
    apply_temperature_calibration,
    calibration_report,
    fit_temperature_calibration,
    temperature_scale_distribution,
)


def _question():
    return DecisionQuestion(
        question_id="q",
        primitive="choice",
        labels=("a", "b"),
    )


def _case(case_id: str, target: str):
    return DecisionCase(
        case_id=case_id,
        state={},
        questions=(_question(),),
        targets={"q": target},
    )


def _result(case_id: str, prediction: str, pa: float):
    probabilities = {"a": pa, "b": 1.0 - pa}
    return DecisionResult(
        case_id=case_id,
        question_id="q",
        primitive="choice",
        labels=("a", "b"),
        prediction=prediction,
        probabilities=probabilities,
        confidence=probabilities[prediction],
        latency_ms=1.0,
        backend="test",
        model="test",
        model_version="1",
    )


def test_temperature_one_is_identity():
    original = {"a": 0.8, "b": 0.2}
    scaled = temperature_scale_distribution(original, temperature=1.0)

    assert abs(scaled["a"] - 0.8) < 1e-12
    assert abs(scaled["b"] - 0.2) < 1e-12


def test_temperature_above_one_softens_distribution():
    original = {"a": 0.95, "b": 0.05}
    scaled = temperature_scale_distribution(original, temperature=2.0)

    assert 0.5 < scaled["a"] < 0.95
    assert math.isclose(sum(scaled.values()), 1.0)


def test_fit_softens_overconfident_imperfect_predictions():
    cases = (
        _case("c1", "a"),
        _case("c2", "a"),
        _case("c3", "b"),
        _case("c4", "b"),
    )
    # Two correct, two wrong, all extremely confident.
    results = (
        _result("c1", "a", 0.99),
        _result("c2", "a", 0.99),
        _result("c3", "a", 0.99),
        _result("c4", "a", 0.99),
    )

    calibration = fit_temperature_calibration(cases, results)
    row = calibration.by_question[0]

    assert row.temperature > 1.0
    assert row.dev_nll_after < row.dev_nll_before


def test_apply_calibration_preserves_schema_and_sets_version():
    cases = (
        _case("c1", "a"),
        _case("c2", "b"),
    )
    results = (
        _result("c1", "a", 0.9),
        _result("c2", "a", 0.9),
    )
    calibration = fit_temperature_calibration(cases, results)

    calibrated = apply_temperature_calibration(calibration, results)

    assert len(calibrated) == 2
    assert all(
        row.calibration_version == calibration.version
        for row in calibrated
    )
    assert all(
        abs(sum(row.probabilities.values()) - 1.0) < 1e-9
        for row in calibrated
    )


def test_calibration_report_uses_common_evaluator():
    cases = (
        _case("c1", "a"),
        _case("c2", "a"),
        _case("c3", "b"),
        _case("c4", "b"),
    )
    results = (
        _result("c1", "a", 0.99),
        _result("c2", "a", 0.99),
        _result("c3", "a", 0.99),
        _result("c4", "a", 0.99),
    )
    calibration = fit_temperature_calibration(cases, results)
    report = calibration_report(cases, results, calibration)

    before = evaluate_results(cases, results)
    assert report["before"]["nll"] == before.nll
    assert report["after"]["nll"] <= report["before"]["nll"]
    assert report["calibration_version"] == calibration.version


def test_invalid_distribution_fails_closed():
    try:
        temperature_scale_distribution(
            {"a": 0.8, "b": 0.8},
            temperature=1.0,
        )
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("invalid probability distribution should fail")
