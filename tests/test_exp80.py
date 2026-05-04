from experiments.exp80_fafsa_kb import (
    DependentFamily,
    FAMILIES,
    CitedValue,
    prove_sai,
    prove_sai_counterfactual,
)
from experiments.exp80_validate_synthetic import NAMED_CASES, make_family


def _step_map(trace):
    return {step.label: step.value for step in trace.steps}


def test_median_family_sai_regression():
    trace = prove_sai(FAMILIES["median_w2_two_parent"])
    assert trace.sai == 2286
    assert not trace.auto_neg1500


def test_max_pell_short_circuit():
    trace = prove_sai(FAMILIES["low_income_max_pell"])
    assert trace.sai == -1500
    assert trace.auto_neg1500
    assert trace.steps == []


def test_trace_steps_have_citations_and_formulas():
    trace = prove_sai(FAMILIES["median_w2_two_parent"])
    assert trace.steps
    for step in trace.steps:
        assert isinstance(step, CitedValue)
        assert step.citation
        assert step.formula


def test_sai_equals_component_sum_with_floor():
    trace = prove_sai(FAMILIES["median_w2_two_parent"])
    steps = _step_map(trace)
    expected = max(
        -1500,
        steps["parent_contribution"]
        + steps["student_contribution_from_income"]
        + steps["student_contribution_from_assets"],
    )
    assert trace.sai == expected


def test_counterfactual_lower_parent_income_lowers_sai():
    base = FAMILIES["median_w2_two_parent"]
    base_trace = prove_sai(base)
    cf_trace = prove_sai_counterfactual(base, {
        "parent_agi": 60_000,
        "parent_earned_income_p1": 36_000,
        "parent_earned_income_p2": 24_000,
        "parent_income_tax_paid": 6_000,
    })
    assert cf_trace.sai < base_trace.sai


def test_synthetic_named_cases_compute_without_errors():
    for _, family in NAMED_CASES:
        trace = prove_sai(family)
        assert trace.sai >= -1500


def test_seeded_synthetic_families_preserve_sai_invariant():
    for seed in range(25):
        trace = prove_sai(make_family(seed))
        steps = _step_map(trace)
        expected = max(
            -1500,
            steps.get("parent_contribution", 0)
            + steps.get("student_contribution_from_income", 0)
            + steps.get("student_contribution_from_assets", 0),
        )
        assert abs(trace.sai - expected) <= 1
