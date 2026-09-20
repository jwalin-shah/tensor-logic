from tensor_logic.metacognition import (
    ComputationOption,
    MetaState,
    choose_meta_action,
    fixed_confidence_policy,
    value_of_computation,
)


def _options():
    return (
        ComputationOption(
            action="ACT",
            expected_accuracy=0.72,
            compute_cost=0.00,
        ),
        ComputationOption(
            action="THINK",
            expected_accuracy=0.88,
            compute_cost=0.06,
            latency_cost=0.02,
        ),
        ComputationOption(
            action="ESCALATE",
            expected_accuracy=0.97,
            compute_cost=0.05,
            latency_cost=0.04,
            escalation_cost=0.08,
        ),
    )


def test_high_stakes_contradictory_case_can_justify_escalation():
    state = MetaState(
        confidence=0.45,
        contradiction=1.0,
        source_staleness=0.8,
        error_cost=5.0,
        compute_budget=0.8,
    )
    decision = choose_meta_action(state, _options())
    assert decision.action == "ESCALATE"
    assert decision.utilities["ESCALATE"] > decision.utilities["ACT"]


def test_low_stakes_case_can_rationally_act_even_below_fixed_threshold():
    state = MetaState(
        confidence=0.70,
        contradiction=0.0,
        source_staleness=0.0,
        error_cost=0.1,
        compute_budget=0.2,
    )
    decision = choose_meta_action(state, _options())

    assert fixed_confidence_policy(0.70) == "THINK"
    assert decision.action == "ACT"


def test_value_of_computation_is_positive_only_when_improvement_is_worth_cost():
    assert value_of_computation(
        act_accuracy=0.6,
        improved_accuracy=0.9,
        error_cost=3.0,
        computation_cost=0.2,
    ) > 0.0

    assert value_of_computation(
        act_accuracy=0.9,
        improved_accuracy=0.92,
        error_cost=0.5,
        computation_cost=0.2,
    ) < 0.0


def test_low_compute_budget_penalizes_thinking_more():
    roomy = MetaState(
        confidence=0.6,
        contradiction=0.1,
        source_staleness=0.1,
        error_cost=1.0,
        compute_budget=1.0,
    )
    scarce = MetaState(
        confidence=0.6,
        contradiction=0.1,
        source_staleness=0.1,
        error_cost=1.0,
        compute_budget=0.0,
    )

    roomy_decision = choose_meta_action(roomy, _options())
    scarce_decision = choose_meta_action(scarce, _options())

    assert (
        scarce_decision.utilities["THINK"]
        < roomy_decision.utilities["THINK"]
    )


def test_duplicate_action_options_fail_closed():
    try:
        choose_meta_action(
            MetaState(
                confidence=0.5,
                contradiction=0.0,
                source_staleness=0.0,
                error_cost=1.0,
                compute_budget=1.0,
            ),
            (
                ComputationOption("ACT", 0.5, 0.0),
                ComputationOption("ACT", 0.6, 0.0),
            ),
        )
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate meta-actions should fail")
