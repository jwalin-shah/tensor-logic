from tensor_logic.rule_hypothesis_eval import (
    RuleSplitResult,
    evaluate_rule_hypothesis,
)


def _row(split: str, accuracy: float, counterexamples: int = 0):
    return RuleSplitResult(
        split_id=split,
        accuracy=accuracy,
        precision=accuracy,
        recall=accuracy,
        counterexamples=counterexamples,
        examples=100,
    )


def test_stable_rule_can_become_admission_candidate():
    result = evaluate_rule_hypothesis(
        [_row("s1", 0.95), _row("s2", 0.94), _row("s3", 0.96)]
    )

    assert result.admission_candidate is True
    assert result.stable is True
    assert result.reasons == ()


def test_split_sensitive_rule_is_rejected_even_with_good_mean():
    result = evaluate_rule_hypothesis(
        [_row("s1", 0.99), _row("s2", 0.82), _row("s3", 0.99)]
    )

    assert result.admission_candidate is False
    assert "heldout_accuracy_below_threshold" in result.reasons
    assert "split_instability" in result.reasons


def test_counterexample_blocks_promotion_by_default():
    result = evaluate_rule_hypothesis(
        [_row("s1", 0.98), _row("s2", 0.98, counterexamples=1)]
    )

    assert result.admission_candidate is False
    assert "counterexamples_exceed_threshold" in result.reasons


def test_one_split_is_not_enough_to_claim_stability():
    try:
        evaluate_rule_hypothesis([_row("only", 1.0)])
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("single-split rule should not be evaluable")
