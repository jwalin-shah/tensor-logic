from tensor_logic.system1_tasks import generate_pairs
from tensor_logic.system1_training import (
    evaluate_structured_baseline,
    train_structured_baseline,
)


def test_structured_baseline_training_reduces_soft_teacher_loss():
    pairs = generate_pairs(
        seed=31,
        iid_count=100,
        ood_per_family=8,
    )
    model, history = train_structured_baseline(
        pairs,
        epochs=80,
        learning_rate=0.08,
        seed=1,
    )

    assert len(history.losses) == 80
    assert history.losses[-1] < history.losses[0]

    evaluation = evaluate_structured_baseline(
        model,
        pairs,
        split="test",
    )

    assert evaluation.cases > 0
    assert evaluation.decisions == evaluation.cases * 5
    assert 0.0 <= evaluation.metrics.accuracy <= 1.0
    assert evaluation.metrics.brier >= 0.0
    assert evaluation.metrics.nll >= 0.0
    assert 0.0 <= evaluation.metrics.ece <= 1.0
    assert evaluation.mean_soft_accuracy is not None
    assert 0.0 <= evaluation.mean_soft_accuracy <= 1.0


def test_structured_baseline_can_be_evaluated_on_each_ood_family():
    pairs = generate_pairs(
        seed=37,
        iid_count=100,
        ood_per_family=6,
    )
    model, _ = train_structured_baseline(
        pairs,
        epochs=60,
        learning_rate=0.08,
        seed=2,
    )

    for split in (
        "ood_compositional",
        "ood_source_failure",
        "ood_confidence_shift",
    ):
        result = evaluate_structured_baseline(
            model,
            pairs,
            split=split,
        )
        assert result.cases == 6
        assert result.decisions == 30
        assert 0.0 <= result.metrics.accuracy <= 1.0


def test_training_rejects_missing_split():
    pairs = generate_pairs(
        seed=41,
        iid_count=10,
        ood_per_family=0,
    )

    try:
        train_structured_baseline(
            pairs,
            split="does_not_exist",
            epochs=1,
        )
    except ValueError as exc:
        assert "no cases" in str(exc)
    else:
        raise AssertionError("missing training split should fail")
