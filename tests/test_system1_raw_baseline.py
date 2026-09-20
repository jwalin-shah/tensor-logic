from tensor_logic.system1_raw_baseline import (
    evaluate_raw_text_baseline,
    hashed_char_ngrams,
    train_raw_text_baseline,
)
from tensor_logic.system1_tasks import generate_pairs


def test_hashed_char_features_are_deterministic_and_normalized():
    left = hashed_char_ngrams("The same text 0.812345", dim=128)
    right = hashed_char_ngrams("The same text 0.812345", dim=128)

    assert left.shape == (128,)
    assert left.equal(right)
    assert abs(float(left.norm().item()) - 1.0) < 1e-6


def test_raw_text_baseline_training_reduces_shared_teacher_loss():
    pairs = generate_pairs(
        seed=47,
        iid_count=100,
        ood_per_family=6,
    )
    model, losses = train_raw_text_baseline(
        pairs,
        dim=192,
        epochs=60,
        learning_rate=0.08,
        seed=3,
    )

    assert losses[-1] < losses[0]

    result = evaluate_raw_text_baseline(
        model,
        pairs,
        split="test",
        dim=192,
    )

    assert result.cases > 0
    assert result.decisions == result.cases * 5
    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert 0.0 <= result.metrics.ece <= 1.0


def test_hashing_rejects_invalid_configuration():
    try:
        hashed_char_ngrams("x", dim=0)
    except ValueError as exc:
        assert "dim" in str(exc)
    else:
        raise AssertionError("zero dimensional hash space should fail")
