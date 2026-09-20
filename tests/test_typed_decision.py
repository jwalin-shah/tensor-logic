import torch

from tensor_logic.typed_decision import (
    TensorDecisionModel,
    binary_brier,
    expected_calibration_error,
    multiclass_brier,
)


def _model():
    model = TensorDecisionModel(
        3,
        choice_questions={
            "department": ("billing", "technical", "sales"),
            "route": ("fast", "deep"),
        },
        score_questions={
            "risk": ("low", "medium", "high"),
        },
        noul_questions=("escalate", "needs_human"),
    )
    with torch.no_grad():
        # department: state[0] -> technical, state[1] -> billing
        model.choice_weight.zero_()
        model.choice_bias.zero_()
        model.choice_weight[0, 0, 1] = 3.0  # billing
        model.choice_weight[0, 1, 0] = 3.0  # technical
        model.choice_weight[0, 2, 2] = 3.0  # sales

        # route: first option responds to feature0, second to feature2.
        model.choice_weight[1, 0, 0] = 2.0
        model.choice_weight[1, 1, 2] = 2.0

        # score risk: high on feature2, low on feature0.
        model.score_weight.zero_()
        model.score_bias.zero_()
        model.score_weight[0, 0, 0] = 3.0
        model.score_weight[0, 2, 2] = 3.0

        # noul questions.
        model.noul_weight.zero_()
        model.noul_bias.zero_()
        model.noul_weight[0, 2] = 4.0
        model.noul_weight[1, 1] = 4.0
    return model


def test_parallel_choice_score_noul_forward_shapes_and_probabilities():
    model = _model()
    state = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    out = model(state)

    assert out["choice_probabilities"].shape == (3, 2, 3)
    assert out["score_probabilities"].shape == (3, 1, 3)
    assert out["score_expected"].shape == (3, 1)
    assert out["noul_probability_yes"].shape == (3, 2)

    # Ragged choice bank: route has 2 options, padded option has probability 0.
    assert torch.allclose(
        out["choice_probabilities"][:, 1, 2],
        torch.zeros(3),
    )

    assert torch.allclose(
        out["choice_probabilities"][:, 0].sum(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )
    assert torch.allclose(
        out["choice_probabilities"][:, 1, :2].sum(dim=-1),
        torch.ones(3),
        atol=1e-6,
    )
    assert torch.allclose(
        out["score_probabilities"].sum(dim=-1),
        torch.ones(3, 1),
        atol=1e-6,
    )
    assert torch.all(
        (out["noul_probability_yes"] >= 0)
        & (out["noul_probability_yes"] <= 1)
    )


def test_decide_returns_only_typed_results_no_free_form_text():
    model = _model()
    result = model.decide(torch.tensor([[1.0, 0.0, 0.0]]))

    assert result.choices[0][0].question == "department"
    assert result.choices[0][0].option == "technical"
    assert set(result.choices[0][0].probabilities) == {
        "billing",
        "technical",
        "sales",
    }
    assert 0.0 <= result.choices[0][0].confidence <= 1.0

    assert result.scores[0][0].question == "risk"
    assert 0.0 <= result.scores[0][0].score <= 2.0
    assert set(result.scores[0][0].probabilities) == {
        "low",
        "medium",
        "high",
    }

    assert result.nouls[0][0].question == "escalate"
    assert 0.0 <= result.nouls[0][0].probability_yes <= 1.0
    assert 0.0 <= result.nouls[0][0].confidence <= 1.0


def test_multiple_questions_share_one_state_tensor_and_are_independent():
    model = _model()
    state = torch.tensor([[0.0, 0.0, 1.0]])

    result = model.decide(state)

    assert result.choices[0][0].option == "sales"
    assert result.choices[0][1].option == "deep"
    assert result.scores[0][0].score > 1.5
    assert result.nouls[0][0].probability_yes > 0.9
    assert result.nouls[0][1].probability_yes == 0.5


def test_calibration_metrics_have_expected_values():
    probs = torch.tensor(
        [
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )
    target = torch.tensor([0, 1])
    brier = multiclass_brier(probs, target)
    assert torch.allclose(brier, torch.tensor(0.05), atol=1e-6)

    binary = binary_brier(
        torch.tensor([0.8, 0.2]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.allclose(binary, torch.tensor(0.04), atol=1e-6)

    confidence = torch.tensor([0.8, 0.9, 0.6, 0.7])
    correct = torch.tensor([1.0, 1.0, 1.0, 0.0])
    ece = expected_calibration_error(
        confidence,
        correct,
        bins=2,
    )
    assert 0.0 <= float(ece.item()) <= 1.0


def test_invalid_question_shapes_fail_closed():
    try:
        TensorDecisionModel(
            4,
            choice_questions={"bad": ("only_one",)},
        )
    except ValueError as exc:
        assert ">=2" in str(exc)
    else:
        raise AssertionError("single-option choice should fail")

    try:
        TensorDecisionModel(
            4,
            score_questions={"bad": ("only_one",)},
        )
    except ValueError as exc:
        assert "2-10" in str(exc)
    else:
        raise AssertionError("single-level score should fail")
