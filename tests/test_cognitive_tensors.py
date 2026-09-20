import torch

from tensor_logic.cognitive_ops import (
    evaluate_plan_steps,
    metacognitive_escalation,
    prediction_error,
    score_operators,
    select_working_set,
)
from tensor_logic.cognitive_schema import (
    COGNITIVE_AXES,
    COGNITIVE_TENSORS,
    build_cognitive_tensor_schema,
)


def test_cognitive_schema_covers_core_memory_and_control_stores():
    world = build_cognitive_tensor_schema(
        {
            "Agent": ("me",),
            "TaskContext": ("task:1",),
            "MemoryItem": ("m0", "m1", "m2"),
            "Episode": ("ep0",),
            "CognitiveState": ("s0", "s1"),
            "Operator": ("op0", "op1"),
            "Skill": ("skill0",),
            "Goal": ("g0",),
            "Subgoal": ("sg0",),
            "Prediction": ("pred0",),
            "Outcome": ("out0",),
            "Hypothesis": ("h0",),
            "Claim": ("c0",),
            "Evidence": ("e0",),
            "Resource": ("time", "tokens"),
            "ErrorSignal": ("err0",),
            "Plan": ("plan0",),
            "PlanStep": ("ps0", "ps1"),
            "Policy": ("policy0",),
            "ModelVersion": ("model0",),
            "Utterance": ("u0",),
            "GenerationStep": ("gen0",),
            "Token": ("yes", "no"),
            "TimeBucket": ("t0",),
        }
    )

    assert set(world.axes) == set(COGNITIVE_AXES)
    assert set(world.tensors) == set(COGNITIVE_TENSORS)
    assert world.tensors["working_memory"].shape == (1, 3)
    assert world.tensors["episode_operator"].shape == (1, 2)
    assert world.tensors["operator_resource_cost"].shape == (2, 2)
    assert world.tensors["generation_token_probability"].shape == (1, 2)


def test_working_set_is_bounded_and_goal_sensitive():
    result = select_working_set(
        relevance=torch.tensor([0.9, 0.2, 0.8, 0.4]),
        activation=torch.tensor([0.3, 1.0, 0.2, 0.7]),
        goal_alignment=torch.tensor([0.8, 0.1, 0.9, 0.3]),
        contradiction_bonus=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        capacity=2,
    )

    assert result.indices.tolist() == [2, 0]
    assert len(result.indices) == 2
    assert result.scores[0] > result.scores[1]


def test_operator_scoring_balances_goal_success_information_and_cost():
    scores = score_operators(
        goal_value=torch.tensor([1.0, 0.8, 0.5]),
        success_probability=torch.tensor([0.9, 0.8, 0.7]),
        expected_information_gain=torch.tensor([0.1, 0.8, 1.0]),
        resource_cost=torch.tensor([0.2, 0.3, 0.7]),
        uncertainty=torch.tensor([0.1, 0.4, 0.6]),
    )

    assert int(torch.argmax(scores).item()) == 0
    assert scores[0] > scores[1] > scores[2]


def test_prediction_error_preserves_direction():
    error = prediction_error(
        torch.tensor([0.8, 0.2, 0.5]),
        torch.tensor([1.0, 0.0, 0.5]),
    )
    assert torch.allclose(
        error,
        torch.tensor([0.2, -0.2, 0.0]),
        atol=1e-6,
    )


def test_metacognition_escalates_uncertain_contradictory_high_regret_case():
    score, escalate = metacognitive_escalation(
        confidence=torch.tensor([0.95, 0.35]),
        contradiction=torch.tensor([0.0, 0.9]),
        source_staleness=torch.tensor([0.0, 0.8]),
        expected_regret=torch.tensor([0.1, 0.9]),
        resource_pressure=torch.tensor([0.2, 0.2]),
        threshold=0.5,
    )

    assert bool(escalate[0].item()) is False
    assert bool(escalate[1].item()) is True
    assert score[1] > score[0]


def test_plan_evaluation_rejects_high_raw_reward_when_probability_and_risk_are_bad():
    value = evaluate_plan_steps(
        reward=torch.tensor([1.0, 0.9, 1.4]),
        probability=torch.tensor([0.9, 0.95, 0.4]),
        time_cost=torch.tensor([0.2, 0.1, 0.5]),
        resource_cost=torch.tensor([0.2, 0.1, 0.6]),
        risk=torch.tensor([0.1, 0.05, 0.8]),
    )

    assert int(torch.argmax(value).item()) == 0
    assert value[0] > value[2]
