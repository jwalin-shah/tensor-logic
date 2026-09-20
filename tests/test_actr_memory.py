import math

import torch

from tensor_logic.actr_memory import (
    base_level_activation,
    compute_activation,
    rank_memories,
    retrieval_latency,
    retrieval_probability,
    tensor_base_level_activation,
)


def test_base_level_rewards_frequency_and_recency():
    recent_once = base_level_activation([90.0], now=100.0)
    old_once = base_level_activation([10.0], now=100.0)
    repeated = base_level_activation([10.0, 40.0, 90.0], now=100.0)

    assert recent_once > old_once
    assert repeated > recent_once


def test_association_can_reorder_memory_without_rewriting_history():
    histories = {
        "old_goal_relevant": [10.0],
        "recent_irrelevant": [95.0],
    }

    baseline = rank_memories(histories, now=100.0)
    goal_context = rank_memories(
        histories,
        now=100.0,
        associative={"old_goal_relevant": 3.0},
    )

    assert baseline[0].item_id == "recent_irrelevant"
    assert goal_context[0].item_id == "old_goal_relevant"


def test_retrieval_probability_and_latency_move_in_opposite_directions():
    low = compute_activation("x", use_times=[1.0], now=100.0)
    high = compute_activation(
        "x",
        use_times=[90.0, 95.0],
        now=100.0,
        associative=1.0,
    )

    assert high.retrieval_probability > low.retrieval_probability
    assert high.latency < low.latency
    assert 0.0 <= low.retrieval_probability <= 1.0
    assert high.latency > 0.0


def test_vectorized_base_level_matches_scalar_equation():
    ages = torch.tensor(
        [
            [10.0, 90.0, 1.0],
            [5.0, 1.0, 1.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    actual = tensor_base_level_activation(ages, mask, decay=0.5)
    expected = torch.tensor(
        [
            math.log(10.0 ** -0.5 + 90.0 ** -0.5),
            math.log(5.0 ** -0.5),
        ]
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_empty_history_is_not_retrievable():
    activation = base_level_activation([], now=100.0)
    assert activation == float("-inf")
    assert retrieval_probability(activation) == 0.0
    assert math.isinf(retrieval_latency(activation))
