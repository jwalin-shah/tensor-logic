import torch

from tensor_logic.tensor_ops import (
    booleanize_sparse,
    sparse_binary_compose,
    topk_indices,
    vectorized_weighted_score,
)
from tensor_logic.world_tensor import TensorWorld


def test_sparse_binary_compose_finds_two_hop_paths():
    world = TensorWorld()
    world.add_axis("Person", "Person", ("p0", "p1"))
    world.add_axis("Event", "Event", ("e0", "e1", "e2"))
    world.add_axis("Topic", "Topic", ("t0", "t1"))
    attends = world.add_tensor(
        "attends",
        ("Person", "Event"),
    )
    event_topic = world.add_tensor(
        "event_topic",
        ("Event", "Topic"),
    )

    attends.set(("p0", "e0"), 1)
    attends.set(("p0", "e1"), 1)
    attends.set(("p1", "e2"), 1)

    event_topic.set(("e0", "t0"), 1)
    event_topic.set(("e1", "t0"), 1)
    event_topic.set(("e2", "t1"), 1)

    product = sparse_binary_compose(attends, event_topic).to_dense()

    assert product.shape == (2, 2)
    assert product[0, 0].item() == 2.0
    assert product[1, 1].item() == 1.0
    assert booleanize_sparse(
        sparse_binary_compose(attends, event_topic)
    ).to_dense().tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_vectorized_scoring_and_topk():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.4, 0.4, 0.4],
        ]
    )
    weights = torch.tensor([0.5, 0.3, 0.2])

    scores = vectorized_weighted_score(features, weights)
    values, indices = topk_indices(scores, 2)

    assert scores.tolist() == [
        0.6000000238418579,
        0.30000001192092896,
        0.4000000059604645,
    ]
    assert indices.tolist() == [0, 2]
    assert values[0] > values[1]


def test_sparse_compose_rejects_axis_mismatch():
    left_world = TensorWorld()
    left_world.add_axis("A", "A", ("a",))
    left_world.add_axis("B", "B", ("b",))
    left = left_world.add_tensor("left", ("A", "B"))

    right_world = TensorWorld()
    right_world.add_axis("B", "B", ("other",))
    right_world.add_axis("C", "C", ("c",))
    right = right_world.add_tensor("right", ("B", "C"))

    try:
        sparse_binary_compose(left, right)
    except ValueError as exc:
        assert "inner axes" in str(exc)
    else:
        raise AssertionError("expected axis mismatch")
