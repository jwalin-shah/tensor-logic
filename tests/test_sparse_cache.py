from tensor_logic.world_tensor import TensorWorld


def test_sparse_materialization_is_cached_until_delta():
    world = TensorWorld()
    world.add_axis("Person", "Person", ("p0", "p1"))
    world.add_axis("Event", "Event", ("e0", "e1"))
    tensor = world.add_tensor("attends", ("Person", "Event"))
    tensor.set(("p0", "e0"), 1.0)

    first = tensor.sparse()
    second = tensor.sparse()

    assert first is second
    assert first._nnz() == 1

    tensor.set(("p1", "e1"), 1.0)
    third = tensor.sparse()

    assert third is not first
    assert third._nnz() == 2

    tensor.remove(("p0", "e0"))
    fourth = tensor.sparse()

    assert fourth is not third
    assert fourth._nnz() == 1


def test_axis_growth_invalidates_sparse_shape_cache():
    world = TensorWorld()
    world.add_axis("Person", "Person", ("p0",))
    world.add_axis("Event", "Event", ("e0",))
    tensor = world.add_tensor("attends", ("Person", "Event"))
    tensor.set(("p0", "e0"), 1.0)

    before = tensor.sparse()
    assert tuple(before.shape) == (1, 1)

    world.extend_axis("Event", ("e1",))
    after = tensor.sparse()

    assert after is not before
    assert tuple(after.shape) == (1, 2)
    assert after._nnz() == 1
