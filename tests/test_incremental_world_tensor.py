from tensor_logic.world_tensor import TensorWorld


def test_axis_can_grow_without_losing_existing_coordinates():
    world = TensorWorld()
    world.add_axis("Person", "Person", ("p0",))
    world.add_axis("Event", "Event", ("e0",))
    attends = world.add_tensor("attends", ("Person", "Event"))
    attends.set(("p0", "e0"), 1.0)

    world.extend_axis("Person", ("p1",))
    world.extend_axis("Event", ("e1",))
    attends.set(("p1", "e1"), 1.0)

    assert attends.shape == (2, 2)
    assert attends.get(("p0", "e0")) == 1.0
    assert attends.get(("p1", "e1")) == 1.0
    assert attends.nnz == 2


def test_coordinate_retraction_is_incremental():
    world = TensorWorld()
    world.add_axis("Person", "Person", ("p0",))
    world.add_axis("Event", "Event", ("e0",))
    attends = world.add_tensor("attends", ("Person", "Event"))
    attends.set(("p0", "e0"), 1.0)

    assert attends.remove(("p0", "e0")) is True
    assert attends.get(("p0", "e0")) == 0.0
    assert attends.nnz == 0
    assert attends.remove(("p0", "e0")) is False
