from tensor_logic.software_schema import (
    SOFTWARE_AXES,
    SOFTWARE_TENSORS,
    build_software_tensor_schema,
)


def test_software_schema_covers_static_dynamic_repair_and_receipt_state():
    world = build_software_tensor_schema(
        {
            "Repository": ("repo",),
            "Commit": ("c1", "c2"),
            "File": ("f.py",),
            "Symbol": ("foo", "bar"),
            "Service": ("api", "db"),
            "Endpoint": ("/x",),
            "Test": ("test_x",),
            "Incident": ("inc1",),
            "Error": ("err1",),
            "Trace": ("tr1",),
            "TraceFrame": ("fr1", "fr2"),
            "Invariant": ("inv1",),
            "Metric": ("latency",),
            "Patch": ("p1",),
            "RepairHypothesis": ("h1",),
            "AgentRun": ("a1",),
            "ToolCall": ("tc1",),
            "Receipt": ("r1",),
            "Build": ("b1",),
            "Deployment": ("d1",),
            "TimeBucket": ("t1",),
        }
    )

    assert set(world.axes) == set(SOFTWARE_AXES)
    assert set(world.tensors) == set(SOFTWARE_TENSORS)
    assert world.tensors["symbol_calls"].shape == (2, 2)
    assert world.tensors["service_depends_on"].shape == (2, 2)
    assert world.tensors["hypothesis_symbol"].shape == (1, 2)


def test_repair_state_can_be_expressed_without_promoting_hypothesis_to_fact():
    world = build_software_tensor_schema(
        {
            "Incident": ("inc1",),
            "Symbol": ("foo",),
            "RepairHypothesis": ("h1",),
            "Patch": ("p1",),
            "Test": ("test1",),
        }
    )

    world.tensors["hypothesis_incident"].set(("h1", "inc1"), 1.0)
    world.tensors["hypothesis_symbol"].set(("h1", "foo"), 1.0)
    world.tensors["hypothesis_confidence"].set(("h1",), 0.72)
    world.tensors["patch_hypothesis"].set(("p1", "h1"), 1.0)
    world.tensors["patch_test"].set(("p1", "test1"), 1.0)

    assert world.tensors["hypothesis_confidence"].get(("h1",)) == 0.72
    assert world.tensors["patch_validation_score"].get(("p1",)) == 0.0
