from tensor_logic.temporal_execution_graph import (
    TraceObservation,
    build_temporal_execution_graph,
)


def test_trace_observations_become_graph_and_tensor_views():
    graph = build_temporal_execution_graph(
        [
            TraceObservation(
                trace_id="trace:1",
                frame_id="frame:root",
                timestamp="t:1",
                symbol="sym:handler",
                service="svc:api",
                evidence_ref="obs:1",
            ),
            TraceObservation(
                trace_id="trace:1",
                frame_id="frame:db",
                parent_frame_id="frame:root",
                timestamp="t:2",
                symbol="sym:query",
                service="svc:db",
                error_id="err:timeout",
                incident_id="inc:1",
                evidence_ref="obs:2",
            ),
        ]
    )

    assert graph.ancestors(
        "frame:db",
        edge_kind="frame_parent",
    ) == frozenset({"frame:root"})

    world = graph.to_tensor_world()
    assert world.tensors["frame_parent"].get(
        ("frame:root", "frame:db")
    ) == 1.0
    assert world.tensors["frame_time"].get(
        ("frame:db", "t:2")
    ) == 1.0
    assert world.tensors["error_frame"].get(
        ("err:timeout", "frame:db")
    ) == 1.0
    assert world.tensors["incident_service"].get(
        ("inc:1", "svc:db")
    ) == 1.0


def test_trace_tensor_coordinate_keeps_observation_evidence():
    graph = build_temporal_execution_graph(
        [
            TraceObservation(
                trace_id="trace:1",
                frame_id="frame:1",
                timestamp="t:1",
                symbol="sym:x",
                service="svc:x",
                evidence_ref="raw-span:abc",
            )
        ]
    )

    world = graph.to_tensor_world()
    provenance = world.tensors["frame_symbol"].provenance(
        ("frame:1", "sym:x")
    )

    assert provenance.evidence_refs == ("raw-span:abc",)
    assert provenance.source_refs == ("runtime_trace",)
