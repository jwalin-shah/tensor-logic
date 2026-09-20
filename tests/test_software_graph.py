from tensor_logic.software_graph import SoftwareEvidenceGraph


def _graph():
    graph = SoftwareEvidenceGraph()
    graph.add_node("svc:api", "Service")
    graph.add_node("svc:db", "Service")
    graph.add_node("sym:handler", "Symbol")
    graph.add_node("sym:query", "Symbol")
    graph.add_node("trace:1", "Trace")
    graph.add_node("frame:root", "TraceFrame")
    graph.add_node("frame:db", "TraceFrame")
    graph.add_node("err:timeout", "Error")

    graph.add_edge(
        "svc:api",
        "svc:db",
        "service_depends_on",
        evidence_refs=("config:services",),
    )
    graph.add_edge(
        "sym:handler",
        "sym:query",
        "symbol_calls",
        evidence_refs=("ast:call:1",),
    )
    graph.add_edge(
        "trace:1",
        "frame:root",
        "trace_frame",
        evidence_refs=("trace:obs:1",),
    )
    graph.add_edge(
        "trace:1",
        "frame:db",
        "trace_frame",
        evidence_refs=("trace:obs:2",),
    )
    graph.add_edge(
        "frame:root",
        "frame:db",
        "frame_parent",
        evidence_refs=("trace:parent:1",),
    )
    graph.add_edge(
        "frame:db",
        "sym:query",
        "frame_symbol",
        evidence_refs=("trace:symbol:1",),
    )
    graph.add_edge(
        "err:timeout",
        "frame:db",
        "error_frame",
        evidence_refs=("error:obs:1",),
    )
    return graph


def test_graph_native_queries_preserve_structural_direction():
    graph = _graph()

    assert graph.ancestors(
        "frame:db",
        edge_kind="frame_parent",
    ) == frozenset({"frame:root"})
    assert len(graph.outgoing("trace:1", kind="trace_frame")) == 2


def test_tensor_projection_matches_graph_edges_and_provenance():
    graph = _graph()
    world = graph.to_tensor_world()

    depends = world.tensors["service_depends_on"]
    assert depends.get(("svc:api", "svc:db")) == 1.0
    assert depends.provenance(
        ("svc:api", "svc:db")
    ).evidence_refs == ("config:services",)

    calls = world.tensors["symbol_calls"]
    assert calls.get(("sym:handler", "sym:query")) == 1.0
    assert calls.provenance(
        ("sym:handler", "sym:query")
    ).metadata["graph_edge_kind"] == "symbol_calls"


def test_unknown_graph_edge_kind_fails_closed():
    graph = SoftwareEvidenceGraph()
    graph.add_node("a", "Service")
    graph.add_node("b", "Service")

    try:
        graph.add_edge("a", "b", "magic_relation")
    except ValueError as exc:
        assert "unknown edge kind" in str(exc)
    else:
        raise AssertionError("unknown graph relation should fail")


def test_conflicting_node_type_is_rejected():
    graph = SoftwareEvidenceGraph()
    graph.add_node("same", "Service")

    try:
        graph.add_node("same", "Symbol")
    except ValueError as exc:
        assert "conflicting node" in str(exc)
    else:
        raise AssertionError("conflicting node typing should fail")
