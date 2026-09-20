from tensor_logic.software_graph import SoftwareEvidenceGraph
from tensor_logic.software_repair_localization import (
    RepairLocalizationPolicy,
    localize_error,
)


def _graph():
    g = SoftwareEvidenceGraph()

    for node_id, kind in (
        ("err:timeout", "Error"),
        ("frame:db", "TraceFrame"),
        ("frame:handler", "TraceFrame"),
        ("sym:query", "Symbol"),
        ("sym:handler", "Symbol"),
        ("sym:controller", "Symbol"),
        ("sym:unrelated", "Symbol"),
        ("test:query", "Test"),
        ("svc:db", "Service"),
    ):
        g.add_node(node_id, kind)

    g.add_edge(
        "err:timeout",
        "frame:db",
        "error_frame",
        evidence_refs=("runtime:error",),
    )
    g.add_edge(
        "frame:db",
        "sym:query",
        "frame_symbol",
        evidence_refs=("runtime:query-frame",),
    )
    g.add_edge(
        "frame:handler",
        "frame:db",
        "frame_parent",
        evidence_refs=("runtime:parent",),
    )
    g.add_edge(
        "frame:handler",
        "sym:handler",
        "frame_symbol",
        evidence_refs=("runtime:handler-frame",),
    )
    g.add_edge(
        "sym:handler",
        "sym:query",
        "symbol_calls",
        evidence_refs=("static:handler-query",),
    )
    g.add_edge(
        "sym:controller",
        "sym:handler",
        "symbol_calls",
        evidence_refs=("static:controller-handler",),
    )
    g.add_edge(
        "test:query",
        "sym:query",
        "test_covers_symbol",
        evidence_refs=("coverage:test-query",),
    )
    g.add_edge(
        "svc:db",
        "sym:query",
        "service_symbol",
        evidence_refs=("service:db-query",),
    )
    return g


def test_direct_failing_symbol_ranks_first_and_keeps_evidence():
    candidates = localize_error(_graph(), "err:timeout")

    assert candidates[0].symbol == "sym:query"
    assert candidates[0].score > candidates[1].score
    assert candidates[0].tests == ("test:query",)
    assert candidates[0].services == ("svc:db",)

    refs = {
        ref
        for contribution in candidates[0].contributions
        for ref in contribution.evidence_refs
    }
    assert "runtime:error" in refs
    assert "runtime:query-frame" in refs
    assert "coverage:test-query" in refs


def test_runtime_parent_and_static_callers_become_candidates():
    candidates = localize_error(_graph(), "err:timeout")
    by_symbol = {item.symbol: item for item in candidates}

    assert "sym:handler" in by_symbol
    assert "sym:controller" in by_symbol
    assert "sym:unrelated" not in by_symbol

    kinds = {
        item.kind
        for item in by_symbol["sym:handler"].contributions
    }
    assert "runtime_ancestor_depth_1" in kinds
    assert "static_caller_depth_1" in kinds


def test_policy_changes_ranking_weights_without_mutating_graph():
    graph = _graph()
    default = localize_error(graph, "err:timeout")

    caller_first_policy = RepairLocalizationPolicy(
        direct_failure=0.1,
        runtime_ancestor=0.1,
        static_caller=2.0,
        test_coverage=0.0,
        version="caller-heavy",
    )
    caller_heavy = localize_error(
        graph,
        "err:timeout",
        policy=caller_first_policy,
    )

    assert default[0].policy_digest != caller_heavy[0].policy_digest
    assert graph.outgoing("err:timeout", kind="error_frame")
    assert any(
        item.symbol == "sym:handler"
        for item in caller_heavy
    )


def test_retracting_static_call_edge_removes_static_contribution():
    graph = _graph()
    before = {
        item.symbol: item
        for item in localize_error(graph, "err:timeout")
    }
    assert "static_caller_depth_1" in {
        item.kind
        for item in before["sym:handler"].contributions
    }

    graph.edges = [
        edge
        for edge in graph.edges
        if not (
            edge.kind == "symbol_calls"
            and edge.source == "sym:handler"
            and edge.target == "sym:query"
        )
    ]
    after = {
        item.symbol: item
        for item in localize_error(graph, "err:timeout")
    }

    assert "sym:handler" in after  # still supported by runtime ancestry
    assert "static_caller_depth_1" not in {
        item.kind
        for item in after["sym:handler"].contributions
    }


def test_removing_runtime_error_edge_removes_all_candidates():
    graph = _graph()
    graph.edges = [
        edge
        for edge in graph.edges
        if not (
            edge.kind == "error_frame"
            and edge.source == "err:timeout"
        )
    ]

    assert localize_error(graph, "err:timeout") == ()


def test_unknown_error_fails_closed():
    try:
        localize_error(_graph(), "err:missing")
    except ValueError as exc:
        assert "unknown error" in str(exc)
    else:
        raise AssertionError("unknown error should fail")
