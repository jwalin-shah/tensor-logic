"""exp110: proof-carrying software repair localization smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensor_logic.software_graph import SoftwareEvidenceGraph
from tensor_logic.software_repair_localization import localize_error


def build_graph() -> SoftwareEvidenceGraph:
    graph = SoftwareEvidenceGraph()
    for node_id, kind in (
        ("err:timeout", "Error"),
        ("frame:query", "TraceFrame"),
        ("frame:handler", "TraceFrame"),
        ("sym:query", "Symbol"),
        ("sym:handler", "Symbol"),
        ("sym:api", "Symbol"),
        ("test:query", "Test"),
        ("svc:db", "Service"),
    ):
        graph.add_node(node_id, kind)

    graph.add_edge(
        "err:timeout",
        "frame:query",
        "error_frame",
        evidence_refs=("trace:error:1",),
    )
    graph.add_edge(
        "frame:query",
        "sym:query",
        "frame_symbol",
        evidence_refs=("trace:frame:query",),
    )
    graph.add_edge(
        "frame:handler",
        "frame:query",
        "frame_parent",
        evidence_refs=("trace:parent:1",),
    )
    graph.add_edge(
        "frame:handler",
        "sym:handler",
        "frame_symbol",
        evidence_refs=("trace:frame:handler",),
    )
    graph.add_edge(
        "sym:handler",
        "sym:query",
        "symbol_calls",
        evidence_refs=("ast:handler-query",),
    )
    graph.add_edge(
        "sym:api",
        "sym:handler",
        "symbol_calls",
        evidence_refs=("ast:api-handler",),
    )
    graph.add_edge(
        "test:query",
        "sym:query",
        "test_covers_symbol",
        evidence_refs=("coverage:query",),
    )
    graph.add_edge(
        "svc:db",
        "sym:query",
        "service_symbol",
        evidence_refs=("ownership:db",),
    )
    return graph


def run() -> dict:
    graph = build_graph()
    candidates = localize_error(graph, "err:timeout")

    return {
        "experiment": "exp110_software_repair_localization",
        "error": "err:timeout",
        "candidate_count": len(candidates),
        "candidates": [
            {
                "symbol": candidate.symbol,
                "score": candidate.score,
                "tests": list(candidate.tests),
                "services": list(candidate.services),
                "policy_digest": candidate.policy_digest,
                "contributions": [
                    {
                        "kind": contribution.kind,
                        "value": contribution.value,
                        "evidence_refs": list(contribution.evidence_refs),
                        "graph_edges": [
                            list(edge)
                            for edge in contribution.graph_edges
                        ],
                    }
                    for contribution in candidate.contributions
                ],
            }
            for candidate in candidates
        ],
        "top_candidate": candidates[0].symbol if candidates else None,
        "claim_boundary": (
            "Synthetic localization proof only. No patch is generated or "
            "executed, and ranking quality is not established on real bugs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp110.json")
    args = parser.parse_args()

    result = run()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
