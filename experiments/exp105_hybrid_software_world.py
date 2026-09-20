"""exp105: end-to-end hybrid world smoke proof.

Evidence log -> Temporal Execution Graph -> sparse tensor materialization ->
tensor composition for an error-to-symbol derived relation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensor_logic.temporal_execution_graph import (
    TraceObservation,
    build_temporal_execution_graph,
)
from tensor_logic.tensor_ops import sparse_binary_compose
from tensor_logic.world_event_log import (
    AppendOnlyWorldLog,
    WorldEvent,
)


def run() -> dict:
    log = AppendOnlyWorldLog()
    log.extend(
        (
            WorldEvent(
                event_id="obs:root",
                event_type="trace_frame",
                occurred_at="2026-09-20T16:00:00Z",
                entity_refs=("trace:1", "frame:root", "sym:handler"),
                source_refs=("runtime_trace",),
                evidence_refs=("span:root",),
                payload={
                    "trace_id": "trace:1",
                    "frame_id": "frame:root",
                    "timestamp": "t:1",
                    "symbol": "sym:handler",
                    "service": "svc:api",
                },
            ),
            WorldEvent(
                event_id="obs:db",
                event_type="trace_frame",
                occurred_at="2026-09-20T16:00:01Z",
                entity_refs=(
                    "trace:1",
                    "frame:db",
                    "sym:query",
                    "err:timeout",
                ),
                source_refs=("runtime_trace",),
                evidence_refs=("span:db",),
                payload={
                    "trace_id": "trace:1",
                    "frame_id": "frame:db",
                    "parent_frame_id": "frame:root",
                    "timestamp": "t:2",
                    "symbol": "sym:query",
                    "service": "svc:db",
                    "error_id": "err:timeout",
                    "incident_id": "inc:1",
                },
            ),
        )
    )

    observations = tuple(
        TraceObservation(
            trace_id=item.event.payload["trace_id"],
            frame_id=item.event.payload["frame_id"],
            parent_frame_id=item.event.payload.get("parent_frame_id"),
            timestamp=item.event.payload["timestamp"],
            symbol=item.event.payload["symbol"],
            service=item.event.payload["service"],
            error_id=item.event.payload.get("error_id"),
            incident_id=item.event.payload.get("incident_id"),
            evidence_ref=(
                item.event.evidence_refs[0]
                if item.event.evidence_refs
                else None
            ),
        )
        for item in log.events
    )

    graph = build_temporal_execution_graph(observations)
    world = graph.to_tensor_world()

    error_symbol = sparse_binary_compose(
        world.tensors["error_frame"],
        world.tensors["frame_symbol"],
    ).to_dense()

    error_axis = world.axes["Error"]
    symbol_axis = world.axes["Symbol"]
    derived = []
    for error in error_axis.symbols:
        for symbol in symbol_axis.symbols:
            if (
                error_symbol[
                    error_axis.position(error),
                    symbol_axis.position(symbol),
                ].item()
                > 0
            ):
                derived.append((error, symbol))

    return {
        "experiment": "exp105_hybrid_software_world",
        "event_count": log.sequence,
        "event_log_digest": log.digest,
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "tensor_count": len(world.tensors),
        "tensor_coordinates": sum(
            tensor.nnz
            for tensor in world.tensors.values()
        ),
        "derived_error_symbol": derived,
        "source_evidence": list(
            world.tensors["error_frame"].provenance(
                ("err:timeout", "frame:db")
            ).evidence_refs
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp105.json")
    args = parser.parse_args()
    result = run()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
