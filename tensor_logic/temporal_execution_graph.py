"""Build a compact Temporal Execution Graph from normalized trace observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .software_graph import SoftwareEvidenceGraph


@dataclass(frozen=True)
class TraceObservation:
    trace_id: str
    frame_id: str
    timestamp: str
    symbol: str
    service: str
    parent_frame_id: str | None = None
    error_id: str | None = None
    incident_id: str | None = None
    evidence_ref: str | None = None


def build_temporal_execution_graph(
    observations: Iterable[TraceObservation],
) -> SoftwareEvidenceGraph:
    rows = tuple(observations)
    graph = SoftwareEvidenceGraph()

    # Declare nodes first so edges can fail closed if normalization is broken.
    for row in rows:
        graph.add_node(row.trace_id, "Trace")
        graph.add_node(row.frame_id, "TraceFrame")
        graph.add_node(row.symbol, "Symbol")
        graph.add_node(row.service, "Service")
        graph.add_node(row.timestamp, "TimeBucket")
        if row.parent_frame_id is not None:
            graph.add_node(row.parent_frame_id, "TraceFrame")
        if row.error_id is not None:
            graph.add_node(row.error_id, "Error")
        if row.incident_id is not None:
            graph.add_node(row.incident_id, "Incident")

    for row in rows:
        evidence = (
            (row.evidence_ref,)
            if row.evidence_ref is not None
            else ()
        )
        graph.add_edge(
            row.trace_id,
            row.frame_id,
            "trace_frame",
            evidence_refs=evidence,
            source_refs=("runtime_trace",),
        )
        graph.add_edge(
            row.frame_id,
            row.symbol,
            "frame_symbol",
            evidence_refs=evidence,
            source_refs=("runtime_trace",),
        )
        graph.add_edge(
            row.frame_id,
            row.timestamp,
            "frame_time",
            evidence_refs=evidence,
            source_refs=("runtime_trace",),
        )

        if row.parent_frame_id is not None:
            graph.add_edge(
                row.parent_frame_id,
                row.frame_id,
                "frame_parent",
                evidence_refs=evidence,
                source_refs=("runtime_trace",),
            )

        if row.error_id is not None:
            graph.add_edge(
                row.error_id,
                row.frame_id,
                "error_frame",
                evidence_refs=evidence,
                source_refs=("runtime_trace",),
            )

        if row.incident_id is not None:
            graph.add_edge(
                row.trace_id,
                row.incident_id,
                "trace_incident",
                evidence_refs=evidence,
                source_refs=("runtime_trace",),
            )
            graph.add_edge(
                row.incident_id,
                row.service,
                "incident_service",
                evidence_refs=evidence,
                source_refs=("runtime_trace",),
            )
            if row.error_id is not None:
                graph.add_edge(
                    row.incident_id,
                    row.error_id,
                    "incident_error",
                    evidence_refs=evidence,
                    source_refs=("runtime_trace",),
                )

    return graph
