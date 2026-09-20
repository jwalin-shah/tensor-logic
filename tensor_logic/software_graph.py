"""Explicit software graph + sparse tensor materialization.

The graph is the structural/temporal source view. Sparse tensors are derived
computational views. This implements the hybrid recommendation behind exp #93:
keep graph semantics for causality/debugging while exposing tensor relations for
fast composition and learned reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .software_schema import build_software_tensor_schema
from .world_tensor import CoordinateProvenance, TensorWorld


EDGE_TO_TENSOR = {
    "repo_commit": "repo_commit",
    "commit_file": "commit_file",
    "file_symbol": "file_symbol",
    "symbol_calls": "symbol_calls",
    "service_symbol": "service_symbol",
    "service_endpoint": "service_endpoint",
    "service_depends_on": "service_depends_on",
    "test_covers_symbol": "test_covers_symbol",
    "test_covers_endpoint": "test_covers_endpoint",
    "invariant_symbol": "invariant_symbol",
    "trace_incident": "trace_incident",
    "trace_frame": "trace_frame",
    "frame_symbol": "frame_symbol",
    "frame_parent": "frame_parent",
    "error_frame": "error_frame",
    "incident_error": "incident_error",
    "incident_service": "incident_service",
    "build_commit": "build_commit",
    "build_test": "build_test",
    "deployment_build": "deployment_build",
    "deployment_service": "deployment_service",
    "deployment_time": "deployment_time",
    "hypothesis_incident": "hypothesis_incident",
    "hypothesis_symbol": "hypothesis_symbol",
    "patch_hypothesis": "patch_hypothesis",
    "patch_commit": "patch_commit",
    "patch_test": "patch_test",
    "agent_patch": "agent_patch",
    "agent_toolcall": "agent_toolcall",
    "toolcall_receipt": "toolcall_receipt",
    "patch_receipt": "patch_receipt",
}


@dataclass(frozen=True)
class SoftwareNode:
    node_id: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftwareEdge:
    source: str
    target: str
    kind: str
    evidence_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class SoftwareEvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, SoftwareNode] = {}
        self.edges: list[SoftwareEdge] = []

    def add_node(
        self,
        node_id: str,
        kind: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not node_id or not kind:
            raise ValueError("node_id and kind are required")
        existing = self.nodes.get(node_id)
        node = SoftwareNode(
            node_id=node_id,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        if existing is not None and existing != node:
            raise ValueError(f"conflicting node definition: {node_id}")
        self.nodes[node_id] = node

    def add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        *,
        evidence_refs: tuple[str, ...] = (),
        source_refs: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("edge endpoints must exist")
        if kind not in EDGE_TO_TENSOR:
            raise ValueError(f"unknown edge kind: {kind}")
        self.edges.append(
            SoftwareEdge(
                source=source,
                target=target,
                kind=kind,
                evidence_refs=evidence_refs,
                source_refs=source_refs,
                metadata=dict(metadata or {}),
            )
        )

    def outgoing(
        self,
        node_id: str,
        *,
        kind: str | None = None,
    ) -> tuple[SoftwareEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.source == node_id
            and (kind is None or edge.kind == kind)
        )

    def incoming(
        self,
        node_id: str,
        *,
        kind: str | None = None,
    ) -> tuple[SoftwareEdge, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.target == node_id
            and (kind is None or edge.kind == kind)
        )

    def ancestors(
        self,
        node_id: str,
        *,
        edge_kind: str,
    ) -> frozenset[str]:
        """Graph-native transitive traversal for structural/temporal queries."""
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for edge in self.incoming(current, kind=edge_kind):
                if edge.source not in seen:
                    seen.add(edge.source)
                    frontier.append(edge.source)
        return frozenset(seen)

    def to_tensor_world(self) -> TensorWorld:
        symbols: dict[str, list[str]] = {}
        for node in self.nodes.values():
            symbols.setdefault(node.kind, []).append(node.node_id)

        world = build_software_tensor_schema(
            {
                kind: tuple(sorted(ids))
                for kind, ids in symbols.items()
            }
        )

        for edge in self.edges:
            tensor_name = EDGE_TO_TENSOR[edge.kind]
            tensor = world.tensors[tensor_name]
            tensor.set(
                (edge.source, edge.target),
                1.0,
                provenance=CoordinateProvenance(
                    evidence_refs=edge.evidence_refs,
                    source_refs=edge.source_refs,
                    metadata={
                        "graph_edge_kind": edge.kind,
                        **edge.metadata,
                    },
                ),
            )
        return world
