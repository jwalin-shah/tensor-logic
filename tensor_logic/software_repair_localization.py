"""Proof-carrying software failure localization.

This is a deterministic candidate generator over SoftwareEvidenceGraph. It does
not edit code or propose patches. World evidence and ranking policy are kept
separate so policy changes cannot mutate the underlying software facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

from .software_graph import SoftwareEdge, SoftwareEvidenceGraph


@dataclass(frozen=True)
class RepairLocalizationPolicy:
    direct_failure: float = 1.0
    runtime_ancestor: float = 0.55
    static_caller: float = 0.35
    test_coverage: float = 0.10
    version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "direct_failure",
            "runtime_ancestor",
            "static_caller",
            "test_coverage",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} weight cannot be negative")

    @property
    def digest(self) -> str:
        raw = json.dumps(
            {
                "direct_failure": self.direct_failure,
                "runtime_ancestor": self.runtime_ancestor,
                "static_caller": self.static_caller,
                "test_coverage": self.test_coverage,
                "version": self.version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RepairContribution:
    kind: str
    value: float
    evidence_refs: tuple[str, ...]
    graph_edges: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class RepairCandidate:
    symbol: str
    score: float
    contributions: tuple[RepairContribution, ...]
    tests: tuple[str, ...]
    services: tuple[str, ...]
    policy_digest: str


def localize_error(
    graph: SoftwareEvidenceGraph,
    error_id: str,
    *,
    policy: RepairLocalizationPolicy | None = None,
    max_runtime_depth: int = 3,
    max_static_depth: int = 2,
) -> tuple[RepairCandidate, ...]:
    policy = policy or RepairLocalizationPolicy()
    if error_id not in graph.nodes:
        raise ValueError(f"unknown error node: {error_id}")
    if graph.nodes[error_id].kind != "Error":
        raise ValueError(f"{error_id!r} is not an Error node")
    if max_runtime_depth < 0 or max_static_depth < 0:
        raise ValueError("depth limits cannot be negative")

    contributions: dict[str, list[RepairContribution]] = {}

    error_edges = graph.outgoing(error_id, kind="error_frame")
    for error_edge in error_edges:
        frame_id = error_edge.target
        for frame_symbol_edge in graph.outgoing(
            frame_id,
            kind="frame_symbol",
        ):
            symbol = frame_symbol_edge.target
            _add(
                contributions,
                symbol,
                RepairContribution(
                    kind="direct_failure",
                    value=policy.direct_failure,
                    evidence_refs=_edge_evidence(
                        error_edge,
                        frame_symbol_edge,
                    ),
                    graph_edges=(
                        _edge_tuple(error_edge),
                        _edge_tuple(frame_symbol_edge),
                    ),
                ),
            )

            _add_runtime_ancestors(
                graph,
                frame_id,
                contributions,
                policy,
                max_depth=max_runtime_depth,
            )
            _add_static_callers(
                graph,
                symbol,
                contributions,
                policy,
                max_depth=max_static_depth,
            )

    candidates: list[RepairCandidate] = []
    for symbol, items in contributions.items():
        tests = tuple(
            sorted(
                edge.source
                for edge in graph.incoming(
                    symbol,
                    kind="test_covers_symbol",
                )
            )
        )
        services = tuple(
            sorted(
                edge.source
                for edge in graph.incoming(
                    symbol,
                    kind="service_symbol",
                )
            )
        )
        if tests:
            items.append(
                RepairContribution(
                    kind="test_coverage",
                    value=policy.test_coverage,
                    evidence_refs=tuple(
                        sorted(
                            {
                                ref
                                for edge in graph.incoming(
                                    symbol,
                                    kind="test_covers_symbol",
                                )
                                for ref in edge.evidence_refs
                            }
                        )
                    ),
                    graph_edges=tuple(
                        _edge_tuple(edge)
                        for edge in graph.incoming(
                            symbol,
                            kind="test_covers_symbol",
                        )
                    ),
                )
            )

        candidates.append(
            RepairCandidate(
                symbol=symbol,
                score=sum(item.value for item in items),
                contributions=tuple(items),
                tests=tests,
                services=services,
                policy_digest=policy.digest,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.symbol,
        )
    )
    return tuple(candidates)


def _add_runtime_ancestors(
    graph: SoftwareEvidenceGraph,
    frame_id: str,
    contributions: dict[str, list[RepairContribution]],
    policy: RepairLocalizationPolicy,
    *,
    max_depth: int,
) -> None:
    frontier = [(frame_id, 0)]
    visited = {frame_id}

    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for parent_edge in graph.incoming(
            current,
            kind="frame_parent",
        ):
            parent = parent_edge.source
            if parent in visited:
                continue
            visited.add(parent)
            parent_depth = depth + 1
            for symbol_edge in graph.outgoing(
                parent,
                kind="frame_symbol",
            ):
                value = policy.runtime_ancestor / parent_depth
                _add(
                    contributions,
                    symbol_edge.target,
                    RepairContribution(
                        kind=f"runtime_ancestor_depth_{parent_depth}",
                        value=value,
                        evidence_refs=_edge_evidence(
                            parent_edge,
                            symbol_edge,
                        ),
                        graph_edges=(
                            _edge_tuple(parent_edge),
                            _edge_tuple(symbol_edge),
                        ),
                    ),
                )
            frontier.append((parent, parent_depth))


def _add_static_callers(
    graph: SoftwareEvidenceGraph,
    symbol: str,
    contributions: dict[str, list[RepairContribution]],
    policy: RepairLocalizationPolicy,
    *,
    max_depth: int,
) -> None:
    frontier = [(symbol, 0)]
    visited = {symbol}

    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for call_edge in graph.incoming(
            current,
            kind="symbol_calls",
        ):
            caller = call_edge.source
            if caller in visited:
                continue
            visited.add(caller)
            caller_depth = depth + 1
            _add(
                contributions,
                caller,
                RepairContribution(
                    kind=f"static_caller_depth_{caller_depth}",
                    value=policy.static_caller / caller_depth,
                    evidence_refs=tuple(call_edge.evidence_refs),
                    graph_edges=(_edge_tuple(call_edge),),
                ),
            )
            frontier.append((caller, caller_depth))


def _add(
    contributions: dict[str, list[RepairContribution]],
    symbol: str,
    contribution: RepairContribution,
) -> None:
    contributions.setdefault(symbol, []).append(contribution)


def _edge_tuple(edge: SoftwareEdge) -> tuple[str, str, str]:
    return (edge.source, edge.kind, edge.target)


def _edge_evidence(*edges: SoftwareEdge) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                ref
                for edge in edges
                for ref in edge.evidence_refs
            }
        )
    )
