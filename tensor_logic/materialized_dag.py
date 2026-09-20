"""Digest-keyed materialization cache for arbitrary compiled query DAGs.

Unlike materialized_view.py, which is optimized for proof-carrying binary tensor
composition, this module is operator-agnostic and supports view-of-view reuse.

Cache identity is:
    view definition digest + exact input artifact digests

Therefore a primitive delta only invalidates downstream views whose dependency
digests actually change; unrelated/intermediate-clean views remain reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

import torch

from .query_execution import QueryExecutionAssessment
from .query_runtime import CompiledQuery, QueryRegistry
from .selective_executor import QueryOperatorRegistry


@dataclass(frozen=True)
class VersionedArtifact:
    name: str
    value: Any
    digest: str


@dataclass(frozen=True)
class CachedDerivedArtifact:
    artifact: VersionedArtifact
    view_digest: str
    input_digests: tuple[str, ...]
    operator_key: str


@dataclass(frozen=True)
class MaterializedDagTrace:
    primitive_reads: tuple[str, ...]
    cache_hits: tuple[str, ...]
    cache_misses: tuple[str, ...]
    executed_views: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedDagResult:
    output: VersionedArtifact
    trace: MaterializedDagTrace


ArtifactLoader = Callable[[str], VersionedArtifact]


class MaterializedDagCache:
    def __init__(self) -> None:
        self._entries: dict[
            tuple[str, tuple[str, ...]],
            CachedDerivedArtifact,
        ] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    def get(
        self,
        view_digest: str,
        input_digests: tuple[str, ...],
    ) -> CachedDerivedArtifact | None:
        return self._entries.get((view_digest, input_digests))

    def put(self, entry: CachedDerivedArtifact) -> None:
        self._entries[
            (entry.view_digest, entry.input_digests)
        ] = entry

    def clear(self) -> None:
        self._entries.clear()


def execute_materialized_dag(
    registry: QueryRegistry,
    query: CompiledQuery,
    primitive_loader: ArtifactLoader,
    operators: QueryOperatorRegistry,
    cache: MaterializedDagCache,
    *,
    assessment: QueryExecutionAssessment | None = None,
) -> MaterializedDagResult:
    if assessment is not None and not assessment.runnable:
        raise ValueError(
            "query execution blocked by preflight: "
            + ";".join(assessment.reasons)
        )

    artifacts: dict[str, VersionedArtifact] = {}
    primitive_reads: list[str] = []

    for name in query.primitive_tensors:
        artifact = primitive_loader(name)
        if artifact.name != name:
            raise ValueError(
                f"primitive loader returned {artifact.name!r} for {name!r}"
            )
        artifacts[name] = artifact
        primitive_reads.append(name)

    hits: list[str] = []
    misses: list[str] = []
    executed: list[str] = []

    for view_name in query.view_order:
        view = registry.views.get(view_name)
        if view is None:
            raise ValueError(
                f"compiled query references unknown view {view_name!r}"
            )
        inputs = tuple(artifacts[name] for name in view.inputs)
        input_digests = tuple(item.digest for item in inputs)

        cached = cache.get(view.digest, input_digests)
        if cached is not None:
            artifacts[view_name] = cached.artifact
            hits.append(view_name)
            continue

        misses.append(view_name)
        operator = operators.get(
            view.operator_id,
            view.operator_version,
        )
        value = operator.fn(
            tuple(item.value for item in inputs),
            view.parameters,
        )
        artifact = VersionedArtifact(
            name=view_name,
            value=value,
            digest=_json_digest(
                {
                    "view_digest": view.digest,
                    "input_digests": list(input_digests),
                    "value_digest": value_digest(value),
                }
            ),
        )
        cache.put(
            CachedDerivedArtifact(
                artifact=artifact,
                view_digest=view.digest,
                input_digests=input_digests,
                operator_key=(
                    f"{view.operator_id}@{view.operator_version}"
                ),
            )
        )
        artifacts[view_name] = artifact
        executed.append(view_name)

    output = artifacts.get(query.target)
    if output is None:
        raise ValueError("query target was not materialized")

    return MaterializedDagResult(
        output=output,
        trace=MaterializedDagTrace(
            primitive_reads=tuple(primitive_reads),
            cache_hits=tuple(hits),
            cache_misses=tuple(misses),
            executed_views=tuple(executed),
        ),
    )


def value_digest(value: Any) -> str:
    candidate = getattr(value, "digest", None)
    if isinstance(candidate, str):
        return candidate

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.layout == torch.sparse_coo:
            tensor = tensor.coalesce()
            payload = {
                "kind": "torch_sparse_coo",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "indices": tensor.indices().tolist(),
                "values": tensor.values().tolist(),
            }
        else:
            payload = {
                "kind": "torch_dense",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "values": tensor.tolist(),
            }
        return _json_digest(payload)

    try:
        return _json_digest(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "materialized values must expose a deterministic digest, "
            "be a torch.Tensor, or be JSON-serializable"
        ) from exc


def primitive_artifact(
    name: str,
    value: Any,
    *,
    revision: str,
) -> VersionedArtifact:
    if not revision:
        raise ValueError("primitive revision is required")
    return VersionedArtifact(
        name=name,
        value=value,
        digest=_json_digest(
            {
                "name": name,
                "revision": revision,
                "value_digest": value_digest(value),
            }
        ),
    )


def _json_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
