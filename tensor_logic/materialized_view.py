"""Proof-carrying materialized tensor views.

Fast query execution and traceability are separate concerns:
- live cache validity uses cheap tensor revision tokens;
- historical replay uses a supplied world snapshot digest;
- individual output cells can enumerate exact input-coordinate witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import torch

from .tensor_ops import sparse_binary_compose
from .world_tensor import CoordinateProvenance, SparseWorldTensor


@dataclass(frozen=True)
class TensorQueryPlan:
    plan_id: str
    operator_id: str
    operator_version: str
    input_tensors: tuple[str, ...]
    output_tensor: str
    freshness_requirements: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "plan_id": self.plan_id,
                "operator_id": self.operator_id,
                "operator_version": self.operator_version,
                "input_tensors": list(self.input_tensors),
                "output_tensor": self.output_tensor,
                "freshness_requirements": list(
                    self.freshness_requirements
                ),
                "parameters": self.parameters,
            }
        )


@dataclass(frozen=True)
class MaterializedView:
    view_name: str
    snapshot_digest: str
    plan_digest: str
    operator_id: str
    operator_version: str
    input_revision_tokens: tuple[str, ...]
    output_shape: tuple[int, ...]
    output_nnz: int
    output_digest: str

    @property
    def digest(self) -> str:
        return _digest(
            {
                "view_name": self.view_name,
                "snapshot_digest": self.snapshot_digest,
                "plan_digest": self.plan_digest,
                "operator_id": self.operator_id,
                "operator_version": self.operator_version,
                "input_revision_tokens": list(
                    self.input_revision_tokens
                ),
                "output_shape": list(self.output_shape),
                "output_nnz": self.output_nnz,
                "output_digest": self.output_digest,
            }
        )


@dataclass(frozen=True)
class BinaryWitness:
    output_coordinate: tuple[str, str]
    intermediate_symbol: str
    left_coordinate: tuple[str, str]
    right_coordinate: tuple[str, str]
    left_value: float
    right_value: float
    left_provenance: dict[str, Any] | None
    right_provenance: dict[str, Any] | None


@dataclass
class CachedView:
    metadata: MaterializedView
    result: torch.Tensor


class ViewCache:
    """Revision-keyed cache with explicit dependency index."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], CachedView] = {}
        self._keys_by_input: dict[str, set[tuple[Any, ...]]] = {}

    def materialize_binary(
        self,
        plan: TensorQueryPlan,
        left: SparseWorldTensor,
        right: SparseWorldTensor,
        *,
        snapshot_digest: str,
    ) -> tuple[CachedView, bool]:
        if plan.input_tensors != (left.name, right.name):
            raise ValueError(
                "query plan input names do not match supplied tensors"
            )

        key = (
            snapshot_digest,
            plan.digest,
            left.revision_token,
            right.revision_token,
        )
        existing = self._entries.get(key)
        if existing is not None:
            return existing, True

        result = sparse_binary_compose(left, right).coalesce()
        metadata = MaterializedView(
            view_name=plan.output_tensor,
            snapshot_digest=snapshot_digest,
            plan_digest=plan.digest,
            operator_id=plan.operator_id,
            operator_version=plan.operator_version,
            input_revision_tokens=(
                left.revision_token,
                right.revision_token,
            ),
            output_shape=tuple(result.shape),
            output_nnz=result._nnz(),
            output_digest=_sparse_digest(result),
        )
        cached = CachedView(metadata=metadata, result=result)
        self._entries[key] = cached
        for name in plan.input_tensors:
            self._keys_by_input.setdefault(name, set()).add(key)
        return cached, False

    def invalidate_input(self, tensor_name: str) -> int:
        keys = self._keys_by_input.pop(tensor_name, set())
        removed = 0
        for key in keys:
            cached = self._entries.pop(key, None)
            if cached is None:
                continue
            removed += 1
            for input_name in _input_names_from_keyed_view(cached):
                if input_name == tensor_name:
                    continue
                bucket = self._keys_by_input.get(input_name)
                if bucket is not None:
                    bucket.discard(key)
        return removed

    @property
    def size(self) -> int:
        return len(self._entries)


def binary_composition_witnesses(
    view: CachedView,
    left: SparseWorldTensor,
    right: SparseWorldTensor,
    output_coordinate: tuple[str, str],
) -> list[BinaryWitness]:
    """Enumerate exact A[x,y] / B[y,z] witnesses for one A@B output cell."""
    expected = (
        left.revision_token,
        right.revision_token,
    )
    if view.metadata.input_revision_tokens != expected:
        raise ValueError(
            "view is stale relative to supplied input tensors"
        )
    if len(left.axes) != 2 or len(right.axes) != 2:
        raise ValueError("binary witness extraction requires 2D tensors")
    if left.axes[1].symbols != right.axes[0].symbols:
        raise ValueError("binary witness inner axes do not match")

    left_outer, right_outer = output_coordinate
    left.axes[0].position(left_outer)
    right.axes[1].position(right_outer)

    witnesses: list[BinaryWitness] = []
    for intermediate in left.axes[1].symbols:
        left_coordinate = (left_outer, intermediate)
        right_coordinate = (intermediate, right_outer)
        left_value = left.get(left_coordinate)
        if left_value == 0.0:
            continue
        right_value = right.get(right_coordinate)
        if right_value == 0.0:
            continue
        witnesses.append(
            BinaryWitness(
                output_coordinate=output_coordinate,
                intermediate_symbol=intermediate,
                left_coordinate=left_coordinate,
                right_coordinate=right_coordinate,
                left_value=left_value,
                right_value=right_value,
                left_provenance=_provenance_dict(
                    left.provenance(left_coordinate)
                ),
                right_provenance=_provenance_dict(
                    right.provenance(right_coordinate)
                ),
            )
        )
    return witnesses


def _input_names_from_keyed_view(
    cached: CachedView,
) -> tuple[str, ...]:
    return tuple(
        token.split(":", 1)[0]
        for token in cached.metadata.input_revision_tokens
    )


def _provenance_dict(
    provenance: CoordinateProvenance | None,
) -> dict[str, Any] | None:
    if provenance is None:
        return None
    return {
        "evidence_refs": list(provenance.evidence_refs),
        "source_refs": list(provenance.source_refs),
        "admission_ref": provenance.admission_ref,
        "valid_from": provenance.valid_from,
        "valid_until": provenance.valid_until,
        "confidence": provenance.confidence,
        "metadata": provenance.metadata,
    }


def _sparse_digest(tensor: torch.Tensor) -> str:
    tensor = tensor.coalesce()
    return _digest(
        {
            "shape": list(tensor.shape),
            "indices": tensor.indices().tolist(),
            "values": tensor.values().tolist(),
        }
    )


def _digest(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
