"""Typed sparse tensor representation for Personal Physics world state.

Numeric values live in tensors. Epistemic metadata lives in a provenance
sidecar keyed by tensor coordinate. This keeps computation efficient without
collapsing evidence, admission, and source authority into a float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable

import torch

from .personal_physics import PersonalPhysicsWorld


@dataclass(frozen=True)
class TensorAxis:
    name: str
    entity_type: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"axis {self.name} contains duplicate symbols")

    @property
    def index(self) -> dict[str, int]:
        return {symbol: i for i, symbol in enumerate(self.symbols)}

    def position(self, symbol: str) -> int:
        try:
            return self.index[symbol]
        except KeyError as exc:
            raise ValueError(
                f"symbol {symbol!r} is not present on axis {self.name!r}"
            ) from exc


@dataclass(frozen=True)
class CoordinateProvenance:
    evidence_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    admission_ref: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SparseWorldTensor:
    """Sparse typed N-ary tensor with provenance keyed per coordinate."""

    def __init__(
        self,
        name: str,
        axes: tuple[TensorAxis, ...],
        *,
        value_kind: str = "boolean",
    ) -> None:
        if not axes:
            raise ValueError("a tensor requires at least one axis")
        if value_kind not in {"boolean", "real"}:
            raise ValueError(
                "value_kind must be either boolean or real"
            )
        self.name = name
        self.axes = axes
        self.value_kind = value_kind
        self._values: dict[tuple[str, ...], float] = {}
        self._provenance: dict[
            tuple[str, ...], CoordinateProvenance
        ] = {}

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis.symbols) for axis in self.axes)

    def set(
        self,
        coordinate: tuple[str, ...],
        value: float,
        *,
        provenance: CoordinateProvenance | None = None,
    ) -> None:
        self._validate_coordinate(coordinate)
        numeric = float(value)
        if self.value_kind == "boolean" and numeric not in {0.0, 1.0}:
            raise ValueError(
                f"boolean tensor {self.name} requires value 0 or 1"
            )
        self._values[coordinate] = numeric
        if provenance is not None:
            self._provenance[coordinate] = provenance

    def get(self, coordinate: tuple[str, ...]) -> float:
        self._validate_coordinate(coordinate)
        return self._values.get(coordinate, 0.0)

    def provenance(
        self,
        coordinate: tuple[str, ...],
    ) -> CoordinateProvenance | None:
        self._validate_coordinate(coordinate)
        return self._provenance.get(coordinate)

    def coordinates(self) -> list[tuple[str, ...]]:
        return sorted(self._values)

    def dense(self) -> torch.Tensor:
        tensor = torch.zeros(self.shape, dtype=torch.float32)
        for coordinate, value in self._values.items():
            indices = tuple(
                axis.position(symbol)
                for axis, symbol in zip(self.axes, coordinate)
            )
            tensor[indices] = value
        return tensor

    def sparse(self) -> torch.Tensor:
        nonzero = [
            (coordinate, value)
            for coordinate, value in sorted(self._values.items())
            if value != 0.0
        ]
        if not nonzero:
            indices = torch.empty(
                (len(self.axes), 0),
                dtype=torch.long,
            )
            values = torch.empty((0,), dtype=torch.float32)
        else:
            indices = torch.tensor(
                [
                    [
                        axis.position(coordinate[axis_index])
                        for coordinate, _ in nonzero
                    ]
                    for axis_index, axis in enumerate(self.axes)
                ],
                dtype=torch.long,
            )
            values = torch.tensor(
                [value for _, value in nonzero],
                dtype=torch.float32,
            )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=self.shape,
        ).coalesce()

    @property
    def digest(self) -> str:
        payload = {
            "name": self.name,
            "value_kind": self.value_kind,
            "axes": [
                {
                    "name": axis.name,
                    "entity_type": axis.entity_type,
                    "symbols": list(axis.symbols),
                }
                for axis in self.axes
            ],
            "values": [
                {
                    "coordinate": list(coordinate),
                    "value": value,
                    "provenance": _provenance_payload(
                        self._provenance.get(coordinate)
                    ),
                }
                for coordinate, value in sorted(self._values.items())
            ],
        }
        return _digest(payload)

    def _validate_coordinate(
        self,
        coordinate: tuple[str, ...],
    ) -> None:
        if len(coordinate) != len(self.axes):
            raise ValueError(
                f"{self.name} expects {len(self.axes)} coordinates, "
                f"got {len(coordinate)}"
            )
        for axis, symbol in zip(self.axes, coordinate):
            axis.position(symbol)


class TensorWorld:
    """Collection of typed sparse tensors sharing named axes."""

    def __init__(self) -> None:
        self.axes: dict[str, TensorAxis] = {}
        self.tensors: dict[str, SparseWorldTensor] = {}

    def add_axis(
        self,
        name: str,
        entity_type: str,
        symbols: Iterable[str],
    ) -> TensorAxis:
        axis = TensorAxis(
            name=name,
            entity_type=entity_type,
            symbols=tuple(sorted(set(symbols))),
        )
        existing = self.axes.get(name)
        if existing is not None and existing != axis:
            raise ValueError(
                f"axis {name!r} already exists with a different definition"
            )
        self.axes[name] = axis
        return axis

    def add_tensor(
        self,
        name: str,
        axis_names: tuple[str, ...],
        *,
        value_kind: str = "boolean",
    ) -> SparseWorldTensor:
        if name in self.tensors:
            raise ValueError(f"tensor {name!r} already exists")
        axes = tuple(self.axes[axis_name] for axis_name in axis_names)
        tensor = SparseWorldTensor(
            name,
            axes,
            value_kind=value_kind,
        )
        self.tensors[name] = tensor
        return tensor

    @property
    def digest(self) -> str:
        return _digest(
            {
                "axes": [
                    {
                        "name": axis.name,
                        "entity_type": axis.entity_type,
                        "symbols": list(axis.symbols),
                    }
                    for axis in sorted(
                        self.axes.values(),
                        key=lambda item: item.name,
                    )
                ],
                "tensors": [
                    {
                        "name": tensor.name,
                        "digest": tensor.digest,
                    }
                    for tensor in sorted(
                        self.tensors.values(),
                        key=lambda item: item.name,
                    )
                ],
            }
        )


def tensorize_personal_world(
    world: PersonalPhysicsWorld,
) -> TensorWorld:
    """Compile admitted Personal Physics facts into typed sparse tensors."""
    tensor_world = TensorWorld()
    by_type: dict[str, list[str]] = {}
    for entity_id, entity_type in world.entities.items():
        by_type.setdefault(entity_type, []).append(entity_id)

    for entity_type, symbols in sorted(by_type.items()):
        tensor_world.add_axis(
            entity_type,
            entity_type,
            symbols,
        )

    for relation_name, schema in sorted(world.relations.items()):
        if (
            schema.subject_type not in tensor_world.axes
            or schema.object_type not in tensor_world.axes
        ):
            continue
        tensor_world.add_tensor(
            relation_name,
            (
                schema.subject_type,
                schema.object_type,
            ),
            value_kind="boolean",
        )

    for fact in sorted(
        world.facts.values(),
        key=lambda item: item.fact_id,
    ):
        if fact.status != "admitted":
            continue
        tensor = tensor_world.tensors[fact.relation]
        tensor.set(
            (fact.subject, fact.object),
            1.0,
            provenance=CoordinateProvenance(
                evidence_refs=fact.evidence_refs,
                source_refs=tuple(
                    ref
                    for ref in (fact.source_ref,)
                    if ref is not None
                ),
                admission_ref=fact.admission_ref,
                valid_from=fact.valid_from,
                valid_until=fact.valid_until,
                confidence=fact.confidence,
                metadata={
                    "fact_id": fact.fact_id,
                    "source_kind": fact.source_kind,
                    **fact.metadata,
                },
            ),
        )

    return tensor_world


def build_world_tensor_schema(
    *,
    people: Iterable[str] = (),
    events: Iterable[str] = (),
    places: Iterable[str] = (),
    time_buckets: Iterable[str] = (),
    modes: Iterable[str] = (),
    projects: Iterable[str] = (),
    goals: Iterable[str] = (),
    tools: Iterable[str] = (),
    tool_calls: Iterable[str] = (),
    evidence: Iterable[str] = (),
    claims: Iterable[str] = (),
) -> TensorWorld:
    """Build the canonical v1 axes and common world tensors."""
    world = TensorWorld()
    axis_values = {
        "Person": people,
        "Event": events,
        "Place": places,
        "TimeBucket": time_buckets,
        "Mode": modes,
        "Project": projects,
        "Goal": goals,
        "Tool": tools,
        "ToolCall": tool_calls,
        "Evidence": evidence,
        "Claim": claims,
    }
    for axis_name, symbols in axis_values.items():
        world.add_axis(axis_name, axis_name, symbols)

    definitions = {
        "attends": (("Person", "Event"), "boolean"),
        "located_at": (("Event", "Place"), "boolean"),
        "starts_at": (("Event", "TimeBucket"), "boolean"),
        "ends_at": (("Event", "TimeBucket"), "boolean"),
        "supports_goal": (("Project", "Goal"), "boolean"),
        "requested_followup": (("Person", "Person"), "boolean"),
        "travel_minutes": (
            ("Place", "Place", "TimeBucket", "Mode"),
            "real",
        ),
        "invoked": (("ToolCall", "Tool"), "boolean"),
        "produced_evidence": (
            ("ToolCall", "Evidence"),
            "boolean",
        ),
        "supports_claim": (
            ("Evidence", "Claim"),
            "boolean",
        ),
        "goal_priority": (("Goal",), "real"),
    }
    for name, (axes, value_kind) in definitions.items():
        world.add_tensor(
            name,
            axes,
            value_kind=value_kind,
        )
    return world


def _provenance_payload(
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


def _digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
