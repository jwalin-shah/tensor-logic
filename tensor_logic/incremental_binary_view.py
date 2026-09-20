"""Coordinate-level incremental maintenance for binary sparse composition.

For C = A @ B:
- a delta to A[x,y] can only affect C[x,z] where B[y,z] != 0
- a delta to B[y,z] can only affect C[x,z] where A[x,y] != 0

This lets a sparse world update invalidate exact output coordinates rather than
forcing a full materialized-view recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .materialized_view import BinaryWitness
from .tensor_ops import sparse_binary_compose
from .world_tensor import SparseWorldTensor


@dataclass(frozen=True)
class BinaryCoordinateDelta:
    tensor_name: str
    coordinate: tuple[str, str]


@dataclass(frozen=True)
class IncrementalCell:
    coordinate: tuple[str, str]
    value: float
    witnesses: tuple[BinaryWitness, ...]


def affected_binary_outputs(
    left: SparseWorldTensor,
    right: SparseWorldTensor,
    delta: BinaryCoordinateDelta,
) -> tuple[tuple[str, str], ...]:
    _validate_binary(left, right)

    if delta.tensor_name == left.name:
        x, y = delta.coordinate
        left.axes[0].position(x)
        left.axes[1].position(y)
        outputs = [
            (x, z)
            for z in right.axes[1].symbols
            if right.get((y, z)) != 0.0
        ]
        return tuple(outputs)

    if delta.tensor_name == right.name:
        y, z = delta.coordinate
        right.axes[0].position(y)
        right.axes[1].position(z)
        outputs = [
            (x, z)
            for x in left.axes[0].symbols
            if left.get((x, y)) != 0.0
        ]
        return tuple(outputs)

    raise ValueError(
        f"delta tensor {delta.tensor_name!r} is not an input to this view"
    )


def recompute_binary_cells(
    left: SparseWorldTensor,
    right: SparseWorldTensor,
    coordinates: tuple[tuple[str, str], ...],
) -> tuple[IncrementalCell, ...]:
    _validate_binary(left, right)
    cells: list[IncrementalCell] = []

    for output_coordinate in coordinates:
        x, z = output_coordinate
        left.axes[0].position(x)
        right.axes[1].position(z)

        value = 0.0
        witnesses: list[BinaryWitness] = []
        for y in left.axes[1].symbols:
            left_coordinate = (x, y)
            right_coordinate = (y, z)
            left_value = left.get(left_coordinate)
            right_value = right.get(right_coordinate)
            if left_value == 0.0 or right_value == 0.0:
                continue
            value += left_value * right_value
            witnesses.append(
                BinaryWitness(
                    output_coordinate=output_coordinate,
                    intermediate_symbol=y,
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

        cells.append(
            IncrementalCell(
                coordinate=output_coordinate,
                value=value,
                witnesses=tuple(witnesses),
            )
        )

    return tuple(cells)


class IncrementalBinaryView:
    """Mutable output map maintained by exact sparse input deltas."""

    def __init__(
        self,
        left: SparseWorldTensor,
        right: SparseWorldTensor,
    ) -> None:
        _validate_binary(left, right)
        self.left_name = left.name
        self.right_name = right.name
        self._values: dict[tuple[str, str], float] = {}
        self._witnesses: dict[
            tuple[str, str],
            tuple[BinaryWitness, ...],
        ] = {}
        self.full_rebuild(left, right)

    def full_rebuild(
        self,
        left: SparseWorldTensor,
        right: SparseWorldTensor,
    ) -> None:
        _validate_expected_inputs(self, left, right)
        product = sparse_binary_compose(left, right).coalesce()
        coordinates = tuple(
            (
                left.axes[0].symbols[int(indices[0])],
                right.axes[1].symbols[int(indices[1])],
            )
            for indices in product.indices().t().tolist()
        )
        cells = recompute_binary_cells(left, right, coordinates)
        self._values.clear()
        self._witnesses.clear()
        for cell in cells:
            if cell.value != 0.0:
                self._values[cell.coordinate] = cell.value
                self._witnesses[cell.coordinate] = cell.witnesses

    def apply_delta(
        self,
        left: SparseWorldTensor,
        right: SparseWorldTensor,
        delta: BinaryCoordinateDelta,
    ) -> tuple[IncrementalCell, ...]:
        _validate_expected_inputs(self, left, right)
        affected = affected_binary_outputs(left, right, delta)

        # If the changed edge was removed, the current opposite tensor still
        # identifies every potentially affected output coordinate.
        cells = recompute_binary_cells(left, right, affected)
        for cell in cells:
            if cell.value == 0.0:
                self._values.pop(cell.coordinate, None)
                self._witnesses.pop(cell.coordinate, None)
            else:
                self._values[cell.coordinate] = cell.value
                self._witnesses[cell.coordinate] = cell.witnesses
        return cells

    def get(self, coordinate: tuple[str, str]) -> float:
        return self._values.get(coordinate, 0.0)

    def witnesses(
        self,
        coordinate: tuple[str, str],
    ) -> tuple[BinaryWitness, ...]:
        return self._witnesses.get(coordinate, ())

    @property
    def nnz(self) -> int:
        return len(self._values)

    @property
    def coordinates(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._values))


def _validate_binary(
    left: SparseWorldTensor,
    right: SparseWorldTensor,
) -> None:
    if len(left.axes) != 2 or len(right.axes) != 2:
        raise ValueError("incremental binary view requires two 2D tensors")
    if left.axes[1].symbols != right.axes[0].symbols:
        raise ValueError("binary composition inner axes do not match")


def _validate_expected_inputs(
    view: IncrementalBinaryView,
    left: SparseWorldTensor,
    right: SparseWorldTensor,
) -> None:
    _validate_binary(left, right)
    if left.name != view.left_name or right.name != view.right_name:
        raise ValueError("view input tensor names changed")


def _provenance_dict(provenance):
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
