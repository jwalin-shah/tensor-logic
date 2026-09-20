import torch

from tensor_logic.incremental_binary_view import (
    BinaryCoordinateDelta,
    IncrementalBinaryView,
    affected_binary_outputs,
)
from tensor_logic.tensor_ops import sparse_binary_compose
from tensor_logic.world_tensor import (
    CoordinateProvenance,
    SparseWorldTensor,
    TensorAxis,
)


def _tensors():
    a = TensorAxis("A", "A", ("a0", "a1", "a2"))
    b = TensorAxis("B", "B", ("b0", "b1", "b2"))
    c = TensorAxis("C", "C", ("c0", "c1", "c2", "c3"))

    left = SparseWorldTensor("left", (a, b), value_kind="real")
    right = SparseWorldTensor("right", (b, c), value_kind="real")

    left.set(
        ("a0", "b0"),
        1.0,
        provenance=CoordinateProvenance(evidence_refs=("L0",)),
    )
    left.set(("a0", "b1"), 2.0)
    left.set(("a1", "b2"), 3.0)

    right.set(("b0", "c0"), 4.0)
    right.set(
        ("b1", "c1"),
        5.0,
        provenance=CoordinateProvenance(evidence_refs=("R1",)),
    )
    right.set(("b2", "c2"), 6.0)
    return left, right


def _dense_incremental(view, left, right):
    out = torch.zeros(
        (len(left.axes[0].symbols), len(right.axes[1].symbols)),
        dtype=torch.float32,
    )
    for x, z in view.coordinates:
        out[
            left.axes[0].position(x),
            right.axes[1].position(z),
        ] = view.get((x, z))
    return out


def test_left_coordinate_delta_only_touches_reachable_output_row_cells():
    left, right = _tensors()

    affected = affected_binary_outputs(
        left,
        right,
        BinaryCoordinateDelta("left", ("a0", "b1")),
    )

    assert affected == (("a0", "c1"),)


def test_right_coordinate_delta_only_touches_reachable_output_column_cells():
    left, right = _tensors()

    affected = affected_binary_outputs(
        left,
        right,
        BinaryCoordinateDelta("right", ("b0", "c0")),
    )

    assert affected == (("a0", "c0"),)


def test_incremental_add_matches_full_sparse_recompute():
    left, right = _tensors()
    view = IncrementalBinaryView(left, right)

    right.set(("b1", "c3"), 7.0)
    cells = view.apply_delta(
        left,
        right,
        BinaryCoordinateDelta("right", ("b1", "c3")),
    )

    assert tuple(cell.coordinate for cell in cells) == (("a0", "c3"),)
    assert view.get(("a0", "c3")) == 14.0

    full = sparse_binary_compose(left, right).to_dense()
    assert torch.allclose(
        _dense_incremental(view, left, right),
        full,
    )


def test_incremental_remove_matches_full_sparse_recompute():
    left, right = _tensors()
    view = IncrementalBinaryView(left, right)

    assert view.get(("a0", "c1")) == 10.0
    right.remove(("b1", "c1"))
    cells = view.apply_delta(
        left,
        right,
        BinaryCoordinateDelta("right", ("b1", "c1")),
    )

    assert tuple(cell.coordinate for cell in cells) == (("a0", "c1"),)
    assert cells[0].value == 0.0
    assert view.get(("a0", "c1")) == 0.0

    full = sparse_binary_compose(left, right).to_dense()
    assert torch.allclose(
        _dense_incremental(view, left, right),
        full,
    )


def test_incremental_value_change_matches_full_sparse_recompute():
    left, right = _tensors()
    view = IncrementalBinaryView(left, right)

    left.set(("a0", "b0"), 10.0)
    view.apply_delta(
        left,
        right,
        BinaryCoordinateDelta("left", ("a0", "b0")),
    )

    assert view.get(("a0", "c0")) == 40.0
    full = sparse_binary_compose(left, right).to_dense()
    assert torch.allclose(
        _dense_incremental(view, left, right),
        full,
    )


def test_incremental_witnesses_preserve_source_evidence():
    left, right = _tensors()
    view = IncrementalBinaryView(left, right)

    witnesses = view.witnesses(("a0", "c1"))

    assert len(witnesses) == 1
    assert witnesses[0].intermediate_symbol == "b1"
    assert witnesses[0].right_provenance["evidence_refs"] == ["R1"]


def test_unrelated_delta_tensor_is_rejected():
    left, right = _tensors()

    try:
        affected_binary_outputs(
            left,
            right,
            BinaryCoordinateDelta("other", ("a0", "b0")),
        )
    except ValueError as exc:
        assert "not an input" in str(exc)
    else:
        raise AssertionError("unrelated delta should fail")
