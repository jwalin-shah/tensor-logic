"""Tests for the einsum -> circuit compiler."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quicksilver.circuit import Circuit
from quicksilver.einsum import (
    alloc_constant_tensor,
    alloc_input_tensor,
    assert_einsum_equals,
    compile_einsum,
    evaluate_einsum,
    flatten_row_major,
    parse_einsum,
    resolve_dims,
)
from quicksilver.field import F
from quicksilver.protocol import run as run_protocol


# ---- Parser / dim resolution -------------------------------------------


def test_parse_einsum_matmul():
    inputs, output = parse_einsum("ij,jk->ik")
    assert inputs == ["ij", "jk"]
    assert output == "ik"


def test_resolve_dims_consistent():
    dims = resolve_dims(["ij", "jk"], [(3, 4), (4, 5)])
    assert dims == {"i": 3, "j": 4, "k": 5}


def test_resolve_dims_inconsistent_raises():
    with pytest.raises(ValueError):
        resolve_dims(["ij", "jk"], [(3, 4), (5, 6)])


# ---- Reference evaluator ------------------------------------------------


def test_evaluate_matmul():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    out = evaluate_einsum("ij,jk->ik", [A, B])
    # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert out == [19, 22, 43, 50]


def test_evaluate_dot_product():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert evaluate_einsum("i,i->", [a, b]) == [32]


def test_evaluate_outer_product():
    a = [1, 2]
    b = [3, 4]
    assert evaluate_einsum("i,j->ij", [a, b]) == [3, 4, 6, 8]


def test_evaluate_trace():
    A = [[1, 2], [3, 4]]
    assert evaluate_einsum("ii->", [A]) == [5]


# ---- Compilation: completeness via real prove/verify --------------------


def _prove_einsum(spec: str, shapes, tensors, expected) -> bool:
    c = Circuit()
    inputs = [alloc_input_tensor(c, shape) for shape in shapes]
    assert_einsum_equals(c, spec, inputs, shapes, expected)
    witness = []
    for t in tensors:
        witness += flatten_row_major(t)
    return run_protocol(c, witness)


def test_compile_matmul_2x2():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = evaluate_einsum("ij,jk->ik", [A, B])
    assert _prove_einsum("ij,jk->ik", [(2, 2), (2, 2)], [A, B], expected)


def test_compile_dot_product():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    expected = evaluate_einsum("i,i->", [a, b])
    assert _prove_einsum("i,i->", [(4,), (4,)], [a, b], expected)


def test_compile_three_tensor_contraction():
    """A_ij * B_jk * C_kl -> D_il"""
    A = [[1, 2], [3, 4]]
    B = [[1, 0], [0, 1]]  # identity
    C = [[2, 1], [1, 2]]
    expected = evaluate_einsum("ij,jk,kl->il", [A, B, C])
    assert _prove_einsum(
        "ij,jk,kl->il", [(2, 2), (2, 2), (2, 2)], [A, B, C], expected
    )


def test_compile_with_one_public_tensor():
    """Public B; private A. Prover proves A @ B = expected."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = evaluate_einsum("ij,jk->ik", [A, B])

    c = Circuit()
    A_wires = alloc_input_tensor(c, (2, 2))
    B_wires = alloc_constant_tensor(c, flatten_row_major(B))
    assert_einsum_equals(
        c, "ij,jk->ik", [A_wires, B_wires], [(2, 2), (2, 2)], expected
    )
    witness = flatten_row_major(A)
    assert run_protocol(c, witness)


def test_compile_outer_product():
    a = [1, 2, 3]
    b = [4, 5]
    expected = evaluate_einsum("i,j->ij", [a, b])
    assert _prove_einsum("i,j->ij", [(3,), (2,)], [a, b], expected)


# ---- Soundness: lying about the output should be caught -----------------


def test_compile_matmul_wrong_output_rejected():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    correct = evaluate_einsum("ij,jk->ik", [A, B])
    bad = correct[:]
    bad[0] += 1  # corrupt the (0,0) entry
    with pytest.raises(ValueError):
        # The honest prover walking the circuit will fail at the first
        # assert_zero (the diff wire is nonzero).
        _prove_einsum("ij,jk->ik", [(2, 2), (2, 2)], [A, B], bad)


def test_flatten_row_major():
    assert flatten_row_major([[1, 2, 3], [4, 5, 6]]) == [1, 2, 3, 4, 5, 6]
    assert flatten_row_major([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) == [
        1, 2, 3, 4, 5, 6, 7, 8,
    ]


# ---- Tensor-logic tie-in: grandparent rule ------------------------------


def test_grandparent_rule_einsum():
    """The Datalog rule  Grandparent(x, z) :- Parent(x, y), Parent(y, z)
    is the einsum  G_xz = sum_y P_xy * P_yz.

    Prove that the prover knows a Parent matrix P with a specified
    Grandparent matrix G as its boolean square.
    """
    # 4 people: 0->1->2 and 0->3, 3->2 (so 0 is grandparent of 2 via two paths).
    P = [
        [0, 1, 0, 1],  # 0 is parent of 1 and 3
        [0, 0, 1, 0],  # 1 is parent of 2
        [0, 0, 0, 0],
        [0, 0, 1, 0],  # 3 is parent of 2
    ]
    expected_G = evaluate_einsum("xy,yz->xz", [P, P])
    # G[0][2] should be 2 (paths 0->1->2 and 0->3->2)
    assert expected_G[0 * 4 + 2] == 2
    assert _prove_einsum(
        "xy,yz->xz", [(4, 4), (4, 4)], [P, P], expected_G
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
