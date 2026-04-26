"""Tests for GF(2^128) and the boolean QuickSilver instantiation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quicksilver.boolean import (
    BBatchedCheck,
    BoolCircuit,
    prove,
    run,
    trusted_dealer_setup,
    verify,
)
from quicksilver.gf2k import GF128, GF2k


# ---- GF(2^128) field ----------------------------------------------------


def test_gf2k_zero_one():
    assert GF128.add(0, 0) == 0
    assert GF128.add(0, 1) == 1
    assert GF128.add(1, 1) == 0  # characteristic 2
    assert GF128.mul(0, 12345) == 0
    assert GF128.mul(1, 12345) == 12345
    assert GF128.mul(12345, 1) == 12345


def test_gf2k_mul_distributive():
    a, b, c = GF128.rand(), GF128.rand(), GF128.rand()
    lhs = GF128.mul(a, b ^ c)
    rhs = GF128.mul(a, b) ^ GF128.mul(a, c)
    assert lhs == rhs


def test_gf2k_mul_associative():
    a, b, c = GF128.rand(), GF128.rand(), GF128.rand()
    assert GF128.mul(GF128.mul(a, b), c) == GF128.mul(a, GF128.mul(b, c))


def test_gf2k_inverse_random():
    for _ in range(20):
        a = GF128.rand_nonzero()
        assert GF128.mul(a, GF128.inv(a)) == 1


def test_gf2k_pow_matches_repeated_mul():
    a = GF128.rand_nonzero()
    expected = 1
    for _ in range(5):
        expected = GF128.mul(expected, a)
    assert GF128.pow(a, 5) == expected


def test_gf2k_squaring_is_frobenius():
    """In characteristic 2, (a+b)^2 = a^2 + b^2."""
    a, b = GF128.rand(), GF128.rand()
    assert GF128.pow(a ^ b, 2) == GF128.pow(a, 2) ^ GF128.pow(b, 2)


# ---- Subspace VOLE ------------------------------------------------------


def test_subspace_vole_correlation_holds():
    p, v = trusted_dealer_setup(64)
    for u_i, v_i, w_i in zip(p.u, p.v, v.w):
        assert u_i in (0, 1)
        expected = ((v.delta if u_i else 0)) ^ v_i
        assert w_i == expected


# ---- Boolean protocol completeness --------------------------------------


def test_completeness_single_and():
    c = BoolCircuit()
    a = c.input()
    b = c.input()
    c.assert_eq_const(c.and_(a, b), 1)
    assert run(c, [1, 1])


def test_completeness_xor_chain_no_ands():
    c = BoolCircuit()
    a, b, d = c.input(), c.input(), c.input()
    s = c.xor(c.xor(a, b), d)
    c.assert_eq_const(s, 0)  # 1 XOR 1 XOR 0 = 0
    assert run(c, [1, 1, 0])


def test_completeness_or_via_xor_and():
    c = BoolCircuit()
    a, b = c.input(), c.input()
    c.assert_eq_const(c.or_(a, b), 1)
    assert run(c, [0, 1])


def test_completeness_three_input_AND():
    c = BoolCircuit()
    a, b, d = c.input(), c.input(), c.input()
    abd = c.and_(c.and_(a, b), d)
    c.assert_eq_const(abd, 1)
    assert run(c, [1, 1, 1])


def test_completeness_full_adder():
    """1-bit full adder: sum = a XOR b XOR cin; cout = (a AND b) OR (cin AND (a XOR b))."""
    c = BoolCircuit()
    a, b, cin = c.input(), c.input(), c.input()
    axb = c.xor(a, b)
    s = c.xor(axb, cin)
    cout = c.or_(c.and_(a, b), c.and_(cin, axb))
    # 1 + 1 + 1 = 11 -> sum=1, cout=1
    c.assert_eq_const(s, 1)
    c.assert_eq_const(cout, 1)
    assert run(c, [1, 1, 1])


def test_completeness_negation():
    c = BoolCircuit()
    a = c.input()
    c.assert_eq_const(c.not_(a), 0)
    assert run(c, [1])


# ---- Soundness ----------------------------------------------------------


def test_soundness_wrong_witness_caught():
    c = BoolCircuit()
    a = c.input()
    b = c.input()
    c.assert_eq_const(c.and_(a, b), 1)
    with pytest.raises(ValueError):
        run(c, [1, 0])  # 1 AND 0 = 0, not 1


def test_soundness_tampered_batched_check_caught():
    c = BoolCircuit()
    a, b = c.input(), c.input()
    c.and_(a, b)
    p, v = trusted_dealer_setup(c.vole_count())
    msg1, batched = prove(c, [1, 1], p)
    chi = GF128.rand_nonzero()
    msg2 = batched(chi)
    bad = BBatchedCheck(U=msg2.U ^ 1, V=msg2.V)
    assert not verify(c, v, msg1, chi, bad)


def test_soundness_tampered_assertion_caught():
    c = BoolCircuit()
    a, b = c.input(), c.input()
    c.assert_eq_const(c.and_(a, b), 1)
    p, v = trusted_dealer_setup(c.vole_count())
    msg1, batched = prove(c, [1, 1], p)
    msg1.assert_openings[0] ^= 1  # flip a bit of the opened MAC tag
    chi = GF128.rand_nonzero()
    msg2 = batched(chi)
    assert not verify(c, v, msg1, chi, msg2)


# ---- Larger circuit -----------------------------------------------------


def test_completeness_4bit_multiplier():
    """Prove knowledge of (x, y) of 4 bits each with x*y = 21 (= 3*7)."""
    c = BoolCircuit()
    x_bits = [c.input() for _ in range(4)]
    y_bits = [c.input() for _ in range(4)]

    # 4x4 -> 8 partial products, schoolbook addition.
    # For an educational test: just verify each output bit equals expected.
    expected_bits = [(21 >> i) & 1 for i in range(8)]

    # Sum of partial products with shifts.
    # Use a simple ripple-carry adder.
    pp = [[c.and_(x_bits[i], y_bits[j]) for j in range(4)] for i in range(4)]
    # partial product i is at column i+0..i+3; build column sums.
    columns: list[list[int]] = [[] for _ in range(8)]
    for i in range(4):
        for j in range(4):
            columns[i + j].append(pp[i][j])

    def half_adder(a, b):
        return c.xor(a, b), c.and_(a, b)

    def full_adder(a, b, ci):
        axb = c.xor(a, b)
        s = c.xor(axb, ci)
        co = c.or_(c.and_(a, b), c.and_(ci, axb))
        return s, co

    # Reduce each column with adders, propagating carries to the next column.
    out_bits: list[int] = []
    for col in range(8):
        bucket = columns[col]
        # add bucket bits one at a time; carry goes to next column
        if not bucket:
            out_bits.append(c.const(0))
            continue
        acc = bucket[0]
        for x in bucket[1:]:
            acc, carry = half_adder(acc, x)
            if col + 1 < 8:
                columns[col + 1].append(carry)
        out_bits.append(acc)

    for bit, expected in zip(out_bits, expected_bits):
        c.assert_eq_const(bit, expected)
    # x = 3 = 0b0011, y = 7 = 0b0111
    witness = [1, 1, 0, 0, 1, 1, 1, 0]
    assert run(c, witness)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
