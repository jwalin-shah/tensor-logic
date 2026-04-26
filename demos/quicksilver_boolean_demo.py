"""Boolean QuickSilver demo.

Same protocol as the prime-field version, but wires are bits and
IT-MAC tags live in GF(2^128). Three vignettes:

1. Knowledge of factors of a 16-bit number, computed bit-by-bit
   inside a schoolbook 8x8 multiplier.
2. Knowledge of an XOR-preimage: prove a SHA-like mixing function
   maps a secret ``x`` to a public ``y``, with no information about
   ``x`` revealed.
3. Boolean-circuit scaling: doubles in cost per added bit of width.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List

from quicksilver.boolean import BoolCircuit, run


def _ripple_adder(c: BoolCircuit, a_bits, b_bits) -> List[int]:
    """Bitwise ripple-carry adder. Output has len(a_bits)+1 bits."""
    out = []
    carry = c.const(0)
    for a, b in zip(a_bits, b_bits):
        axb = c.xor(a, b)
        s = c.xor(axb, carry)
        carry = c.or_(c.and_(a, b), c.and_(carry, axb))
        out.append(s)
    out.append(carry)
    return out


def _multiplier(c: BoolCircuit, x_bits, y_bits) -> List[int]:
    """Schoolbook unsigned multiplier. Output is len(x)+len(y) bits."""
    n_x, n_y = len(x_bits), len(y_bits)
    columns: list[list[int]] = [[] for _ in range(n_x + n_y)]
    for i, x in enumerate(x_bits):
        for j, y in enumerate(y_bits):
            columns[i + j].append(c.and_(x, y))
    out_bits = []
    for col in range(n_x + n_y):
        bucket = columns[col]
        if not bucket:
            out_bits.append(c.const(0))
            continue
        acc = bucket[0]
        for x in bucket[1:]:
            s = c.xor(acc, x)
            carry = c.and_(acc, x)
            acc = s
            if col + 1 < n_x + n_y:
                columns[col + 1].append(carry)
        out_bits.append(acc)
    return out_bits


def _bits_of(x: int, n: int) -> List[int]:
    return [(x >> i) & 1 for i in range(n)]


def _assert_eq_int(c: BoolCircuit, bits, expected: int) -> None:
    for i, b in enumerate(bits):
        c.assert_eq_const(b, (expected >> i) & 1)


def demo_factorisation_8bit() -> None:
    print("=" * 64)
    print("Demo 1: knowledge of 8-bit factors, computed in a boolean multiplier")
    print("=" * 64)
    p, q = 13, 17
    N = p * q
    print(f"  Public N = {N} = {p} x {q}")
    print(f"  Witness: bit decomposition of p and q (8 bits each)")

    c = BoolCircuit()
    p_bits = [c.input() for _ in range(8)]
    q_bits = [c.input() for _ in range(8)]
    out = _multiplier(c, p_bits, q_bits)
    _assert_eq_int(c, out, N)

    print(f"  circuit: {c.num_inputs} inputs, {c.num_ands} AND gates, "
          f"{c.num_asserts} asserts")
    print(f"  VOLE elements consumed: {c.vole_count()}")

    witness = _bits_of(p, 8) + _bits_of(q, 8)
    t0 = time.time()
    ok = run(c, witness)
    print(f"  verifier accepts: {ok}   (in {(time.time() - t0)*1000:.1f} ms)\n")
    assert ok


def demo_xor_preimage() -> None:
    print("=" * 64)
    print("Demo 2: knowledge of XOR-mixer preimage")
    print("=" * 64)
    # A toy mixing function: y = x XOR (x rotated left by 3) XOR constant.
    # Prover knows x; verifier knows y.
    n = 16
    K = 0xA5A5
    x = 0x4321
    y = x ^ ((x << 3 | x >> (n - 3)) & ((1 << n) - 1)) ^ K
    print(f"  Public y = 0x{y:04X}")
    print(f"  Mixer: y = x XOR rot_left(x, 3) XOR 0x{K:04X}")
    print(f"  (Secret x = 0x{x:04X})")

    c = BoolCircuit()
    x_bits = [c.input() for _ in range(n)]
    rot_bits = [x_bits[(i - 3) % n] for i in range(n)]
    out_bits = []
    for i in range(n):
        s = c.xor(x_bits[i], rot_bits[i])
        s = c.xor_const(s, (K >> i) & 1)
        out_bits.append(s)
    _assert_eq_int(c, out_bits, y)

    print(f"  circuit: {c.num_inputs} inputs, {c.num_ands} AND gates "
          f"(all linear -> 0 ANDs)")

    witness = _bits_of(x, n)
    t0 = time.time()
    ok = run(c, witness)
    print(f"  verifier accepts: {ok}   (in {(time.time() - t0)*1000:.1f} ms)\n")
    assert ok


def demo_scaling() -> None:
    print("=" * 64)
    print("Demo 3: multiplier scaling -- (n_bit_factor, AND gates, time)")
    print("=" * 64)
    print(f"  {'bits':>5} {'inputs':>8} {'ANDs':>6} {'asserts':>8} {'time(ms)':>10}")
    for nbits in (4, 6, 8, 10):
        c = BoolCircuit()
        p_bits = [c.input() for _ in range(nbits)]
        q_bits = [c.input() for _ in range(nbits)]
        out = _multiplier(c, p_bits, q_bits)
        # Pick small primes that fit.
        if nbits == 4:
            P, Q = 3, 5
        elif nbits == 6:
            P, Q = 7, 11
        elif nbits == 8:
            P, Q = 13, 17
        else:
            P, Q = 31, 29
        _assert_eq_int(c, out, P * Q)
        witness = _bits_of(P, nbits) + _bits_of(Q, nbits)
        t0 = time.time()
        ok = run(c, witness)
        dt = (time.time() - t0) * 1000
        assert ok
        print(f"  {nbits:>5} {c.num_inputs:>8} {c.num_ands:>6} "
              f"{c.num_asserts:>8} {dt:>10.1f}")
    print()


if __name__ == "__main__":
    demo_factorisation_8bit()
    demo_xor_preimage()
    demo_scaling()
    print("All boolean QuickSilver demos passed.")
