"""QuickSilver ZK demo.

Three vignettes that exercise the full prover/verifier loop:

1. Knowledge of factors: prover convinces verifier they know p, q with
   p * q == N, without revealing p or q.
2. Knowledge of a cubic root: x^3 + x + 5 == y for a specified y.
3. Polynomial-extension demo: prove three quadratic constraints with a
   single batched degree-2 check, no extra circuit wires.

Run from the repo root:

    python demos/quicksilver_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from quicksilver.circuit import Circuit
from quicksilver.field import F
from quicksilver.itmac import prover_commit, verifier_receive
from quicksilver.polynomial import Polynomial, run_poly_check
from quicksilver.protocol import run
from quicksilver.vole import trusted_dealer_setup


def demo_factorisation() -> None:
    print("=" * 60)
    print("Demo 1: knowledge of factorisation")
    print("=" * 60)
    p, q = 1_000_003, 999_983  # both prime
    N = p * q
    print(f"  Public N = {N}")
    print(f"  Prover claims to know p, q with p * q = N")
    print(f"  (Secret witness: p={p}, q={q})\n")

    c = Circuit()
    pw = c.input()
    qw = c.input()
    prod = c.mul(pw, qw)
    c.assert_eq(prod, N)

    print(f"  Circuit: {len(c.gates)} gates, {c.num_inputs} inputs, "
          f"{c.num_muls} mul gates, {c.num_asserts} asserts")
    print(f"  VOLE elements consumed: {c.vole_count()}")

    t0 = time.time()
    ok = run(c, [p, q])
    dt = (time.time() - t0) * 1000
    print(f"\n  Verifier accepts: {ok}   (in {dt:.2f} ms)\n")
    assert ok

    # Soundness sanity: try to convince with wrong factors.
    print("  Sanity: try the same with a wrong witness (p=2, q=3) ...")
    try:
        run(c, [2, 3])
        print("    UNEXPECTED: ran to completion (should have failed!)")
    except ValueError as e:
        print(f"    rejected at assertion-zero check: {e}")
    print()


def demo_cubic_root() -> None:
    print("=" * 60)
    print("Demo 2: knowledge of x with x^3 + x + 5 = y")
    print("=" * 60)
    x = 17
    y = (x ** 3 + x + 5) % F.p
    print(f"  Public y = {y}")
    print(f"  (Secret witness: x = {x})\n")

    c = Circuit()
    xw = c.input()
    x2 = c.mul(xw, xw)
    x3 = c.mul(x2, xw)
    s = c.add(x3, xw)
    s = c.add_const(s, 5)
    c.assert_eq(s, y)

    print(f"  Circuit: {c.num_muls} mul gates, {c.num_inputs} private inputs")
    t0 = time.time()
    ok = run(c, [x])
    dt = (time.time() - t0) * 1000
    print(f"  Verifier accepts: {ok}   (in {dt:.2f} ms)\n")
    assert ok


def demo_polynomial_extension() -> None:
    print("=" * 60)
    print("Demo 3: degree-d polynomial check (no extra mul gates)")
    print("=" * 60)
    # Prove three independent quadratic relations on three committed
    # values, all batched into a single degree-2 polynomial check.
    a, b, c = 5, 7, 11
    polys = [
        Polynomial(terms=((1, (0, 0)), (-(a * a), ()))),
        Polynomial(terms=((1, (1, 1)), (-(b * b), ()))),
        Polynomial(terms=((1, (2, 2)), (-(c * c), ()))),
    ]
    print(f"  Constraints (all on committed wires):")
    for w, val in zip("abc", (a, b, c)):
        print(f"    wire_{w}^2 == {val * val}")
    print()

    p_share, v_share = trusted_dealer_setup(3)
    delta = v_share.delta
    pwires, vwires = {}, {}
    for i, val in enumerate((a, b, c)):
        pw, d = prover_commit(val, p_share.u[i], p_share.v[i])
        vw = verifier_receive(d, v_share.w[i], delta)
        pwires[i] = pw
        vwires[i] = vw

    t0 = time.time()
    ok = run_poly_check(polys, pwires, vwires, delta)
    dt = (time.time() - t0) * 1000
    print(f"  Batched check accepts: {ok}   (in {dt:.2f} ms)")
    print(f"  Communication: d = 2 field elements, regardless of how many\n"
          f"  constraints share that degree.\n")
    assert ok


if __name__ == "__main__":
    demo_factorisation()
    demo_cubic_root()
    demo_polynomial_extension()
    print("All demos passed.")
