"""Tests for the QuickSilver implementation."""

from __future__ import annotations

import os
import sys

# Allow running with `python tests/test_quicksilver.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quicksilver.circuit import Circuit
from quicksilver.field import F
from quicksilver.itmac import (
    ProverWire,
    VerifierWire,
    prover_commit,
    verifier_receive,
)
from quicksilver.polynomial import (
    Polynomial,
    prove_polys,
    run_poly_check,
    verify_polys,
)
from quicksilver.protocol import (
    BatchedCheck,
    CommitMessage,
    _ProverWalker,
    _VerifierWalker,
    prove,
    run,
    verify,
)
from quicksilver.vole import trusted_dealer_setup


# ---- Field --------------------------------------------------------------


def test_field_inverse():
    for _ in range(50):
        a = F.rand_nonzero()
        assert F.mul(a, F.inv(a)) == 1


def test_field_arith_in_range():
    for _ in range(20):
        a, b = F.rand(), F.rand()
        for x in (F.add(a, b), F.sub(a, b), F.mul(a, b), F.neg(a)):
            assert 0 <= x < F.p


# ---- VOLE / IT-MAC ------------------------------------------------------


def test_vole_correlation_holds():
    p_share, v_share = trusted_dealer_setup(64)
    for u, v, w in zip(p_share.u, p_share.v, v_share.w):
        assert w == F.add(F.mul(u, v_share.delta), v)


def test_itmac_relation_holds():
    p_share, v_share = trusted_dealer_setup(1)
    x = F.rand()
    pw, d = prover_commit(x, p_share.u[0], p_share.v[0])
    vw = verifier_receive(d, v_share.w[0], v_share.delta)
    # K == M + Delta * x
    assert vw.k == F.add(pw.m, F.mul(v_share.delta, pw.x))


def test_itmac_addition_preserves_relation():
    p_share, v_share = trusted_dealer_setup(2)
    x, y = F.rand(), F.rand()
    pw_x, dx = prover_commit(x, p_share.u[0], p_share.v[0])
    pw_y, dy = prover_commit(y, p_share.u[1], p_share.v[1])
    vw_x = verifier_receive(dx, v_share.w[0], v_share.delta)
    vw_y = verifier_receive(dy, v_share.w[1], v_share.delta)

    pw_sum = pw_x.add(pw_y)
    vw_sum = vw_x.add(vw_y)
    assert vw_sum.k == F.add(pw_sum.m, F.mul(v_share.delta, pw_sum.x))


# ---- Circuit / protocol completeness -----------------------------------


def test_completeness_single_mul():
    c = Circuit()
    x = c.input()
    y = c.input()
    z = c.mul(x, y)
    c.assert_eq(z, 56)
    assert run(c, [7, 8])


def test_completeness_linear_only():
    c = Circuit()
    a = c.input()
    b = c.input()
    s = c.add(a, b)
    s2 = c.mul_const(s, 3)
    s3 = c.add_const(s2, 1)
    c.assert_eq(s3, 3 * (10 + 5) + 1)
    assert run(c, [10, 5])


def test_completeness_polynomial_circuit():
    # Prove knowledge of x with x^3 + x + 5 = y for public y.
    x_val = 17
    y_val = (x_val ** 3 + x_val + 5) % F.p
    c = Circuit()
    x = c.input()
    x2 = c.mul(x, x)
    x3 = c.mul(x2, x)
    s = c.add(x3, x)
    s = c.add_const(s, 5)
    c.assert_eq(s, y_val)
    assert run(c, [x_val])


def test_completeness_many_muls():
    # Inner product: <a, b> = c.
    n = 16
    a = [F.rand() for _ in range(n)]
    b = [F.rand() for _ in range(n)]
    expected = 0
    for ai, bi in zip(a, b):
        expected = F.add(expected, F.mul(ai, bi))

    c = Circuit()
    a_w = [c.input() for _ in range(n)]
    b_w = [c.input() for _ in range(n)]
    acc = c.const(0)
    for ai, bi in zip(a_w, b_w):
        prod = c.mul(ai, bi)
        acc = c.add(acc, prod)
    c.assert_eq(acc, expected)
    assert run(c, a + b)


# ---- Soundness ----------------------------------------------------------


def _build_mul_circuit(target_z: int) -> Circuit:
    c = Circuit()
    x = c.input()
    y = c.input()
    z = c.mul(x, y)
    c.assert_eq(z, target_z)
    return c


def test_soundness_wrong_witness_caught_at_assertion():
    # Prover lies about x*y by claiming a different z. The assertion check
    # will fail because the opened tag won't match the verifier's key.
    c = _build_mul_circuit(target_z=99)  # but real x*y will be 56
    # Witness is (7, 8) -> 7*8 = 56, not 99. Honest prover walking this
    # circuit will produce assertion-zero on (56 - 99), which is nonzero;
    # `open_to_zero` raises.
    with pytest.raises(ValueError):
        run(c, [7, 8])


def test_soundness_prover_lies_about_mul_output():
    """A cheating prover who tries to make 7*8 == 99 pass.

    They have to commit to z=99 but claim it equals 7*8. We simulate by
    hand-crafting a malicious commit message and watching the batched
    multiplication check reject.
    """
    c = Circuit()
    x = c.input()
    y = c.input()
    z = c.mul(x, y)
    # Note: no assertion, so the only thing that catches the lie is the
    # batched mul check.

    p_share, v_share = trusted_dealer_setup(c.vole_count())
    delta = v_share.delta

    # Honest commits for x=7, y=8.
    x_val, y_val = 7, 8
    fake_z = 99  # the lie
    u0, v0 = p_share.u[0], p_share.v[0]
    u1, v1 = p_share.u[1], p_share.v[1]
    u2, v2 = p_share.u[2], p_share.v[2]
    d_x = F.sub(x_val, u0)
    d_y = F.sub(y_val, u1)
    d_z = F.sub(fake_z, u2)  # commit to fake z
    msg1 = CommitMessage(d_values=[d_x, d_y, d_z], assert_openings=[])

    # Prover then sends a batched check assuming z = x*y = 56, i.e., uses
    # M values consistent with z = 56. (Equivalently, an honest A0/A1 over
    # the fake commitment.)
    Mx, My = v0, v1
    Mz_real = v2  # tag the prover holds for the committed value (which is 99)
    A0 = F.mul(Mx, My)
    A1 = F.sub(F.add(F.mul(x_val, My), F.mul(y_val, Mx)), Mz_real)

    chi = F.rand_nonzero()
    # Mask using the last VOLE element (index 3 = num_inputs + num_muls).
    a, b = p_share.u[3], p_share.v[3]
    U = F.add(F.mul(chi, A0), b)
    V = F.add(F.mul(chi, A1), a)
    msg2 = BatchedCheck(U=U, V=V)

    assert not verify(c, v_share, msg1, chi, msg2)


def test_soundness_tampered_assertion_caught():
    c = Circuit()
    x = c.input()
    y = c.input()
    z = c.mul(x, y)
    c.assert_eq(z, 56)

    p_share, v_share = trusted_dealer_setup(c.vole_count())
    msg1, batched = prove(c, [7, 8], p_share)
    # Tamper with the opened tag.
    msg1.assert_openings[0] = F.add(msg1.assert_openings[0], 1)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    assert not verify(c, v_share, msg1, chi, msg2)


def test_soundness_tampered_batched_check_caught():
    c = Circuit()
    x = c.input()
    y = c.input()
    z = c.mul(x, y)
    p_share, v_share = trusted_dealer_setup(c.vole_count())
    msg1, batched = prove(c, [7, 8], p_share)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    msg2 = BatchedCheck(U=F.add(msg2.U, 1), V=msg2.V)
    assert not verify(c, v_share, msg1, chi, msg2)


# ---- Polynomial extension ----------------------------------------------


def _commit_values(values, share, delta):
    """Helper: produce ProverWire / VerifierWire dicts keyed by 0..n-1."""
    pwires, vwires = {}, {}
    for i, val in enumerate(values):
        pw, d = prover_commit(val, share.u[i], share.v[i])
        vw = verifier_receive(d, F.add(F.mul(share.u[i], delta), share.v[i]), delta)
        pwires[i] = pw
        vwires[i] = vw
    return pwires, vwires


def test_polynomial_completeness_cubic():
    # Prove x^3 - 8 == 0 with x=2.
    p_share, v_share = trusted_dealer_setup(1)
    delta = v_share.delta
    pwires, vwires = _commit_values([2], p_share, delta)
    poly = Polynomial(terms=((1, (0, 0, 0)), (-8, ())))
    assert run_poly_check([poly], pwires, vwires, delta)


def test_polynomial_completeness_multivariate():
    # Prove x*y*z + 2*x - 3 == 0 with x=1, y=1, z=1: 1+2-3 = 0.
    p_share, v_share = trusted_dealer_setup(3)
    delta = v_share.delta
    pwires, vwires = _commit_values([1, 1, 1], p_share, delta)
    poly = Polynomial(terms=((1, (0, 1, 2)), (2, (0,)), (-3, ())))
    assert run_poly_check([poly], pwires, vwires, delta)


def test_polynomial_batched_quadratic():
    # Three quadratic constraints, all zero on the witness.
    p_share, v_share = trusted_dealer_setup(3)
    delta = v_share.delta
    a, b, c = 5, 7, 11
    pwires, vwires = _commit_values([a, b, c], p_share, delta)
    polys = [
        Polynomial(terms=((1, (0, 0)), (-a * a, ()))),  # x0^2 - 25
        Polynomial(terms=((1, (1, 1)), (-b * b, ()))),  # x1^2 - 49
        Polynomial(terms=((1, (2, 2)), (-c * c, ()))),  # x2^2 - 121
    ]
    assert run_poly_check(polys, pwires, vwires, delta)


def test_polynomial_soundness_caught():
    # x is committed to value 2, but we try to prove x^3 == 9 (false: 2^3 = 8).
    p_share, v_share = trusted_dealer_setup(1)
    delta = v_share.delta
    pwires, vwires = _commit_values([2], p_share, delta)
    poly = Polynomial(terms=((1, (0, 0, 0)), (-9, ())))

    # Honest prove call ignores soundness and just reports A_0..A_{d-1}.
    # The verifier's check should still fail because A_d != 0.
    mask_p, mask_v = trusted_dealer_setup(2, delta=delta)  # d-1 = 2 mask elements
    chi = F.rand_nonzero()
    proof = prove_polys([poly], pwires, chi, mask_p)
    assert not verify_polys([poly], vwires, chi, proof, mask_v)


# ---- Sanity: low-level walker computes coherent state ------------------


def test_walker_produces_consistent_views():
    c = Circuit()
    x = c.input()
    y = c.input()
    z = c.mul(x, y)
    s = c.add_const(z, 5)
    c.assert_eq(s, 7 * 8 + 5)

    p_share, v_share = trusted_dealer_setup(c.vole_count())
    pw = _ProverWalker(circuit=c, witness=[7, 8], share=p_share)
    msg1 = pw.commit()
    vw = _VerifierWalker(circuit=c, share=v_share)
    vw.receive(msg1)

    # Spot check: the IT-MAC relation holds on every wire.
    for wid, prover_wire in pw.wires.items():
        verifier_wire = vw.wires[wid]
        expected_k = F.add(prover_wire.m, F.mul(v_share.delta, prover_wire.x))
        assert verifier_wire.k == expected_k


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
