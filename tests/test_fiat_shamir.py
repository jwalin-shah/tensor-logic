"""Tests for the Fiat-Shamir non-interactive variant."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quicksilver.circuit import Circuit
from quicksilver.field import F
from quicksilver.fiat_shamir import (
    NIProof,
    Transcript,
    prove_ni,
    run_ni,
    verify_ni,
)
from quicksilver.protocol import BatchedCheck
from quicksilver.vole import trusted_dealer_setup


def _factor_circuit(target: int) -> Circuit:
    c = Circuit()
    p = c.input()
    q = c.input()
    c.assert_eq(c.mul(p, q), target)
    return c


def test_ni_completeness():
    c = _factor_circuit(56)
    ok, _ = run_ni(c, [7, 8])
    assert ok


def test_ni_proof_is_deterministic_given_setup():
    c = _factor_circuit(56)
    p_share, v_share = trusted_dealer_setup(c.vole_count())
    proof_a = prove_ni(c, [7, 8], p_share)
    proof_b = prove_ni(c, [7, 8], p_share)
    # Same setup + same witness -> same proof (Fiat-Shamir is deterministic
    # once the prover side has its share).
    assert proof_a.msg1.d_values == proof_b.msg1.d_values
    assert proof_a.msg2.U == proof_b.msg2.U
    assert proof_a.msg2.V == proof_b.msg2.V
    assert verify_ni(c, v_share, proof_a)


def test_ni_tamper_msg2_rejected():
    c = _factor_circuit(56)
    p_share, v_share = trusted_dealer_setup(c.vole_count())
    proof = prove_ni(c, [7, 8], p_share)
    bad = NIProof(
        msg1=proof.msg1,
        msg2=BatchedCheck(U=F.add(proof.msg2.U, 1), V=proof.msg2.V),
    )
    assert not verify_ni(c, v_share, bad)


def test_ni_tamper_msg1_invalidates_chi():
    """Mutating msg1 changes the challenge but msg2 was bound to the
    original chi. The verifier recomputes chi from the new msg1 and
    rejects."""
    c = _factor_circuit(56)
    p_share, v_share = trusted_dealer_setup(c.vole_count())
    proof = prove_ni(c, [7, 8], p_share)
    bad = NIProof(
        msg1=type(proof.msg1)(
            d_values=[F.add(proof.msg1.d_values[0], 1)] + proof.msg1.d_values[1:],
            assert_openings=proof.msg1.assert_openings,
        ),
        msg2=proof.msg2,
    )
    assert not verify_ni(c, v_share, bad)


def test_ni_label_must_match():
    c = _factor_circuit(56)
    p_share, v_share = trusted_dealer_setup(c.vole_count())
    proof = prove_ni(c, [7, 8], p_share, label=b"context-A")
    assert verify_ni(c, v_share, proof, label=b"context-A")
    assert not verify_ni(c, v_share, proof, label=b"context-B")


def test_transcript_squeezes_are_independent():
    t = Transcript()
    t.absorb_int(b"x", 42)
    a = t.challenge()
    b = t.challenge()
    assert a != b
    assert 0 < a < F.p and 0 < b < F.p


def test_ni_works_for_polynomial_circuit():
    # Same x^3 + x + 5 = y demo as the interactive version.
    x = 17
    y = (x ** 3 + x + 5) % F.p
    c = Circuit()
    xw = c.input()
    x2 = c.mul(xw, xw)
    x3 = c.mul(x2, xw)
    s = c.add(x3, xw)
    s = c.add_const(s, 5)
    c.assert_eq(s, y)
    ok, _ = run_ni(c, [x])
    assert ok


def test_ni_rejects_wrong_witness_at_assertion():
    # Same shape circuit, different witness.
    c = _factor_circuit(56)
    p_share, _ = trusted_dealer_setup(c.vole_count())
    with pytest.raises(ValueError):
        prove_ni(c, [2, 3], p_share)  # 2*3 != 56; assert_zero raises


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
