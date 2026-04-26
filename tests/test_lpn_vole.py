"""Tests for the LPN-based VOLE extension."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quicksilver.circuit import Circuit
from quicksilver.field import F
from quicksilver.lpn_vole import (
    LpnParams,
    correlation_holds,
    derive_matrix,
    lpn_vole_extend,
)
from quicksilver.protocol import prove, verify


# ---- Correlation correctness --------------------------------------------


def test_lpn_vole_correlation_holds():
    p, v = lpn_vole_extend(LpnParams.default(n_out=64, seed=b"x" * 32))
    assert correlation_holds(p, v)


def test_lpn_vole_returns_requested_length():
    n = 128
    p, v = lpn_vole_extend(LpnParams.default(n_out=n, seed=b"y" * 32))
    assert len(p) == n
    assert len(v) == n


def test_lpn_vole_uses_supplied_delta():
    delta = F.rand_nonzero()
    p, v = lpn_vole_extend(LpnParams.default(n_out=32), delta=delta)
    assert v.delta == delta
    assert correlation_holds(p, v)


# ---- Determinism in the matrix derivation ------------------------------


def test_derive_matrix_deterministic():
    seed = b"\x42" * 32
    H1 = derive_matrix(seed, 4, 8, F)
    H2 = derive_matrix(seed, 4, 8, F)
    assert H1 == H2


def test_derive_matrix_seed_varies_output():
    H1 = derive_matrix(b"\x01" * 32, 4, 8, F)
    H2 = derive_matrix(b"\x02" * 32, 4, 8, F)
    assert H1 != H2


# ---- u_out looks unstructured even though u_base is sparse -------------


def test_extension_hides_sparsity_of_base():
    """u_base has Hamming weight t out of k=2*n; u_out should have ~no zeros."""
    n = 64
    params = LpnParams.default(n_out=n)
    p, _ = lpn_vole_extend(params)
    zeros = sum(1 for x in p.u if x == 0)
    # Probability of any single u_out[i] being zero by chance is ~1/p, vanishing.
    assert zeros < 2


# ---- Drop-in replacement for trusted dealer in the main protocol -------


def test_lpn_vole_plugs_into_protocol():
    c = Circuit()
    a = c.input()
    b = c.input()
    c.assert_eq(c.mul(a, b), 56)
    p, v = lpn_vole_extend(LpnParams.default(n_out=c.vole_count()))
    msg1, batched = prove(c, [7, 8], p)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    assert verify(c, v, msg1, chi, msg2)


def test_lpn_vole_plugs_into_polynomial_circuit():
    """x^3 + x + 5 = y for x=17 via LPN-extended VOLE."""
    x = 17
    y = (x ** 3 + x + 5) % F.p
    c = Circuit()
    xw = c.input()
    x2 = c.mul(xw, xw)
    x3 = c.mul(x2, xw)
    s = c.add(x3, xw)
    s = c.add_const(s, 5)
    c.assert_eq(s, y)
    p, v = lpn_vole_extend(LpnParams.default(n_out=c.vole_count()))
    msg1, batched = prove(c, [x], p)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    assert verify(c, v, msg1, chi, msg2)


def test_soundness_preserved_with_lpn_vole():
    """Tampering with the batched check is still rejected when using LPN."""
    from quicksilver.protocol import BatchedCheck

    c = Circuit()
    a = c.input()
    b = c.input()
    c.assert_eq(c.mul(a, b), 56)
    p, v = lpn_vole_extend(LpnParams.default(n_out=c.vole_count()))
    msg1, batched = prove(c, [7, 8], p)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    bad = BatchedCheck(U=F.add(msg2.U, 1), V=msg2.V)
    assert not verify(c, v, msg1, chi, bad)


# ---- Parameters guard rails ---------------------------------------------


def test_lpn_params_default_picks_consistent_seed_length():
    params = LpnParams.default(n_out=32)
    assert isinstance(params.seed, bytes)
    assert len(params.seed) == 32


def test_lpn_extend_rejects_oversized_weight():
    bad = LpnParams(n_out=8, k_base=4, t_weight=10, seed=b"z" * 32)
    with pytest.raises(ValueError):
        lpn_vole_extend(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
