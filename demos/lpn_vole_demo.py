"""LPN-based VOLE extension demo.

Replaces the trusted dealer in the main protocol with a pseudorandom
correlation generator: a sparse base correlation gets locally
expanded to a long pseudorandom one via a public LPN matrix.

The output VOLE is a drop-in for ``trusted_dealer_setup`` -- the
ZK proof code does not change.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from quicksilver.circuit import Circuit
from quicksilver.field import F
from quicksilver.lpn_vole import LpnParams, lpn_vole_extend
from quicksilver.protocol import prove, verify


def demo_drop_in_replacement() -> None:
    print("=" * 64)
    print("Demo: LPN-extended VOLE replaces the trusted dealer")
    print("=" * 64)
    # Same factorisation circuit as the original demo.
    p, q = 1_000_003, 999_983
    N = p * q
    c = Circuit()
    pw = c.input()
    qw = c.input()
    c.assert_eq(c.mul(pw, qw), N)
    print(f"  circuit needs {c.vole_count()} VOLE elements")

    params = LpnParams.default(n_out=c.vole_count())
    print(f"  LPN params: k_base={params.k_base} (length of base correlation),")
    print(f"              t={params.t_weight} (Hamming weight of base error)")
    t0 = time.time()
    p_share, v_share = lpn_vole_extend(params)
    setup_dt = (time.time() - t0) * 1000
    print(f"  LPN extension: {setup_dt:.1f} ms")

    msg1, batched = prove(c, [p, q], p_share)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    ok = verify(c, v_share, msg1, chi, msg2)
    print(f"  verifier accepts: {ok}\n")
    assert ok


def demo_pseudorandomness() -> None:
    print("=" * 64)
    print("Demo: u_base is sparse; u_out is indistinguishable from random")
    print("=" * 64)
    n = 64
    params = LpnParams.default(n_out=n)
    p_share, _ = lpn_vole_extend(params)
    # Base would have been H · sparse, but we never expose it; we can show
    # the OUTPUT u looks unstructured.
    zeros = sum(1 for x in p_share.u if x == 0)
    small = sum(1 for x in p_share.u if x < (1 << 64))  # very biased towards small
    print(f"  output u length:        {len(p_share.u)}")
    print(f"  zeros in u:             {zeros}  (expect ~ 0)")
    print(f"  values < 2^64 in u:     {small} of {n}  (expect ~ 0 by chance)")
    print(f"  first u entries:        {[hex(x)[:18] for x in p_share.u[:3]]}")
    print()


def demo_scaling() -> None:
    print("=" * 64)
    print("Demo: setup time vs output VOLE length")
    print("=" * 64)
    print(f"  {'n_out':>8} {'k_base':>8} {'setup(ms)':>12}")
    for n in (32, 128, 512, 2048):
        params = LpnParams.default(n_out=n)
        t0 = time.time()
        lpn_vole_extend(params)
        dt = (time.time() - t0) * 1000
        print(f"  {n:>8} {params.k_base:>8} {dt:>12.1f}")
    print()


if __name__ == "__main__":
    demo_drop_in_replacement()
    demo_pseudorandomness()
    demo_scaling()
    print("All LPN-VOLE demos passed.")
