"""LPN-based VOLE extension (pseudorandom correlation generator).

The trusted dealer in ``vole.py`` hands out N VOLE correlations
explicitly. Real VOLE-based ZK protocols replace this with a
*pseudorandom correlation generator* (PCG): a small sparse base
correlation gets locally expanded to N pseudorandom correlations
using public structure, with no further communication. This module
implements the LPN / regular-syndrome-decoding flavour due to
Boyle, Couteau, Gilboa, Ishai (CCS 2019) -- the same primitive used
by Wolverine, Mac'n'Cheese, and (in subspace form) by QuickSilver.

Construction
------------

Let `k` be a base length, `t` the noise weight, and `N` the output
length.  The PCG:

1. Produces a length-k VOLE correlation whose prover-side ``u_base``
   has Hamming weight exactly ``t`` (rest zero). Soundness reduction
   to "regular LPN / regular syndrome decoding" is well-studied.
2. Both parties hold a public N x k matrix ``H`` derived from a seed.
3. Output VOLE = H * base (matrix-vector product on each share).
   Linearity preserves the IT-MAC relation; ``u_out = H * u_base`` is
   pseudorandom under regular-LPN with the right (k, t) parameters.

What this module *does*
-----------------------

- Generates a sparse base VOLE via the trusted dealer (still
  centralised; the cryptographic step we are replacing is the *expansion*
  from base to output, not the base itself).
- Implements the H-multiply expansion deterministically from a seed.
- Returns standard ``VoleProverShare`` / ``VoleVerifierShare`` objects so
  the result is a drop-in replacement for ``trusted_dealer_setup``.

What this module *does not* do
------------------------------

- Implement base OT or SPCOT (the GGM-tree single-point VOLE that
  removes the trusted base). That is a separate primitive of comparable
  scope; the extension step here is the more cryptographically
  characteristic part of the PCG.
- Pick LPN parameters at any concrete security level. The defaults
  ``(k=2N, t=N//4)`` are *not* secure -- they are merely large enough
  to demonstrate the technique. Real parameters from BCGI19 are
  k around 2^14, t around 760 for 80-bit primary security.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Sequence

from quicksilver.field import F, Fp
from quicksilver.vole import VoleProverShare, VoleVerifierShare


@dataclass(frozen=True)
class LpnParams:
    n_out: int   # number of output VOLE elements
    k_base: int  # base correlation length
    t_weight: int  # Hamming weight of the base error
    seed: bytes  # public 32-byte seed for the H matrix

    @classmethod
    def default(cls, n_out: int, seed: bytes | None = None) -> "LpnParams":
        if seed is None:
            seed = secrets.token_bytes(32)
        return cls(n_out=n_out, k_base=2 * n_out, t_weight=max(4, n_out // 4),
                   seed=seed)


def derive_matrix(seed: bytes, n_rows: int, n_cols: int, field: Fp) -> list:
    """Pseudorandom field-valued ``n_rows x n_cols`` matrix from ``seed``.

    Bulk-streams bytes from SHA-256 in counter mode and reduces each
    chunk modulo p. This introduces a sub-2^-128 bias (negligible vs p),
    far below any cryptographic threshold and far cheaper than per-entry
    rejection sampling.
    """
    p = field.p
    bytes_per = (p.bit_length() + 128 + 7) // 8  # 16-byte slack drowns the bias
    needed = n_rows * n_cols * bytes_per

    out = bytearray()
    counter = 0
    while len(out) < needed:
        h = hashlib.sha256()
        h.update(b"lpn-vole/H/v1")
        h.update(seed)
        h.update(counter.to_bytes(8, "big"))
        out += h.digest()
        counter += 1

    H = []
    pos = 0
    for _ in range(n_rows):
        row = [0] * n_cols
        for j in range(n_cols):
            x = int.from_bytes(out[pos:pos + bytes_per], "big") % p
            row[j] = x
            pos += bytes_per
        H.append(row)
    return H


def _matvec(M, x, field: Fp) -> list:
    out = [0] * len(M)
    for i, row in enumerate(M):
        acc = 0
        for j, h_ij in enumerate(row):
            if h_ij == 0:
                continue
            acc = field.add(acc, field.mul(h_ij, x[j]))
        out[i] = acc
    return out


def _sparse_base_vole(
    k: int, t: int, field: Fp, delta: int
) -> tuple[list[int], list[int], list[int]]:
    """Trusted-dealer base correlation: ``u`` has Hamming weight exactly ``t``."""
    if t > k:
        raise ValueError("hamming weight cannot exceed base length")
    rng = secrets.SystemRandom()
    positions = rng.sample(range(k), t)
    u = [0] * k
    for pos in positions:
        u[pos] = field.rand_nonzero()
    v = [field.rand() for _ in range(k)]
    w = [field.add(field.mul(u_i, delta), v_i) for u_i, v_i in zip(u, v)]
    return u, v, w


def lpn_vole_extend(
    params: LpnParams,
    field: Fp = F,
    delta: int | None = None,
) -> tuple[VoleProverShare, VoleVerifierShare]:
    """Generate ``params.n_out`` VOLE correlations via LPN extension.

    Drop-in replacement for ``trusted_dealer_setup``: the returned
    ``VoleProverShare`` and ``VoleVerifierShare`` plug straight into
    ``protocol.prove`` and ``protocol.verify``.
    """
    if delta is None:
        delta = field.rand()
    H = derive_matrix(params.seed, params.n_out, params.k_base, field)
    u_base, v_base, w_base = _sparse_base_vole(
        params.k_base, params.t_weight, field, delta
    )
    u_out = _matvec(H, u_base, field)
    v_out = _matvec(H, v_base, field)
    w_out = _matvec(H, w_base, field)
    return (
        VoleProverShare(u=u_out, v=v_out),
        VoleVerifierShare(delta=delta, w=w_out),
    )


def correlation_holds(
    p_share: VoleProverShare, v_share: VoleVerifierShare, field: Fp = F
) -> bool:
    if len(p_share) != len(v_share):
        return False
    for u, v, w in zip(p_share.u, p_share.v, v_share.w):
        if w != field.add(field.mul(u, v_share.delta), v):
            return False
    return True
