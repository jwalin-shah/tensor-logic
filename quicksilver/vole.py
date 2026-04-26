"""VOLE setup, simulated by a trusted dealer.

A single VOLE element gives:

    Prover:   (u, v) in F_p x F_p   with u uniform
    Verifier: w = u * Delta + v     in F_p

The verifier's global secret Delta is fixed across all VOLE elements
(a "subspace VOLE"). This is exactly what is needed to derive an
information-theoretic MAC: see ``itmac.py``.

In a real deployment the (u, v, w) triples would be produced by an
LPN-based VOLE extension protocol (e.g. SoftSpoken, Wolverine's Ferret).
That phase is the heavy hitter cryptographically but is orthogonal to
QuickSilver itself. Here we model it as preprocessing handed out by an
honest dealer; the rest of the protocol is faithful to the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from quicksilver.field import F, Fp


@dataclass
class VoleProverShare:
    """Prover's view of the preprocessed VOLE."""

    u: List[int]
    v: List[int]

    def __len__(self) -> int:
        return len(self.u)


@dataclass
class VoleVerifierShare:
    """Verifier's view: a single global key Delta plus per-element w_i."""

    delta: int
    w: List[int]

    def __len__(self) -> int:
        return len(self.w)


def trusted_dealer_setup(
    n: int, field: Fp = F, delta: int | None = None
) -> Tuple[VoleProverShare, VoleVerifierShare]:
    """Generate ``n`` VOLE correlations honestly.

    If ``delta`` is None a fresh random global key is sampled.
    """
    if delta is None:
        delta = field.rand()
    u = [field.rand() for _ in range(n)]
    v = [field.rand() for _ in range(n)]
    w = [field.add(field.mul(u_i, delta), v_i) for u_i, v_i in zip(u, v)]
    return VoleProverShare(u=u, v=v), VoleVerifierShare(delta=delta, w=w)
