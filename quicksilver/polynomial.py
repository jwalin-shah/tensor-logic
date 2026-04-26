"""Degree-d polynomial constraint check (Section 5 of the paper).

The basic QuickSilver protocol in ``protocol.py`` only proves degree-2
relations (multiplication gates). The same VOLE-IT-MAC machinery
extends, with no extra committed wires, to *arbitrary* degree-d
polynomial relations between committed values. Communication cost is
``d`` field elements per batch (regardless of how many constraints
of that degree are batched), versus ``d-1`` extra multiplication
gates per constraint if you tried to express it via the mul protocol.

Identity exploited
------------------

For each committed wire i, ``K_i = M_i + Delta * x_i``. To make the
"leading Delta coefficient = constraint value" identity hold for
polynomials with mixed-degree monomials, we work with the
Delta-homogenisation of P:

    P~(K; Delta) := sum_t coeff_t * Delta^{d - deg(t)} * prod_{i in idx_t} K_i

(scaling each degree-e monomial by Delta^{d-e}). Then

    P~(M + Delta x; Delta) = sum_{k=0}^{d} A_k * Delta^k

with ``A_d == P(x_1, ..., x_n)``. Honest prover: ``A_d = 0``. Prover
sends the lower coefficients ``(A_0, ..., A_{d-1})``; verifier
recomputes ``P~(K_1, ..., K_n; Delta)`` and checks the identity.

Batching m such constraints with a verifier challenge chi is
straightforward: ``A_e := sum_j chi^j * A_{e,j}``.

ZK masking uses ``d-1`` fresh VOLE elements; the d=2 case recovers the
single-mask construction used for multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

from quicksilver.field import F, Fp
from quicksilver.itmac import ProverWire, VerifierWire
from quicksilver.vole import (
    VoleProverShare,
    VoleVerifierShare,
    trusted_dealer_setup,
)


# A monomial is a coefficient and a tuple of wire-ids (with repetition).
# Example: 3 * x_1 * x_2^2  ==  (3, (1, 2, 2)).
Monomial = Tuple[int, Tuple[int, ...]]


@dataclass(frozen=True)
class Polynomial:
    """Sparse multivariate polynomial over wire-id variables."""

    terms: Tuple[Monomial, ...]

    @property
    def degree(self) -> int:
        return max((len(idx) for _, idx in self.terms), default=0)

    def evaluate(self, values: dict, field: Fp = F) -> int:
        out = 0
        for coeff, idx in self.terms:
            term = field.encode(coeff)
            for wid in idx:
                term = field.mul(term, values[wid])
            out = field.add(out, term)
        return out


def _delta_coefficients(
    poly: Polynomial,
    prover_wires: dict,
    field: Fp,
) -> List[int]:
    """Coefficients A_0..A_d of P~(M + Delta x; Delta) as a poly in Delta.

    A degree-e monomial coeff * prod_{i in idx} (M_i + Delta x_i),
    pre-scaled by Delta^{d-e}, contributes to A_k for k = (d-e) + |S|
    where S ranges over subsets of idx (the indices substituted with x).
    """
    d = poly.degree
    coeffs = [0] * (d + 1)
    for coeff, idx in poly.terms:
        c = field.encode(coeff)
        e = len(idx)
        positions = list(range(e))
        for s in range(e + 1):
            k = d - e + s
            for sset in combinations(positions, s):
                term = c
                sset_set = set(sset)
                for i, wid in enumerate(idx):
                    w = prover_wires[wid]
                    factor = w.x if i in sset_set else w.m
                    term = field.mul(term, factor)
                coeffs[k] = field.add(coeffs[k], term)
    return coeffs


def _evaluate_on_keys(
    poly: Polynomial, verifier_wires: dict, delta: int, field: Fp
) -> int:
    """Evaluate P~(K_1, ..., K_n; Delta) (the Delta-homogenisation)."""
    d = poly.degree
    out = 0
    for coeff, idx in poly.terms:
        e = len(idx)
        term = field.mul(field.encode(coeff), field.pow(delta, d - e))
        for wid in idx:
            term = field.mul(term, verifier_wires[wid].k)
        out = field.add(out, term)
    return out


@dataclass
class PolyProof:
    masked_A: List[int]  # length d


def prove_polys(
    polys: Sequence[Polynomial],
    prover_wires: dict,
    chi: int,
    mask: VoleProverShare,
    field: Fp = F,
) -> PolyProof:
    """Prover side of a batched degree-d polynomial-zero check.

    ``mask`` must hold exactly ``d-1`` VOLE elements.  ``chi`` is the
    verifier's challenge.  Honest prover: every polynomial in ``polys``
    evaluates to zero on the committed wire values.
    """
    if not polys:
        raise ValueError("need at least one polynomial")
    d = max(p.degree for p in polys)
    if any(p.degree != d for p in polys):
        raise ValueError("batched polys must share the same degree")
    if d < 1:
        raise ValueError("degree must be at least 1")
    if len(mask) != d - 1:
        raise ValueError(f"need exactly {d - 1} masking VOLE elements, got {len(mask)}")

    # Aggregate A_e = sum_j chi^j * A_{e,j}, e = 0..d.  Honest -> A_d = 0.
    A = [0] * (d + 1)
    chi_j = 1
    for poly in polys:
        chi_j = field.mul(chi_j, chi)  # chi^1, chi^2, ...
        coeffs = _delta_coefficients(poly, prover_wires, field)
        for e, c in enumerate(coeffs):
            A[e] = field.add(A[e], field.mul(chi_j, c))

    # Mask using the construction described in the module docstring:
    # A_0' = A_0 + b^{(0)}
    # A_e' = A_e + a^{(e-1)} + b^{(e)}     for 1 <= e <= d-2
    # A_{d-1}' = A_{d-1} + a^{(d-2)}
    masked = list(A[:d])  # drop A_d (must be zero)
    for j in range(d - 1):
        a_j, b_j = mask.u[j], mask.v[j]
        masked[j] = field.add(masked[j], b_j)
        masked[j + 1] = field.add(masked[j + 1], a_j)
    return PolyProof(masked_A=masked)


def verify_polys(
    polys: Sequence[Polynomial],
    verifier_wires: dict,
    chi: int,
    proof: PolyProof,
    mask: VoleVerifierShare,
    field: Fp = F,
) -> bool:
    if not polys:
        return False
    d = max(p.degree for p in polys)
    if any(p.degree != d for p in polys):
        return False
    if len(proof.masked_A) != d:
        return False
    if len(mask) != d - 1:
        return False
    delta = mask.delta

    # LHS: sum_e A_e' * Delta^e
    lhs = 0
    delta_pow = 1
    for e in range(d):
        lhs = field.add(lhs, field.mul(proof.masked_A[e], delta_pow))
        delta_pow = field.mul(delta_pow, delta)

    # RHS: sum_j chi^j * P_j(K_1, ..., K_n) + sum_j c^{(j)} * Delta^j
    rhs = 0
    chi_j = 1
    for poly in polys:
        chi_j = field.mul(chi_j, chi)
        b_j = _evaluate_on_keys(poly, verifier_wires, delta, field)
        rhs = field.add(rhs, field.mul(chi_j, b_j))
    delta_pow = 1
    for j in range(d - 1):
        rhs = field.add(rhs, field.mul(mask.w[j], delta_pow))
        delta_pow = field.mul(delta_pow, delta)
    return lhs == rhs


# ---- One-shot helper for tests / demos -------------------------------------


def run_poly_check(
    polys: Sequence[Polynomial],
    prover_wires: dict,
    verifier_wires: dict,
    delta: int,
    field: Fp = F,
) -> bool:
    """End-to-end batched polynomial-zero check given existing IT-MACs.

    Generates the ``d-1`` masking VOLE elements with ``delta`` fixed
    (so they are consistent with the wires already committed under
    that Delta).
    """
    d = max(p.degree for p in polys)
    p_share, v_share = trusted_dealer_setup(d - 1, field, delta=delta)
    chi = field.rand_nonzero()
    proof = prove_polys(polys, prover_wires, chi, p_share, field)
    return verify_polys(polys, verifier_wires, chi, proof, v_share, field)
