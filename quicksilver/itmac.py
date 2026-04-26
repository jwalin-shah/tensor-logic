"""Information-theoretic MACs derived from VOLE.

For every wire value ``x in F_p`` the prover holds a tag ``M`` and the
verifier holds a key ``K`` related by

    K = M + Delta * x.

Linear operations on (x, M) and on K are coordinated so the relation is
preserved without communication. Only multiplications need the
QuickSilver consistency check.

Two views of a wire are exposed:

    ProverWire(x, m)   -- prover side
    VerifierWire(k)    -- verifier side

Constants are lifted into wires by setting ``M = 0`` on the prover and
``K = Delta * c`` on the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from quicksilver.field import F, Fp


@dataclass(frozen=True)
class ProverWire:
    """A wire as seen by the prover: cleartext value and IT-MAC tag."""

    x: int
    m: int

    def add(self, other: "ProverWire", field: Fp = F) -> "ProverWire":
        return ProverWire(field.add(self.x, other.x), field.add(self.m, other.m))

    def sub(self, other: "ProverWire", field: Fp = F) -> "ProverWire":
        return ProverWire(field.sub(self.x, other.x), field.sub(self.m, other.m))

    def add_const(self, c: int, field: Fp = F) -> "ProverWire":
        # Adding a public constant does not change the tag.
        return ProverWire(field.add(self.x, c), self.m)

    def mul_const(self, c: int, field: Fp = F) -> "ProverWire":
        return ProverWire(field.mul(self.x, c), field.mul(self.m, c))


@dataclass(frozen=True)
class VerifierWire:
    """A wire as seen by the verifier: only the key is known."""

    k: int

    def add(self, other: "VerifierWire", field: Fp = F) -> "VerifierWire":
        return VerifierWire(field.add(self.k, other.k))

    def sub(self, other: "VerifierWire", field: Fp = F) -> "VerifierWire":
        return VerifierWire(field.sub(self.k, other.k))

    def add_const(self, c: int, delta: int, field: Fp = F) -> "VerifierWire":
        # The key for a public constant ``c`` is Delta * c.
        return VerifierWire(field.add(self.k, field.mul(delta, c)))

    def mul_const(self, c: int, field: Fp = F) -> "VerifierWire":
        return VerifierWire(field.mul(self.k, c))


def prover_commit(value: int, u: int, v: int, field: Fp = F) -> tuple[ProverWire, int]:
    """Convert one VOLE element into an IT-MAC commitment.

    Returns the prover's wire and the message ``d = x - u`` to send to
    the verifier.
    """
    d = field.sub(value, u)
    # M = v, since K = w + Delta * d = Delta * u + v + Delta * (x - u) = v + Delta * x.
    return ProverWire(x=value, m=v), d


def verifier_receive(d: int, w: int, delta: int, field: Fp = F) -> VerifierWire:
    """Verifier's side of ``prover_commit``."""
    return VerifierWire(k=field.add(w, field.mul(delta, d)))


def prover_const(c: int) -> ProverWire:
    return ProverWire(x=c, m=0)


def verifier_const(c: int, delta: int, field: Fp = F) -> VerifierWire:
    return VerifierWire(k=field.mul(delta, c))


def open_to_zero(w: ProverWire) -> int:
    """Reveal a wire that the prover claims is zero, by sending its tag.

    The verifier checks ``K == M`` (since ``K = M + Delta*0 = M``).
    """
    if w.x != 0:
        raise ValueError("prover wire claimed to be zero is nonzero")
    return w.m


# Sugar: simple alias used in the protocol module.
Wire = ProverWire
