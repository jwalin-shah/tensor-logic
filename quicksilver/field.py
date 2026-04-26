"""Prime-field arithmetic over the Mersenne prime p = 2**127 - 1.

A 127-bit prime gives soundness error ~2^-126 per check, comfortably
beyond the cryptographic threshold while keeping pure-Python arithmetic
fast (no big-int reductions worth optimising further at this scale).

The field is exposed as a singleton ``F`` whose elements are plain ints
in [0, p). Operations take and return ints; this stays cheap and avoids
the per-op overhead of a wrapper class.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class Fp:
    """Prime field F_p with p = 2**bits - 1 (Mersenne) or generic prime."""

    p: int

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def neg(self, a: int) -> int:
        return (-a) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def inv(self, a: int) -> int:
        if a % self.p == 0:
            raise ZeroDivisionError("inverse of zero in F_p")
        return pow(a, self.p - 2, self.p)

    def div(self, a: int, b: int) -> int:
        return self.mul(a, self.inv(b))

    def pow(self, a: int, e: int) -> int:
        return pow(a, e, self.p)

    def rand(self) -> int:
        """Uniform sample from F_p via rejection."""
        # 2**127 - 1 is so close to a power of two that rejection almost
        # never fires; for general primes this is still correct.
        bits = self.p.bit_length()
        while True:
            r = secrets.randbits(bits)
            if r < self.p:
                return r

    def rand_nonzero(self) -> int:
        while True:
            r = self.rand()
            if r != 0:
                return r

    def encode(self, x: int) -> int:
        """Reduce an arbitrary int into [0, p)."""
        return x % self.p


# Mersenne prime 2^127 - 1. Soundness error per check ~ 1 / 2^127.
F = Fp(p=(1 << 127) - 1)
