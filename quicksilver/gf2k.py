"""GF(2^128) arithmetic with the GCM irreducible polynomial.

Elements are integers in [0, 2^128). Addition is XOR (characteristic 2).
Multiplication is carry-less polynomial multiplication followed by
reduction mod ``x^128 + x^7 + x^2 + x + 1`` -- the same polynomial used
by AES-GCM, so test vectors are widely available.

Pure Python, ~5 us per multiply on modest hardware. Fast enough for the
educational MAC-field role and slow enough that you would never use this
in production. Replace with ``cryptography.hazmat.primitives.poly1305``
or a CLMUL intrinsic if you actually care.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

_MASK = (1 << 128) - 1
_REDUCTION = 0x87  # low bits of x^128 + x^7 + x^2 + x + 1
_HIGH_BIT = 1 << 127


def _mul(a: int, b: int) -> int:
    a &= _MASK
    b &= _MASK
    result = 0
    for _ in range(128):
        if b & 1:
            result ^= a
        b >>= 1
        if a & _HIGH_BIT:
            a = ((a << 1) & _MASK) ^ _REDUCTION
        else:
            a <<= 1
    return result


@dataclass(frozen=True)
class GF2k:
    """Field abstraction with the same interface as ``Fp``.

    The ``p`` attribute is set to ``2**128`` for convenience -- it is
    *not* a prime, and the field has 2^128 elements. Anything that
    needs a "field size" should use ``order`` instead.
    """

    bits: int = 128
    p: int = 1 << 128  # purely for interface compatibility with Fp

    @property
    def order(self) -> int:
        return 1 << self.bits

    def encode(self, x: int) -> int:
        return x & _MASK

    def add(self, a: int, b: int) -> int:
        return (a ^ b) & _MASK

    def sub(self, a: int, b: int) -> int:
        return (a ^ b) & _MASK  # characteristic 2

    def neg(self, a: int) -> int:
        return a & _MASK

    def mul(self, a: int, b: int) -> int:
        return _mul(a, b)

    def pow(self, a: int, e: int) -> int:
        """Square-and-multiply."""
        result = 1
        base = a & _MASK
        while e > 0:
            if e & 1:
                result = _mul(result, base)
            base = _mul(base, base)
            e >>= 1
        return result

    def inv(self, a: int) -> int:
        """Inversion via Fermat's little theorem: a^(2^128 - 2)."""
        if a == 0:
            raise ZeroDivisionError("inverse of zero in GF(2^128)")
        return self.pow(a, (1 << 128) - 2)

    def div(self, a: int, b: int) -> int:
        return self.mul(a, self.inv(b))

    def rand(self) -> int:
        return secrets.randbits(128)

    def rand_nonzero(self) -> int:
        while True:
            r = self.rand()
            if r != 0:
                return r


GF128 = GF2k()
