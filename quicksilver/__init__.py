"""QuickSilver: VOLE-based zero-knowledge proofs for arithmetic circuits.

Implements the protocol of Yang, Sarkar, Weng, Wang (CCS 2021,
https://eprint.iacr.org/2021/076) for the prime-field setting.

Educational implementation: VOLE setup is performed by a trusted dealer
rather than via LPN/OT-based extension. Everything else (IT-MACs, the
batched multiplication check, the degree-d polynomial extension) follows
the paper.
"""

from quicksilver.field import F, Fp
from quicksilver.itmac import Wire
from quicksilver.circuit import Circuit
from quicksilver.protocol import prove, verify, run

__all__ = ["F", "Fp", "Wire", "Circuit", "prove", "verify", "run"]
