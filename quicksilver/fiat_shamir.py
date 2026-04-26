"""Fiat-Shamir transform for the QuickSilver protocol.

The interactive protocol has one verifier message: the random
challenge ``chi`` for the batched multiplication-consistency check.
Replacing it with a hash of the transcript so far makes the proof
non-interactive: the prover produces a single object ``NIProof =
(msg1, msg2)`` and any party holding the verifier key can check it.

Note this remains *designated-verifier*. The verifier's secret
``Delta`` is required to validate the IT-MAC openings; without it a
third party cannot verify. Fiat-Shamir gives non-interactivity, not
public verifiability.

Random oracle: SHA-256 in counter mode. Honest-verifier ZK in the
interactive protocol implies ZK in the random oracle model after the
transform (standard result, e.g. Pointcheval-Stern).

Whether to absorb the circuit
-----------------------------

The circuit must be bound into the transcript so a malicious prover
cannot pick gates after seeing ``chi``. We hash a canonical
serialisation of (op, args) for every gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List

from quicksilver.circuit import Circuit, Op
from quicksilver.field import F, Fp
from quicksilver.protocol import (
    BatchedCheck,
    CommitMessage,
    prove,
    verify,
)
from quicksilver.vole import VoleProverShare, VoleVerifierShare, trusted_dealer_setup


_DOMAIN = b"quicksilver/v1/fiat-shamir"


class Transcript:
    """Sponge-like transcript over SHA-256.

    ``absorb_*`` mixes new bytes into the state; ``challenge`` squeezes
    a field element. Each squeeze advances the state so subsequent
    squeezes are independent.
    """

    __slots__ = ("_buf",)

    def __init__(self, label: bytes = _DOMAIN):
        self._buf = bytearray()
        self._buf += b"label:"
        self._buf += len(label).to_bytes(4, "big")
        self._buf += label

    def absorb_bytes(self, tag: bytes, data: bytes) -> None:
        self._buf += b"|"
        self._buf += tag
        self._buf += b":"
        self._buf += len(data).to_bytes(4, "big")
        self._buf += data

    def absorb_int(self, tag: bytes, x: int, nbytes: int = 32) -> None:
        self.absorb_bytes(tag, x.to_bytes(nbytes, "big"))

    def absorb_ints(self, tag: bytes, xs: Iterable[int], nbytes: int = 32) -> None:
        body = bytearray()
        for x in xs:
            body += x.to_bytes(nbytes, "big")
        self.absorb_bytes(tag, bytes(body))

    def absorb_circuit(self, circuit: Circuit) -> None:
        body = bytearray()
        # Top-line shape so num-* changes invalidate the transcript even
        # if gates compare equal (defence in depth).
        for n in (circuit.num_wires, circuit.num_inputs,
                  circuit.num_muls, circuit.num_asserts):
            body += n.to_bytes(8, "big")
        for g in circuit.gates:
            body += g.op.value.encode()
            body += b"("
            for a in g.args:
                body += int(a).to_bytes(32, "big", signed=True)
                body += b","
            body += b")|"
            if g.out is not None:
                body += g.out.to_bytes(8, "big")
        self.absorb_bytes(b"circuit", bytes(body))

    def challenge(self, field: Fp = F) -> int:
        # Hash-to-field via rejection on a 32-byte squeeze. p ~= 2^127 so
        # rejection happens with probability ~ 1/2 per draw; cheap enough.
        counter = 0
        while True:
            h = hashlib.sha256()
            h.update(b"squeeze")
            h.update(counter.to_bytes(8, "big"))
            h.update(bytes(self._buf))
            digest = h.digest()
            x = int.from_bytes(digest, "big") % (1 << field.p.bit_length())
            if x < field.p and x != 0:
                # Mix the digest back so subsequent squeezes are independent.
                self._buf += b"|squeezed:"
                self._buf += digest
                return x
            counter += 1


@dataclass
class NIProof:
    """A non-interactive QuickSilver proof for a designated verifier."""

    msg1: CommitMessage
    msg2: BatchedCheck


def _absorb_msg1(t: Transcript, msg: CommitMessage) -> None:
    t.absorb_ints(b"d_values", msg.d_values)
    t.absorb_ints(b"openings", msg.assert_openings)


def prove_ni(
    circuit: Circuit,
    witness: List[int],
    share: VoleProverShare,
    field: Fp = F,
    label: bytes = _DOMAIN,
) -> NIProof:
    """Produce a non-interactive QuickSilver proof."""
    msg1, batched = prove(circuit, witness, share, field)
    t = Transcript(label)
    t.absorb_circuit(circuit)
    _absorb_msg1(t, msg1)
    chi = t.challenge(field)
    return NIProof(msg1=msg1, msg2=batched(chi))


def verify_ni(
    circuit: Circuit,
    share: VoleVerifierShare,
    proof: NIProof,
    field: Fp = F,
    label: bytes = _DOMAIN,
) -> bool:
    t = Transcript(label)
    t.absorb_circuit(circuit)
    _absorb_msg1(t, proof.msg1)
    chi = t.challenge(field)
    return verify(circuit, share, proof.msg1, chi, proof.msg2, field)


def run_ni(
    circuit: Circuit,
    witness: List[int],
    field: Fp = F,
    label: bytes = _DOMAIN,
):
    """End-to-end: setup + non-interactive prove + verify."""
    p_share, v_share = trusted_dealer_setup(circuit.vole_count(), field)
    proof = prove_ni(circuit, witness, p_share, field, label)
    return verify_ni(circuit, v_share, proof, field, label), proof
