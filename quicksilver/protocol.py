"""QuickSilver interactive protocol.

Three rounds (after VOLE preprocessing):

    P -> V : per-input and per-mul commitments d_i = x_i - u_i,
             plus opened tag M for each assert-zero wire.
    V -> P : random challenge chi in F_p.
    P -> V : batched multiplication-consistency proof (U, V),
             masked by one fresh VOLE element so it leaks nothing.

The verifier accepts iff every assert-zero check passes and the
batched identity ``W + c == U + V * Delta`` holds, where W and c are
verifier-side aggregates (see ``_VerifierWalker.batched_check``).

Soundness error per execution is bounded by ``(m + 2) / |F|`` for ``m``
multiplication gates (Lemma 3 of the paper, specialised to a large
prime field). With p = 2^127 - 1 this is well below 2^-120 for any
circuit you would ever run in pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterator, List, Tuple

from quicksilver.circuit import Circuit, Gate, Op
from quicksilver.field import F, Fp
from quicksilver.itmac import (
    ProverWire,
    VerifierWire,
    open_to_zero,
    prover_commit,
    prover_const,
    verifier_const,
    verifier_receive,
)
from quicksilver.vole import (
    VoleProverShare,
    VoleVerifierShare,
    trusted_dealer_setup,
)


# ---- Messages exchanged between prover and verifier ------------------------


@dataclass
class CommitMessage:
    """Round 1: prover -> verifier."""

    d_values: List[int]      # one per input or mul gate, in gate order
    assert_openings: List[int]  # one M tag per assert_zero gate


@dataclass
class BatchedCheck:
    """Round 3: prover -> verifier."""

    U: int
    V: int


# ---- Prover walker ---------------------------------------------------------


@dataclass
class _ProverWalker:
    circuit: Circuit
    witness: List[int]
    share: VoleProverShare
    field: Fp = F
    wires: dict = dc_field(default_factory=dict)
    d_values: List[int] = dc_field(default_factory=list)
    assert_openings: List[int] = dc_field(default_factory=list)
    mul_triples: List[Tuple[ProverWire, ProverWire, ProverWire]] = dc_field(default_factory=list)
    mask_a: int = 0
    mask_b: int = 0

    def commit(self) -> CommitMessage:
        if len(self.witness) != self.circuit.num_inputs:
            raise ValueError(
                f"witness length {len(self.witness)} != num inputs {self.circuit.num_inputs}"
            )
        vole = iter(zip(self.share.u, self.share.v))
        wit = iter(self.witness)
        for g in self.circuit.gates:
            self._exec(g, wit, vole)
        # Reserve one final VOLE element to mask the batched mul check.
        self.mask_a, self.mask_b = next(vole)
        return CommitMessage(self.d_values, self.assert_openings)

    def _exec(self, g: Gate, wit: Iterator[int], vole: Iterator[Tuple[int, int]]) -> None:
        F_ = self.field
        op = g.op
        if op is Op.INPUT:
            x = F_.encode(next(wit))
            u, v = next(vole)
            wire, d = prover_commit(x, u, v, F_)
            self.wires[g.out] = wire
            self.d_values.append(d)
        elif op is Op.CONST:
            (c,) = g.args
            self.wires[g.out] = prover_const(F_.encode(c))
        elif op is Op.ADD:
            a, b = g.args
            self.wires[g.out] = self.wires[a].add(self.wires[b], F_)
        elif op is Op.SUB:
            a, b = g.args
            self.wires[g.out] = self.wires[a].sub(self.wires[b], F_)
        elif op is Op.ADD_CONST:
            a, c = g.args
            self.wires[g.out] = self.wires[a].add_const(F_.encode(c), F_)
        elif op is Op.MUL_CONST:
            a, c = g.args
            self.wires[g.out] = self.wires[a].mul_const(F_.encode(c), F_)
        elif op is Op.MUL:
            a, b = g.args
            xw = self.wires[a]
            yw = self.wires[b]
            z = F_.mul(xw.x, yw.x)
            u, v = next(vole)
            zw, d = prover_commit(z, u, v, F_)
            self.wires[g.out] = zw
            self.d_values.append(d)
            self.mul_triples.append((xw, yw, zw))
        elif op is Op.ASSERT_ZERO:
            (a,) = g.args
            self.assert_openings.append(open_to_zero(self.wires[a]))
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unknown op {op}")

    def batched_check(self, chi: int) -> BatchedCheck:
        """Compute (U, V) such that U + V * Delta = sum_i chi^i * B_i + c.

        Honest prover: A0_i = M_x M_y, A1_i = x M_y + y M_x - M_z.
        We then mask with the reserved VOLE element to make U, V uniform.
        """
        F_ = self.field
        U_raw = 0
        V_raw = 0
        chi_i = 1
        for xw, yw, zw in self.mul_triples:
            chi_i = F_.mul(chi_i, chi)  # start at chi^1
            A0 = F_.mul(xw.m, yw.m)
            A1 = F_.sub(
                F_.add(F_.mul(xw.x, yw.m), F_.mul(yw.x, xw.m)),
                zw.m,
            )
            U_raw = F_.add(U_raw, F_.mul(chi_i, A0))
            V_raw = F_.add(V_raw, F_.mul(chi_i, A1))
        return BatchedCheck(
            U=F_.add(U_raw, self.mask_b),
            V=F_.add(V_raw, self.mask_a),
        )


# ---- Verifier walker -------------------------------------------------------


@dataclass
class _VerifierWalker:
    circuit: Circuit
    share: VoleVerifierShare
    field: Fp = F
    wires: dict = dc_field(default_factory=dict)
    mul_keys: List[Tuple[VerifierWire, VerifierWire, VerifierWire]] = dc_field(default_factory=list)
    assert_keys: List[VerifierWire] = dc_field(default_factory=list)
    mask_c: int = 0
    _assert_openings: List[int] = dc_field(default_factory=list)

    def receive(self, msg: CommitMessage) -> None:
        F_ = self.field
        d_iter = iter(msg.d_values)
        w_iter = iter(self.share.w)
        delta = self.share.delta
        for g in self.circuit.gates:
            op = g.op
            if op is Op.INPUT:
                d = next(d_iter)
                w = next(w_iter)
                self.wires[g.out] = verifier_receive(d, w, delta, F_)
            elif op is Op.CONST:
                (c,) = g.args
                self.wires[g.out] = verifier_const(F_.encode(c), delta, F_)
            elif op is Op.ADD:
                a, b = g.args
                self.wires[g.out] = self.wires[a].add(self.wires[b], F_)
            elif op is Op.SUB:
                a, b = g.args
                self.wires[g.out] = self.wires[a].sub(self.wires[b], F_)
            elif op is Op.ADD_CONST:
                a, c = g.args
                self.wires[g.out] = self.wires[a].add_const(F_.encode(c), delta, F_)
            elif op is Op.MUL_CONST:
                a, c = g.args
                self.wires[g.out] = self.wires[a].mul_const(F_.encode(c), F_)
            elif op is Op.MUL:
                a, b = g.args
                d = next(d_iter)
                w = next(w_iter)
                kz = verifier_receive(d, w, delta, F_)
                self.wires[g.out] = kz
                self.mul_keys.append((self.wires[a], self.wires[b], kz))
            elif op is Op.ASSERT_ZERO:
                (a,) = g.args
                self.assert_keys.append(self.wires[a])
            else:  # pragma: no cover
                raise AssertionError(f"unknown op {op}")
        # Last VOLE element is the masking element.
        self.mask_c = next(w_iter)
        self._assert_openings = list(msg.assert_openings)

    def check_assertions(self) -> bool:
        # For a wire claimed to be zero, K = M + Delta * 0 = M.
        if len(self._assert_openings) != len(self.assert_keys):
            return False
        for opened_m, key_wire in zip(self._assert_openings, self.assert_keys):
            if key_wire.k != opened_m:
                return False
        return True

    def check_batched(self, chi: int, msg: BatchedCheck) -> bool:
        F_ = self.field
        delta = self.share.delta
        W = 0
        chi_i = 1
        for kx, ky, kz in self.mul_keys:
            chi_i = F_.mul(chi_i, chi)
            B = F_.sub(F_.mul(kx.k, ky.k), F_.mul(delta, kz.k))
            W = F_.add(W, F_.mul(chi_i, B))
        lhs = F_.add(W, self.mask_c)
        rhs = F_.add(msg.U, F_.mul(msg.V, delta))
        return lhs == rhs


# ---- Public API ------------------------------------------------------------


def prove(circuit: Circuit, witness: List[int], share: VoleProverShare, field: Fp = F):
    """Run the prover side. Returns (commit_msg, batched_fn).

    ``batched_fn(chi)`` produces the round-3 message after the verifier
    sends its challenge. Splitting the prover this way keeps the
    interaction explicit.
    """
    walker = _ProverWalker(circuit=circuit, witness=witness, share=share, field=field)
    msg1 = walker.commit()
    return msg1, walker.batched_check


def verify(
    circuit: Circuit,
    share: VoleVerifierShare,
    msg1: CommitMessage,
    chi: int,
    msg2: BatchedCheck,
    field: Fp = F,
) -> bool:
    walker = _VerifierWalker(circuit=circuit, share=share, field=field)
    walker.receive(msg1)
    if not walker.check_assertions():
        return False
    return walker.check_batched(chi, msg2)


def run(circuit: Circuit, witness: List[int], field: Fp = F) -> bool:
    """End-to-end: setup + prove + challenge + verify, all in-process."""
    p_share, v_share = trusted_dealer_setup(circuit.vole_count(), field)
    msg1, batch = prove(circuit, witness, p_share, field)
    chi = field.rand_nonzero()
    msg2 = batch(chi)
    return verify(circuit, v_share, msg1, chi, msg2, field)
