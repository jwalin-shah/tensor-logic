"""QuickSilver over F_2 with subspace VOLE in GF(2^128).

This is the binary-circuit instantiation of the same protocol that
``protocol.py`` runs over a prime field. Wire values are bits; IT-MAC
tags and keys live in GF(2^128); soundness error is ~m * 2^-128 for
``m`` AND gates.

Wire semantics
--------------

For each committed bit ``x in {0,1}``, the prover holds
``M in GF(2^128)`` and the verifier holds ``K in GF(2^128)`` with

    K = M + Delta * x      (Delta in GF(2^128), addition is XOR).

Subspace VOLE preprocessing yields, for each element, ``u in {0,1}``,
``v in GF(2^128)`` for the prover and ``w = u * Delta + v`` for the
verifier. To commit to bit ``x`` the prover sends ``d = x XOR u``; if
``d = 0`` the verifier sets ``K = w``, else ``K = w + Delta``. Both
sides arrive at the IT-MAC relation above with ``M = v``.

Gates supported
---------------

XOR (free), AND (one VOLE element + a slot in the batched check),
NOT (free), constants, and an ``assert_zero`` constraint that opens a
wire claimed to be 0. NAND/OR/etc. compose from these.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Iterator, List, Tuple

from quicksilver.gf2k import GF128, GF2k


# ---- Subspace VOLE -------------------------------------------------------


@dataclass
class SubspaceVoleProverShare:
    u: List[int]  # bits
    v: List[int]  # GF(2^128) elements

    def __len__(self) -> int:
        return len(self.u)


@dataclass
class SubspaceVoleVerifierShare:
    delta: int  # GF(2^128)
    w: List[int]

    def __len__(self) -> int:
        return len(self.w)


def trusted_dealer_setup(
    n: int, mac_field: GF2k = GF128, delta: int | None = None
) -> Tuple[SubspaceVoleProverShare, SubspaceVoleVerifierShare]:
    if delta is None:
        delta = mac_field.rand()
    u = [secrets.randbits(1) for _ in range(n)]
    v = [mac_field.rand() for _ in range(n)]
    # u_i is 0 or 1, so u_i * Delta is just 0 or Delta.
    w = [(delta if u_i else 0) ^ v_i for u_i, v_i in zip(u, v)]
    return (
        SubspaceVoleProverShare(u=u, v=v),
        SubspaceVoleVerifierShare(delta=delta, w=w),
    )


# ---- Boolean wires (IT-MACs) ---------------------------------------------


@dataclass(frozen=True)
class BoolProverWire:
    x: int  # 0 or 1
    m: int  # GF(2^128)


@dataclass(frozen=True)
class BoolVerifierWire:
    k: int  # GF(2^128)


def _commit_bit(
    bit: int, u: int, v: int, mac_field: GF2k = GF128
) -> tuple[BoolProverWire, int]:
    d = bit ^ u
    return BoolProverWire(x=bit, m=v), d


def _verifier_receive(
    d: int, w: int, delta: int, mac_field: GF2k = GF128
) -> BoolVerifierWire:
    # K = w + Delta * d_bit. d is 0 or 1, so this is just w or w XOR Delta.
    return BoolVerifierWire(k=(w ^ delta) if d else w)


# ---- Circuit DSL ---------------------------------------------------------


class BOp(Enum):
    INPUT = "input"
    CONST = "const"
    XOR = "xor"
    NOT = "not"
    AND = "and"
    XOR_CONST = "xor_const"
    ASSERT_ZERO = "assert_zero"


@dataclass
class BGate:
    op: BOp
    args: tuple
    out: int | None = None


@dataclass
class BoolCircuit:
    gates: List[BGate] = dc_field(default_factory=list)
    num_wires: int = 0
    num_inputs: int = 0
    num_ands: int = 0
    num_asserts: int = 0

    def _new(self) -> int:
        wid = self.num_wires
        self.num_wires += 1
        return wid

    def input(self) -> int:
        wid = self._new()
        self.gates.append(BGate(BOp.INPUT, (), out=wid))
        self.num_inputs += 1
        return wid

    def const(self, c: int) -> int:
        if c not in (0, 1):
            raise ValueError("boolean const must be 0 or 1")
        wid = self._new()
        self.gates.append(BGate(BOp.CONST, (c,), out=wid))
        return wid

    def xor(self, a: int, b: int) -> int:
        wid = self._new()
        self.gates.append(BGate(BOp.XOR, (a, b), out=wid))
        return wid

    def xor_const(self, a: int, c: int) -> int:
        if c not in (0, 1):
            raise ValueError("boolean const must be 0 or 1")
        wid = self._new()
        self.gates.append(BGate(BOp.XOR_CONST, (a, c), out=wid))
        return wid

    def not_(self, a: int) -> int:
        return self.xor_const(a, 1)

    def and_(self, a: int, b: int) -> int:
        wid = self._new()
        self.gates.append(BGate(BOp.AND, (a, b), out=wid))
        self.num_ands += 1
        return wid

    def or_(self, a: int, b: int) -> int:
        # a OR b = a XOR b XOR (a AND b)
        return self.xor(self.xor(a, b), self.and_(a, b))

    def assert_zero(self, a: int) -> None:
        self.gates.append(BGate(BOp.ASSERT_ZERO, (a,)))
        self.num_asserts += 1

    def assert_eq(self, a: int, b: int) -> None:
        self.assert_zero(self.xor(a, b))

    def assert_eq_const(self, a: int, c: int) -> None:
        self.assert_zero(self.xor_const(a, c))

    def vole_count(self) -> int:
        return self.num_inputs + self.num_ands + 1


# ---- Protocol ------------------------------------------------------------


@dataclass
class BCommitMessage:
    d_values: List[int]      # bits
    assert_openings: List[int]  # GF(2^128) elements


@dataclass
class BBatchedCheck:
    U: int
    V: int


@dataclass
class _BProverWalker:
    circuit: BoolCircuit
    witness: List[int]
    share: SubspaceVoleProverShare
    mac_field: GF2k = GF128
    wires: dict = dc_field(default_factory=dict)
    d_values: List[int] = dc_field(default_factory=list)
    assert_openings: List[int] = dc_field(default_factory=list)
    and_triples: List[Tuple[BoolProverWire, BoolProverWire, BoolProverWire]] = dc_field(default_factory=list)
    mask_a: int = 0
    mask_b: int = 0

    def commit(self) -> BCommitMessage:
        if len(self.witness) != self.circuit.num_inputs:
            raise ValueError(
                f"witness length {len(self.witness)} != "
                f"num inputs {self.circuit.num_inputs}"
            )
        vole = iter(zip(self.share.u, self.share.v))
        wit = iter(self.witness)
        for g in self.circuit.gates:
            self._exec(g, wit, vole)
        # Reserve one final VOLE element for batched-check masking.
        self.mask_a, self.mask_b = next(vole)
        return BCommitMessage(self.d_values, self.assert_openings)

    def _exec(self, g, wit, vole):
        f = self.mac_field
        op = g.op
        if op is BOp.INPUT:
            x = int(next(wit)) & 1
            u, v = next(vole)
            wire, d = _commit_bit(x, u, v, f)
            self.wires[g.out] = wire
            self.d_values.append(d)
        elif op is BOp.CONST:
            (c,) = g.args
            self.wires[g.out] = BoolProverWire(x=c, m=0)
        elif op is BOp.XOR:
            a, b = g.args
            wa, wb = self.wires[a], self.wires[b]
            self.wires[g.out] = BoolProverWire(x=wa.x ^ wb.x, m=wa.m ^ wb.m)
        elif op is BOp.XOR_CONST:
            a, c = g.args
            wa = self.wires[a]
            # K_z = K_a + Delta * c. Prover side: M unchanged, x toggles.
            self.wires[g.out] = BoolProverWire(x=wa.x ^ c, m=wa.m)
        elif op is BOp.AND:
            a, b = g.args
            wa, wb = self.wires[a], self.wires[b]
            z = wa.x & wb.x
            u, v = next(vole)
            wz, d = _commit_bit(z, u, v, f)
            self.wires[g.out] = wz
            self.d_values.append(d)
            self.and_triples.append((wa, wb, wz))
        elif op is BOp.ASSERT_ZERO:
            (a,) = g.args
            w = self.wires[a]
            if w.x != 0:
                raise ValueError("prover wire claimed to be zero is nonzero")
            self.assert_openings.append(w.m)
        else:  # pragma: no cover
            raise AssertionError(f"unknown op {op}")

    def batched_check(self, chi: int) -> BBatchedCheck:
        f = self.mac_field
        U_raw = 0
        V_raw = 0
        chi_i = 1
        for xw, yw, zw in self.and_triples:
            chi_i = f.mul(chi_i, chi)
            # In characteristic 2: B = K_x*K_y + Delta*K_z; A_0 = M_x*M_y;
            # A_1 = x*M_y + y*M_x + M_z (with x,y in {0,1}).
            A0 = f.mul(xw.m, yw.m)
            xMy = yw.m if xw.x else 0
            yMx = xw.m if yw.x else 0
            A1 = xMy ^ yMx ^ zw.m
            U_raw ^= f.mul(chi_i, A0)
            V_raw ^= f.mul(chi_i, A1)
        return BBatchedCheck(U=U_raw ^ self.mask_b, V=V_raw ^ self.mask_a)


@dataclass
class _BVerifierWalker:
    circuit: BoolCircuit
    share: SubspaceVoleVerifierShare
    mac_field: GF2k = GF128
    wires: dict = dc_field(default_factory=dict)
    and_keys: List[Tuple[BoolVerifierWire, BoolVerifierWire, BoolVerifierWire]] = dc_field(default_factory=list)
    assert_keys: List[BoolVerifierWire] = dc_field(default_factory=list)
    mask_c: int = 0
    _openings: List[int] = dc_field(default_factory=list)

    def receive(self, msg: BCommitMessage) -> None:
        f = self.mac_field
        d_iter = iter(msg.d_values)
        w_iter = iter(self.share.w)
        delta = self.share.delta
        for g in self.circuit.gates:
            op = g.op
            if op is BOp.INPUT:
                d = next(d_iter)
                w = next(w_iter)
                self.wires[g.out] = _verifier_receive(d, w, delta, f)
            elif op is BOp.CONST:
                (c,) = g.args
                self.wires[g.out] = BoolVerifierWire(k=delta if c else 0)
            elif op is BOp.XOR:
                a, b = g.args
                self.wires[g.out] = BoolVerifierWire(
                    k=self.wires[a].k ^ self.wires[b].k
                )
            elif op is BOp.XOR_CONST:
                a, c = g.args
                self.wires[g.out] = BoolVerifierWire(
                    k=self.wires[a].k ^ (delta if c else 0)
                )
            elif op is BOp.AND:
                a, b = g.args
                d = next(d_iter)
                w = next(w_iter)
                kz = _verifier_receive(d, w, delta, f)
                self.wires[g.out] = kz
                self.and_keys.append((self.wires[a], self.wires[b], kz))
            elif op is BOp.ASSERT_ZERO:
                (a,) = g.args
                self.assert_keys.append(self.wires[a])
            else:  # pragma: no cover
                raise AssertionError(f"unknown op {op}")
        self.mask_c = next(w_iter)
        self._openings = list(msg.assert_openings)

    def check_assertions(self) -> bool:
        if len(self._openings) != len(self.assert_keys):
            return False
        # K = M + Delta * 0 = M for a wire claimed to be 0.
        for opened_m, key in zip(self._openings, self.assert_keys):
            if key.k != opened_m:
                return False
        return True

    def check_batched(self, chi: int, msg: BBatchedCheck) -> bool:
        f = self.mac_field
        delta = self.share.delta
        W = 0
        chi_i = 1
        for kx, ky, kz in self.and_keys:
            chi_i = f.mul(chi_i, chi)
            B = f.mul(kx.k, ky.k) ^ f.mul(delta, kz.k)
            W ^= f.mul(chi_i, B)
        lhs = W ^ self.mask_c
        rhs = msg.U ^ f.mul(msg.V, delta)
        return lhs == rhs


def prove(
    circuit: BoolCircuit,
    witness: List[int],
    share: SubspaceVoleProverShare,
    mac_field: GF2k = GF128,
):
    walker = _BProverWalker(
        circuit=circuit, witness=witness, share=share, mac_field=mac_field
    )
    return walker.commit(), walker.batched_check


def verify(
    circuit: BoolCircuit,
    share: SubspaceVoleVerifierShare,
    msg1: BCommitMessage,
    chi: int,
    msg2: BBatchedCheck,
    mac_field: GF2k = GF128,
) -> bool:
    walker = _BVerifierWalker(
        circuit=circuit, share=share, mac_field=mac_field
    )
    walker.receive(msg1)
    if not walker.check_assertions():
        return False
    return walker.check_batched(chi, msg2)


def run(
    circuit: BoolCircuit, witness: List[int], mac_field: GF2k = GF128
) -> bool:
    p_share, v_share = trusted_dealer_setup(circuit.vole_count(), mac_field)
    msg1, batched = prove(circuit, witness, p_share, mac_field)
    chi = mac_field.rand_nonzero()
    msg2 = batched(chi)
    return verify(circuit, v_share, msg1, chi, msg2, mac_field)
