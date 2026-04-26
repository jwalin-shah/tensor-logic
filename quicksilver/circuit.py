"""Tiny arithmetic-circuit DSL.

A circuit is a sequence of gates referencing previously-defined wires
by integer id. The same ``Circuit`` object is consumed by both prover
and verifier in ``protocol.py``; they walk the gate list in lockstep
and accumulate their respective wire views.

Supported operations
--------------------

input()                fresh secret input from the prover
const(c)               public constant
add(a, b)              wire a + wire b
sub(a, b)              wire a - wire b
add_const(a, c)        wire a + public constant
mul_const(a, c)        wire a * public constant
mul(a, b)              wire a * wire b   (needs one VOLE element + check)
assert_zero(a)         add an output check that wire a equals 0
assert_eq(a, c)        sugar for assert_zero(sub(a, const(c)))

Costs
-----

VOLE elements consumed: ``num_inputs + num_muls + 1`` (the trailing
"+1" is the masking element for the batched multiplication check).
Communication: one field element per input, two per mul gate (the
``A0, A1`` opening in the batched check is amortised), one MAC per
output assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Op(Enum):
    INPUT = "input"
    CONST = "const"
    ADD = "add"
    SUB = "sub"
    ADD_CONST = "add_const"
    MUL_CONST = "mul_const"
    MUL = "mul"
    ASSERT_ZERO = "assert_zero"


@dataclass
class Gate:
    op: Op
    args: tuple  # interpretation depends on op
    out: int | None = None  # wire id this gate produces (None for asserts)


@dataclass
class Circuit:
    gates: List[Gate] = field(default_factory=list)
    num_wires: int = 0
    num_inputs: int = 0
    num_muls: int = 0
    num_asserts: int = 0

    def _new_wire(self) -> int:
        wid = self.num_wires
        self.num_wires += 1
        return wid

    def input(self) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.INPUT, (), out=wid))
        self.num_inputs += 1
        return wid

    def const(self, c: int) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.CONST, (c,), out=wid))
        return wid

    def add(self, a: int, b: int) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.ADD, (a, b), out=wid))
        return wid

    def sub(self, a: int, b: int) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.SUB, (a, b), out=wid))
        return wid

    def add_const(self, a: int, c: int) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.ADD_CONST, (a, c), out=wid))
        return wid

    def mul_const(self, a: int, c: int) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.MUL_CONST, (a, c), out=wid))
        return wid

    def mul(self, a: int, b: int) -> int:
        wid = self._new_wire()
        self.gates.append(Gate(Op.MUL, (a, b), out=wid))
        self.num_muls += 1
        return wid

    def assert_zero(self, a: int) -> None:
        self.gates.append(Gate(Op.ASSERT_ZERO, (a,)))
        self.num_asserts += 1

    def assert_eq(self, a: int, c: int) -> None:
        # a - c == 0
        diff = self.sub(a, self.const(c))
        self.assert_zero(diff)

    # Convenience: total VOLE elements consumed by the protocol.
    def vole_count(self) -> int:
        return self.num_inputs + self.num_muls + 1  # +1 for batched-check mask
