"""Compile a tensor-logic einsum into a QuickSilver arithmetic circuit.

The repo's premise -- that a Datalog rule head and an einsum are the
same operation -- gives a one-line frontend for ZK: any rule head you
can write as an einsum becomes a QuickSilver circuit that proves
"I know inputs such that this einsum produces this output."

API
---

::

    layout = compile_einsum(circuit, spec, input_wires, shapes)
    # layout[multi_index] -> wire id for that output element

Where ``input_wires[t]`` is a flat row-major list of wire ids for the
t-th input tensor. The caller decides whether each tensor is public
(``alloc_constant_tensor``), prover-private (``alloc_input_tensor``),
or computed by an earlier subgraph.

Cost model
----------

For an einsum with output indices ``O``, contraction indices ``C``,
and ``T`` input tensors, the compiler emits

    prod_{i in O} d_i * prod_{j in C} d_j * (T - 1)     mul gates
    prod_{i in O} d_i * prod_{j in C} d_j               add gates

i.e. it materialises the contraction explicitly. No optimisation
passes (no early reductions, no factoring out shared sub-products).
That keeps the lowering as transparent as possible; for serious work
you would lower via opt-einsum's contraction path, but the asymptotic
cost is the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

from quicksilver.circuit import Circuit


def parse_einsum(spec: str) -> tuple[list[str], str]:
    """Split ``'ij,jk->ik'`` into (['ij','jk'], 'ik')."""
    if "->" not in spec:
        raise ValueError("einsum spec must contain '->'")
    lhs, rhs = spec.split("->")
    inputs = [s.strip() for s in lhs.split(",")]
    output = rhs.strip()
    if not all(part for part in inputs):
        raise ValueError("empty input spec")
    return inputs, output


def resolve_dims(
    input_specs: Sequence[str], shapes: Sequence[Sequence[int]]
) -> Dict[str, int]:
    """Match index symbols to dimensions, checking consistency."""
    if len(input_specs) != len(shapes):
        raise ValueError("number of input specs and shapes must match")
    dims: Dict[str, int] = {}
    for spec, shape in zip(input_specs, shapes):
        if len(spec) != len(shape):
            raise ValueError(
                f"spec '{spec}' has rank {len(spec)} but shape has {len(shape)}"
            )
        for sym, d in zip(spec, shape):
            if sym in dims:
                if dims[sym] != d:
                    raise ValueError(
                        f"index '{sym}' has inconsistent dims {dims[sym]} vs {d}"
                    )
            else:
                dims[sym] = d
    return dims


def _flat_index(spec: str, assignment: Dict[str, int], dims: Dict[str, int]) -> int:
    """Row-major flat offset for the index assignment in a tensor of given spec."""
    offset = 0
    for sym in spec:
        offset = offset * dims[sym] + assignment[sym]
    return offset


def alloc_input_tensor(c: Circuit, shape: Sequence[int]) -> List[int]:
    """Allocate a row-major flat list of fresh ``c.input()`` wires."""
    n = 1
    for d in shape:
        n *= d
    return [c.input() for _ in range(n)]


def alloc_constant_tensor(c: Circuit, values: Sequence[int]) -> List[int]:
    """Lift a flat sequence of ints into ``c.const`` wires."""
    return [c.const(v) for v in values]


def compile_einsum(
    c: Circuit,
    spec: str,
    input_wires: Sequence[Sequence[int]],
    shapes: Sequence[Sequence[int]],
) -> Dict[Tuple[int, ...], int]:
    """Append the einsum subgraph to ``c``.

    Returns ``output_wires[(i_1, ..., i_k)] -> wire id`` for each
    multi-index in the output's row-major iteration order.
    """
    input_specs, output_spec = parse_einsum(spec)
    if len(input_wires) != len(input_specs):
        raise ValueError(
            f"got {len(input_wires)} input wire lists for {len(input_specs)} tensors"
        )
    dims = resolve_dims(input_specs, shapes)
    for sym in output_spec:
        if sym not in dims:
            raise ValueError(f"output index '{sym}' missing from inputs")

    contraction_syms = [s for s in dims if s not in output_spec]
    output_dims = [dims[s] for s in output_spec]
    contraction_dims = [dims[s] for s in contraction_syms]

    out: Dict[Tuple[int, ...], int] = {}
    for out_assign in product(*(range(d) for d in output_dims)):
        out_map = dict(zip(output_spec, out_assign))
        acc = c.const(0)
        contraction_iter = (
            product(*(range(d) for d in contraction_dims))
            if contraction_dims
            else [()]
        )
        for con_assign in contraction_iter:
            con_map = dict(zip(contraction_syms, con_assign))
            full = {**out_map, **con_map}
            # Multiply one input tensor entry per input.
            term = None
            for t, ispec in enumerate(input_specs):
                offset = _flat_index(ispec, full, dims)
                wid = input_wires[t][offset]
                term = wid if term is None else c.mul(term, wid)
            if term is None:  # only happens for "->" with no inputs (degenerate)
                term = c.const(1)
            acc = c.add(acc, term)
        out[tuple(out_assign)] = acc
    return out


# ---- High-level helpers --------------------------------------------------


def assert_einsum_equals(
    c: Circuit,
    spec: str,
    input_wires: Sequence[Sequence[int]],
    shapes: Sequence[Sequence[int]],
    expected_values: Sequence[int],
) -> Dict[Tuple[int, ...], int]:
    """Compile the einsum and assert each output element equals a public value.

    ``expected_values`` is a row-major flat list over the output multi-indices.
    """
    out = compile_einsum(c, spec, input_wires, shapes)
    iter_specs, _ = parse_einsum(spec)
    dims = resolve_dims(iter_specs, shapes)
    output_dims = [dims[s] for s in parse_einsum(spec)[1]]
    n_out = 1
    for d in output_dims:
        n_out *= d
    if len(expected_values) != n_out:
        raise ValueError(
            f"expected_values has {len(expected_values)} entries; need {n_out}"
        )
    flat_iter = product(*(range(d) for d in output_dims))
    for idx, expected in zip(flat_iter, expected_values):
        c.assert_eq(out[idx], int(expected))
    return out


def flatten_row_major(tensor) -> List[int]:
    """Flatten a nested list (of any rank) in row-major order."""
    out: List[int] = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            out.append(int(x))

    walk(tensor)
    return out


def evaluate_einsum(spec: str, tensors: Sequence) -> List[int]:
    """Reference implementation, used by tests/demos to compute the expected output."""
    input_specs, output_spec = parse_einsum(spec)
    flats = [flatten_row_major(t) for t in tensors]
    shapes = []
    for spec_i, t in zip(input_specs, tensors):
        shape = []
        cur = t
        for _ in spec_i:
            shape.append(len(cur))
            cur = cur[0] if isinstance(cur, (list, tuple)) else cur
        shapes.append(shape)
    dims = resolve_dims(input_specs, shapes)

    contraction_syms = [s for s in dims if s not in output_spec]
    output_dims = [dims[s] for s in output_spec]
    contraction_dims = [dims[s] for s in contraction_syms]

    n_out = 1
    for d in output_dims:
        n_out *= d
    out = [0] * n_out

    for i, out_assign in enumerate(product(*(range(d) for d in output_dims))):
        out_map = dict(zip(output_spec, out_assign))
        total = 0
        contraction_iter = (
            product(*(range(d) for d in contraction_dims))
            if contraction_dims
            else [()]
        )
        for con_assign in contraction_iter:
            con_map = dict(zip(contraction_syms, con_assign))
            full = {**out_map, **con_map}
            term = 1
            for t, spec_i in enumerate(input_specs):
                offset = _flat_index(spec_i, full, dims)
                term *= flats[t][offset]
            total += term
        out[i] = total
    return out
