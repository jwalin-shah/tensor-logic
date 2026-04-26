"""ZK proof of graph reachability via the tensor-logic einsum recurrence.

The Datalog rule for reachability,

    Path(x, z) :- Edge(x, z).
    Path(x, z) :- Path(x, y), Edge(y, z).

is the same operation as the einsum

    next_frontier = step( frontier @ Edge ).

We use that recurrence to express a zero-knowledge statement:

    "I know a graph G on n vertices and a length-k walk through G
     from public source s to public target t."

The prover commits to G (as an n*n adjacency matrix) and to k-1
intermediate one-hot frontier vectors. The boundary frontiers are
fixed to the one-hot indicators of s and t (public). The QuickSilver
circuit enforces:

    1. Every entry of G is in {0, 1}                       (booleanity)
    2. Every committed frontier is one-hot                 (booleanity + sum-to-one)
    3. For each step i in 0..k-1:
           sum_{u, v}  alpha_i[u] * G[u][v] * alpha_{i+1}[v]  ==  1
       i.e. there is at least one edge between the vertex
       indicated by alpha_i and the vertex indicated by alpha_{i+1}.

The verifier learns nothing about G or the path beyond (n, k, s, t).

Cost (in QuickSilver multiplication gates):

    booleanity of G                 :  n^2
    booleanity of intermediate alphas: (k-1) * n
    step transitions                : 2 * (n^2) + n   for an interior step
                                       n           for a boundary step

So for n vertices, walk length k, total muls ~= n^2 * k. The pure-Python
prover handles n=8, k=4 in well under a second.

Tying back to ``transitive_closure.py``: the same step
``alpha @ Edge`` that drives boolean reachability is the inner product
that this circuit's edge constraint computes. ZK adds nothing
mathematically; the einsum is the proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from quicksilver.circuit import Circuit
from quicksilver.field import F, Fp
from quicksilver.protocol import run as run_protocol


@dataclass
class ReachabilityCircuit:
    circuit: Circuit
    n: int
    k: int
    source: int
    target: int


def build_circuit(n: int, k: int, source: int, target: int) -> ReachabilityCircuit:
    """Build the reachability circuit. ``k`` is the walk length (number of edges).

    Vertices are integers 0..n-1. Source and target are public; the
    intermediate vertices and the entire graph are committed.
    """
    if not (0 <= source < n and 0 <= target < n):
        raise ValueError("source/target out of range")
    if k < 1:
        raise ValueError("walk length must be at least 1")

    c = Circuit()

    # ---- 1. Commit edge matrix and enforce booleanity ----------------------
    E = [[c.input() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            _enforce_boolean(c, E[i][j])

    # ---- 2. Build the k+1 frontier vectors --------------------------------
    # alpha_0 and alpha_k are public one-hot vectors at source/target.
    # alpha_1..alpha_{k-1} are committed and constrained to be one-hot.
    alpha: List[List[_Slot]] = []
    for i in range(k + 1):
        if i == 0:
            alpha.append([_Public(1 if j == source else 0) for j in range(n)])
        elif i == k:
            alpha.append([_Public(1 if j == target else 0) for j in range(n)])
        else:
            row = [c.input() for _ in range(n)]
            for w in row:
                _enforce_boolean(c, w)
            _enforce_one_hot(c, row)
            alpha.append([_Wire(w) for w in row])

    # ---- 3. Each step contributes one edge-existence constraint -----------
    for i in range(k):
        _enforce_step(c, alpha[i], E, alpha[i + 1], n)

    return ReachabilityCircuit(circuit=c, n=n, k=k, source=source, target=target)


def assemble_witness(
    rc: ReachabilityCircuit,
    edge_matrix: Sequence[Sequence[int]],
    path: Sequence[int],
) -> List[int]:
    """Pack the prover's witness into the order ``circuit.input()`` consumed it."""
    n, k = rc.n, rc.k
    if len(edge_matrix) != n or any(len(row) != n for row in edge_matrix):
        raise ValueError(f"edge_matrix must be {n}x{n}")
    if len(path) != k + 1:
        raise ValueError(f"path must have length {k + 1} (k+1 vertices)")
    if path[0] != rc.source or path[-1] != rc.target:
        raise ValueError("path endpoints do not match circuit source/target")
    for v in path:
        if not (0 <= v < n):
            raise ValueError(f"path vertex {v} out of range")
    for i in range(k):
        u, v = path[i], path[i + 1]
        if edge_matrix[u][v] not in (0, 1):
            raise ValueError("edge_matrix entries must be 0 or 1")
        if edge_matrix[u][v] != 1:
            raise ValueError(
                f"path uses non-edge ({u} -> {v}); witness is invalid"
            )

    witness: List[int] = []
    # First inputs were the n*n edge entries (row-major).
    for i in range(n):
        for j in range(n):
            witness.append(int(edge_matrix[i][j]))
    # Then the (k-1) intermediate frontier one-hots.
    for i in range(1, k):
        v = path[i]
        for j in range(n):
            witness.append(1 if j == v else 0)
    return witness


def prove_path(
    n: int,
    k: int,
    source: int,
    target: int,
    edge_matrix: Sequence[Sequence[int]],
    path: Sequence[int],
    field: Fp = F,
) -> bool:
    """End-to-end: build circuit, assemble witness, run prover/verifier."""
    rc = build_circuit(n, k, source, target)
    witness = assemble_witness(rc, edge_matrix, path)
    return run_protocol(rc.circuit, witness, field)


# ---- internal helpers --------------------------------------------------------

# A slot in a frontier vector is either a public constant (already known
# on both sides) or a committed wire id.


@dataclass(frozen=True)
class _Public:
    value: int


@dataclass(frozen=True)
class _Wire:
    wid: int


_Slot = _Public | _Wire


def _enforce_boolean(c: Circuit, w: int) -> None:
    """Constrain a wire to be in {0, 1} via x*(x-1) == 0."""
    w_minus_1 = c.add_const(w, -1)
    c.assert_zero(c.mul(w, w_minus_1))


def _enforce_one_hot(c: Circuit, wires: Sequence[int]) -> None:
    """Constrain ``sum_i wires[i] == 1`` (linear, no mul gates)."""
    acc = c.const(0)
    for w in wires:
        acc = c.add(acc, w)
    c.assert_zero(c.add_const(acc, -1))


def _enforce_step(
    c: Circuit, a: List[_Slot], E: List[List[int]], b: List[_Slot], n: int
) -> None:
    """Enforce sum_{u,v} a[u] * E[u][v] * b[v] == 1.

    Decomposes into: w[v] = sum_u a[u] * E[u][v]; then sum_v w[v] * b[v].
    Constants in ``a`` or ``b`` shrink mul-gate count automatically.
    """
    w = [_inner_with_coeffs(c, a, [E[u][v] for u in range(n)]) for v in range(n)]
    total = _inner_two_wire_lists(c, w, b)
    c.assert_zero(c.add_const(total, -1))


def _inner_with_coeffs(c: Circuit, a: List[_Slot], wires: List[int]) -> int:
    """sum_u a[u] * wires[u], where a[u] may be public (use mul_const) or a wire."""
    acc = c.const(0)
    for slot, w in zip(a, wires):
        if isinstance(slot, _Public):
            if slot.value == 0:
                continue
            if slot.value == 1:
                acc = c.add(acc, w)
            else:
                acc = c.add(acc, c.mul_const(w, slot.value))
        else:
            acc = c.add(acc, c.mul(slot.wid, w))
    return acc


def _inner_two_wire_lists(c: Circuit, w: List[int], b: List[_Slot]) -> int:
    """sum_v w[v] * b[v], with b possibly public."""
    acc = c.const(0)
    for wv, slot in zip(w, b):
        if isinstance(slot, _Public):
            if slot.value == 0:
                continue
            if slot.value == 1:
                acc = c.add(acc, wv)
            else:
                acc = c.add(acc, c.mul_const(wv, slot.value))
        else:
            acc = c.add(acc, c.mul(wv, slot.wid))
    return acc
