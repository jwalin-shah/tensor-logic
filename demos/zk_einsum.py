"""ZK proofs of tensor-logic einsum identities.

The same einsum-as-rule-head story behind ``transitive_closure.py`` and
``train_kg.py`` becomes a ZK frontend: any rule head expressible as an
einsum is automatically a QuickSilver circuit. This demo covers the
three canonical shapes.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from quicksilver.circuit import Circuit
from quicksilver.einsum import (
    alloc_constant_tensor,
    alloc_input_tensor,
    assert_einsum_equals,
    evaluate_einsum,
    flatten_row_major,
)
from quicksilver.protocol import run as run_protocol


def _prove(circuit, witness, tag):
    t0 = time.time()
    ok = run_protocol(circuit, witness)
    dt = (time.time() - t0) * 1000
    print(f"  {tag}: verifier accepts: {ok}   ({dt:.1f} ms)")
    print(f"  circuit: {circuit.num_inputs} inputs, "
          f"{circuit.num_muls} muls, {circuit.num_asserts} asserts")
    assert ok


def demo_matmul() -> None:
    print("=" * 64)
    print("Demo 1: prove A @ B = C with A and B private")
    print("=" * 64)
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    C = evaluate_einsum("ij,jk->ik", [A, B])
    print(f"  3x3 matmul; public output:")
    for i in range(3):
        print(f"    {C[i*3:(i+1)*3]}")
    c = Circuit()
    Aw = alloc_input_tensor(c, (3, 3))
    Bw = alloc_input_tensor(c, (3, 3))
    assert_einsum_equals(c, "ij,jk->ik", [Aw, Bw], [(3, 3), (3, 3)], C)
    witness = flatten_row_major(A) + flatten_row_major(B)
    _prove(c, witness, "matmul")
    print()


def demo_grandparent_rule() -> None:
    print("=" * 64)
    print("Demo 2: Datalog rule  Grandparent(x, z) :- Parent(x, y), Parent(y, z)")
    print("        compiled directly from the einsum 'xy,yz->xz'")
    print("=" * 64)
    # Family tree: 0 is parent of 1 and 3; 1 of 2; 3 of 2.
    # So 0 is grandparent of 2 (twice, via 1 and via 3).
    P = [
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
    ]
    G = evaluate_einsum("xy,yz->xz", [P, P])
    print(f"  4-person Parent matrix is private to the prover")
    print(f"  public Grandparent matrix:")
    for i in range(4):
        print(f"    {G[i*4:(i+1)*4]}")
    print(f"  (G[0][2] = {G[2]} -> two paths from 0 to 2)")
    c = Circuit()
    Pw1 = alloc_input_tensor(c, (4, 4))
    Pw2 = alloc_input_tensor(c, (4, 4))
    # Same tensor used twice; we force the prover to commit twice and
    # the verifier doesn't know they're the same. To prove they really
    # are the same matrix, add equality assertions.
    assert_einsum_equals(c, "xy,yz->xz", [Pw1, Pw2], [(4, 4), (4, 4)], G)
    for w1, w2 in zip(Pw1, Pw2):
        c.assert_zero(c.sub(w1, w2))
    witness = flatten_row_major(P) + flatten_row_major(P)
    _prove(c, witness, "grandparent")
    print()


def demo_one_step_closure() -> None:
    print("=" * 64)
    print("Demo 3: one step of  Path = step(Edge + einsum('xy,yz->xz', Path, Edge))")
    print("=" * 64)
    # Reachability after exactly two hops on a 5-node graph.
    Edge = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    # After one step, Path = Edge.
    Path1 = Edge
    # After two steps (no step nonlinearity, integer counting):
    # Path2 = Edge + Path1 @ Edge
    PathEdge = evaluate_einsum("xy,yz->xz", [Path1, Edge])
    Edge_flat = flatten_row_major(Edge)
    Path2 = [e + pe for e, pe in zip(Edge_flat, PathEdge)]
    print("  Edge (private to prover):")
    for i in range(5):
        print(f"    {Edge[i]}")
    print("  Public claim: Path after 2 hops =")
    for i in range(5):
        print(f"    {Path2[i*5:(i+1)*5]}")

    c = Circuit()
    Ew1 = alloc_input_tensor(c, (5, 5))
    Ew2 = alloc_input_tensor(c, (5, 5))
    # Compile the matmul piece into the circuit; result is wires
    # accessible by output multi-index.
    from quicksilver.einsum import compile_einsum

    matmul_out = compile_einsum(
        c, "xy,yz->xz", [Ew1, Ew2], [(5, 5), (5, 5)]
    )
    # Add the elementwise sum  Path2[x][z] = Edge[x][z] + matmul[x][z]
    # and assert against public Path2.
    for (x, z), wid in matmul_out.items():
        flat = x * 5 + z
        edge_w = Ew1[flat]  # Edge appears twice; constrain Ew1 == Ew2 elementwise
        sum_w = c.add(edge_w, wid)
        c.assert_eq(sum_w, int(Path2[flat]))
    for w1, w2 in zip(Ew1, Ew2):
        c.assert_zero(c.sub(w1, w2))

    witness = flatten_row_major(Edge) * 2
    _prove(c, witness, "one closure step")
    print()


if __name__ == "__main__":
    demo_matmul()
    demo_grandparent_rule()
    demo_one_step_closure()
    print("All einsum-ZK demos passed.")
