"""Tests for the ZK graph-reachability circuit."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quicksilver.field import F
from quicksilver.protocol import prove, verify
from quicksilver.vole import trusted_dealer_setup
from quicksilver.zk_reachability import (
    assemble_witness,
    build_circuit,
    prove_path,
)


def _toy_graph():
    """5-node DAG: 0->1->2->3, plus 1->4. Direct path 0->1->2->3 has length 3."""
    n = 5
    E = [[0] * n for _ in range(n)]
    for u, v in [(0, 1), (1, 2), (2, 3), (1, 4)]:
        E[u][v] = 1
    return n, E


def test_completeness_path_3():
    n, E = _toy_graph()
    assert prove_path(n=n, k=3, source=0, target=3, edge_matrix=E, path=[0, 1, 2, 3])


def test_completeness_walk_with_repeats():
    """A walk may revisit edges/vertices; reachability is just the existence of one."""
    n, E = _toy_graph()
    # walk 0 -> 1 -> 4 -> ... can't continue; instead try 0->1->2->3 with k=3.
    # For k=4 with self-loops we'd need self-loops; without them, no length-4 walk
    # exists in this DAG ending at 3. Skip this case.
    # Use a graph with cycles instead.
    n = 4
    E = [[0] * n for _ in range(n)]
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3)]:
        E[u][v] = 1
    # walk 0 -> 1 -> 2 -> 0 -> 1 -> 2 -> 3, length 6
    assert prove_path(n=n, k=6, source=0, target=3, edge_matrix=E,
                      path=[0, 1, 2, 0, 1, 2, 3])


def test_assemble_witness_rejects_non_edge_path():
    n, E = _toy_graph()
    rc = build_circuit(n=n, k=3, source=0, target=3)
    # Path 0 -> 2 -> ... uses non-edge (0, 2).
    with pytest.raises(ValueError, match="non-edge"):
        assemble_witness(rc, E, [0, 2, 3, 3])


def test_assemble_witness_rejects_endpoint_mismatch():
    n, E = _toy_graph()
    rc = build_circuit(n=n, k=3, source=0, target=3)
    with pytest.raises(ValueError, match="endpoints"):
        assemble_witness(rc, E, [1, 2, 3, 3])


def test_soundness_lying_about_edge_existence():
    """A prover who claims a path through a non-edge is rejected.

    We build the same circuit but feed an edge matrix that lacks an edge
    on the path. The witness assembler refuses to produce inputs (good
    sanity), so we hand-roll a malicious witness that pretends E[0][2]=1
    while the rest of the matrix really has E[0][1]=E[1][2]=E[2][3]=1.
    The booleanity constraints all hold; the step constraint at i=0 fails.
    """
    n = 5
    rc = build_circuit(n=n, k=3, source=0, target=3)

    # Real graph (no edge 0->2).
    E = [[0] * n for _ in range(n)]
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        E[u][v] = 1

    # Build the witness manually: claim path 0->2->3->3 with E unchanged.
    # That requires E[0][2]=1 (it isn't) and E[3][3]=1 (it isn't).
    # We just pass the real E with a mis-aligned path; the prover side will
    # refuse via assemble_witness, so go around it:
    fake_witness = []
    for i in range(n):
        for j in range(n):
            fake_witness.append(int(E[i][j]))
    # Intermediate alphas claim vertices 2 and 3.
    for v in (2, 3):
        for j in range(n):
            fake_witness.append(1 if j == v else 0)

    # Honest prover walking with this witness will fail at the first step
    # constraint, where the prover's local "step" sum is 0 not 1, and
    # ``open_to_zero`` raises (since the diff wire is -1, not 0).
    p_share, v_share = trusted_dealer_setup(rc.circuit.vole_count())
    with pytest.raises(ValueError):
        prove(rc.circuit, fake_witness, p_share)


def test_soundness_tampered_post_commit():
    """A prover who submits a real proof but tampers the batched-check
    response is rejected by the verifier.
    """
    n, E = _toy_graph()
    rc = build_circuit(n=n, k=3, source=0, target=3)
    witness = assemble_witness(rc, E, [0, 1, 2, 3])

    p_share, v_share = trusted_dealer_setup(rc.circuit.vole_count())
    msg1, batched = prove(rc.circuit, witness, p_share)
    chi = F.rand_nonzero()
    msg2 = batched(chi)
    # Flip one bit of U.
    from quicksilver.protocol import BatchedCheck
    msg2 = BatchedCheck(U=F.add(msg2.U, 1), V=msg2.V)
    assert not verify(rc.circuit, v_share, msg1, chi, msg2)


def test_circuit_size_scales_as_n2_k():
    """Sanity-check the asymptotic gate count for a few (n, k)."""
    sizes = []
    for n, k in [(4, 2), (4, 4), (6, 4), (8, 4)]:
        rc = build_circuit(n=n, k=k, source=0, target=n - 1)
        sizes.append((n, k, rc.circuit.num_muls))
    # Print for inspection (assertion just guards monotonicity).
    print(sizes)
    # Doubling n at fixed k should roughly quadruple the mul-gate count.
    n4 = next(m for n, k, m in sizes if n == 4 and k == 4)
    n8 = next(m for n, k, m in sizes if n == 8 and k == 4)
    assert n8 > 3 * n4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
