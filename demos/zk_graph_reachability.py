"""Zero-knowledge proof of graph reachability via the tensor-logic einsum.

Bridges ``transitive_closure.py`` (the boolean closure recurrence
``Path = step(Edge + Path @ Edge)``) and ``quicksilver/`` (VOLE-based
ZK proofs). The same einsum that defines the transitive closure
becomes the constraint that the prover satisfies inside QuickSilver.

Statement proved:

    "I know an n-vertex graph G and a length-k walk through G from
     public source s to public target t."

Verifier learns nothing about G or about the intermediate vertices of
the walk beyond what (n, k, s, t) already imply.

Run from the repo root:

    python demos/zk_graph_reachability.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from collections import deque
from typing import List, Optional, Tuple

from quicksilver.field import F
from quicksilver.protocol import prove, verify, run as run_protocol
from quicksilver.vole import trusted_dealer_setup
from quicksilver.zk_reachability import assemble_witness, build_circuit


def bfs_path(E: List[List[int]], source: int, target: int) -> Optional[List[int]]:
    """Return the shortest s->t path in E, or None if unreachable."""
    n = len(E)
    parent = {source: None}
    q = deque([source])
    while q:
        u = q.popleft()
        if u == target:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return list(reversed(path))
        for v in range(n):
            if E[u][v] and v not in parent:
                parent[v] = u
                q.append(v)
    return None


def random_dag(n: int, seed: int = 0) -> List[List[int]]:
    """Build a random small DAG with n vertices."""
    import random

    rng = random.Random(seed)
    E = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            # Slightly denser than random so reachability is interesting.
            if rng.random() < 0.45:
                E[u][v] = 1
    return E


def show_graph(E: List[List[int]], path: List[int]) -> None:
    n = len(E)
    print(f"  graph (private to prover): {n} vertices, "
          f"{sum(sum(r) for r in E)} edges")
    edges_on_path = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    print("    adjacency matrix (* = edge on the chosen walk):")
    for i in range(n):
        row = ""
        for j in range(n):
            if E[i][j]:
                row += " *" if (i, j) in edges_on_path else " 1"
            else:
                row += " ."
        print(f"      {i}: {row}")
    print(f"  walk:    {' -> '.join(map(str, path))}")


def demo_basic() -> None:
    print("=" * 64)
    print("Demo: ZK reachability on an 8-vertex random DAG")
    print("=" * 64)
    n = 8
    E = random_dag(n, seed=1)
    source, target = 0, n - 1
    path = bfs_path(E, source, target)
    if path is None:
        # Add a guaranteed path so the demo is reproducible.
        for u, v in zip(range(n - 1), range(1, n)):
            E[u][v] = 1
        path = list(range(n))
    k = len(path) - 1
    show_graph(E, path)

    rc = build_circuit(n=n, k=k, source=source, target=target)
    c = rc.circuit
    print(f"\n  circuit: {c.num_inputs} private inputs, "
          f"{c.num_muls} multiplication gates, {c.num_asserts} assertions")
    print(f"  VOLE elements consumed: {c.vole_count()}")

    witness = assemble_witness(rc, E, path)
    t0 = time.time()
    ok = run_protocol(c, witness)
    dt = (time.time() - t0) * 1000
    print(f"\n  verifier accepts: {ok}   (in {dt:.1f} ms)\n")
    assert ok


def demo_walk_through_cycle() -> None:
    print("=" * 64)
    print("Demo: walk that revisits vertices through a cycle")
    print("=" * 64)
    # 4-vertex cycle with a tail to a target.
    n = 4
    E = [[0] * n for _ in range(n)]
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3)]:
        E[u][v] = 1
    source, target = 0, 3
    path = [0, 1, 2, 0, 1, 2, 3]
    k = len(path) - 1
    show_graph(E, path)

    rc = build_circuit(n=n, k=k, source=source, target=target)
    c = rc.circuit
    print(f"\n  circuit: {c.num_muls} mul gates, {c.num_asserts} asserts, "
          f"{c.vole_count()} VOLE elements")

    witness = assemble_witness(rc, E, path)
    t0 = time.time()
    ok = run_protocol(c, witness)
    print(f"  verifier accepts: {ok}   (in {(time.time() - t0)*1000:.1f} ms)\n")
    assert ok


def demo_soundness_tampering() -> None:
    print("=" * 64)
    print("Demo: cheating prover who tampers the batched check is rejected")
    print("=" * 64)
    n = 6
    E = [[0] * n for _ in range(n)]
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        E[u][v] = 1
    rc = build_circuit(n=n, k=5, source=0, target=5)
    witness = assemble_witness(rc, E, [0, 1, 2, 3, 4, 5])

    p_share, v_share = trusted_dealer_setup(rc.circuit.vole_count())
    msg1, batched = prove(rc.circuit, witness, p_share)
    chi = F.rand_nonzero()
    msg2 = batched(chi)

    # Honest path: verifier accepts.
    print(f"  honest run accepts: {verify(rc.circuit, v_share, msg1, chi, msg2)}")

    # Tampered path: flip one bit of the U component of the batched check.
    from quicksilver.protocol import BatchedCheck

    bad = BatchedCheck(U=F.add(msg2.U, 1), V=msg2.V)
    print(f"  tampered run accepts: {verify(rc.circuit, v_share, msg1, chi, bad)}")
    print()


def demo_size_scaling() -> None:
    print("=" * 64)
    print("Demo: circuit-size scaling (mul gates ~ n^2 * k)")
    print("=" * 64)
    print(f"  {'n':>4} {'k':>4} {'inputs':>8} {'muls':>8} {'time(ms)':>10}")
    for n, k in [(4, 3), (8, 4), (12, 5), (16, 5)]:
        # Build a path graph 0->1->...->n-1 so we always have a walk of length k
        # for k <= n-1.
        E = [[0] * n for _ in range(n)]
        for i in range(n - 1):
            E[i][i + 1] = 1
        # walk uses the first k+1 vertices.
        path = list(range(k + 1))
        rc = build_circuit(n=n, k=k, source=0, target=k)
        witness = assemble_witness(rc, E, path)
        t0 = time.time()
        ok = run_protocol(rc.circuit, witness)
        dt = (time.time() - t0) * 1000
        assert ok
        print(f"  {n:>4} {k:>4} {rc.circuit.num_inputs:>8} "
              f"{rc.circuit.num_muls:>8} {dt:>10.1f}")
    print()


if __name__ == "__main__":
    demo_basic()
    demo_walk_through_cycle()
    demo_soundness_tampering()
    demo_size_scaling()
    print("All ZK reachability demos passed.")
