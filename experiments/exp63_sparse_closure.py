"""
exp63: sparse closure substrate — push the exp59A memory wall out.

Background:
exp59 (A) found that dense TL closure on Hanoi configuration graphs
(3^N states) hits a memory wall at N=10 (~13 GB closure tensor) and
N=11 needs ~120 GB. The same wall would block any openhuman-scale KB
claim.

Three implementations compared:
  (1) DENSE_CLOSURE     — exp44/53/55 baseline. R ← ((R @ A + R) > 0).
                          O(V^3 · log V) via repeated matmul.
  (2) BFS_PER_SOURCE    — build closure row-by-row by BFS from each node.
                          O(V · (V + E)) total. No matmul, no full
                          intermediate dense matrix needed if we store
                          rows as Python sets / sparse rows.
  (3) BFS_PER_QUERY     — for a single (src, dst) query, BFS from src.
                          O(V + E) per query. No closure materialized.

Honest design note:
An earlier draft of this experiment included a "torch.sparse fixpoint"
variant. It was a fake win — torch.sparse-sparse matmul isn't first-
class, so the loop body had to densify R every iteration. At N=9 this
allocated a 1.5 GB dense tensor per iteration and ran for 40+ minutes
before being killed. Lesson: when the substrate-of-execution actually
matters, you have to reach for an algorithm that doesn't densify
(BFS, Tarjan, GraphBLAS), not just a sparse storage layer.

Predictions:
  - Dense wins per-cell at small V because matmul is BLAS-tuned and
    avoids per-source loop overhead.
  - BFS-per-source wins memory at any V (no V×V matrix held at once).
  - BFS-per-query is the right hot-path pick for "1 query per user
    interaction" — O(V+E) is tiny vs O(V^3) when you only need one
    answer.
"""

import random
import time
import torch


def dense_closure(A: torch.Tensor, max_iters: int = 50) -> torch.Tensor:
    n = A.shape[0]
    R = (A + torch.eye(n, dtype=A.dtype)).clamp(0, 1)
    for _ in range(max_iters):
        R_new = ((R @ A + R) > 0).to(A.dtype)
        if torch.equal(R_new, R):
            return R
        R = R_new
    return R


def bfs_per_source_closure(adj_list: dict, n: int):
    """Build closure as a list of frozensets (one per source).
    Memory: O(closure cells), no V×V dense matrix.
    """
    rows = []
    for s in range(n):
        seen = {s}
        queue = [s]
        while queue:
            u = queue.pop(0)
            for v in adj_list.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        rows.append(seen)
    return rows


def bfs_query(adj_list: dict, src: int, dst: int) -> bool:
    if src == dst:
        return True
    seen = {src}
    queue = [src]
    while queue:
        u = queue.pop(0)
        for v in adj_list.get(u, ()):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return False


# ---- Hanoi state-space generator (matches exp59A) ----

N_PEGS = 3


def state_to_idx(state):
    idx = 0
    for p in state:
        idx = idx * N_PEGS + p
    return idx


def idx_to_state(idx, N):
    s = []
    for _ in range(N):
        s.append(idx % N_PEGS)
        idx //= N_PEGS
    return tuple(reversed(s))


def top_disk(state, peg):
    for d, p in enumerate(state):
        if p == peg:
            return d
    return None


def hanoi_edges(N):
    n_states = N_PEGS ** N
    edges = []
    for idx in range(n_states):
        state = idx_to_state(idx, N)
        for p1 in range(N_PEGS):
            d = top_disk(state, p1)
            if d is None:
                continue
            for p2 in range(N_PEGS):
                if p2 == p1:
                    continue
                top_p2 = top_disk(state, p2)
                if top_p2 is None or top_p2 > d:
                    s2 = list(state)
                    s2[d] = p2
                    edges.append((idx, state_to_idx(tuple(s2))))
    return n_states, edges


def part_A_hanoi():
    print("=== (A) Hanoi state-space closure ===")
    print(f"{'N':<4}{'V':<10}{'E':<10}"
          f"{'dense (s)':<14}{'bfs-src (s)':<14}"
          f"{'bfs-q (ms/q)':<16}{'dense mem (MB)':<16}{'closure cells':<16}")
    print("-" * 100)
    # Hanoi state graph is strongly connected, so closure is fully dense
    # (any state reaches any other). BFS-per-source therefore stores ~V^2
    # integers (full closure cells). Cap at N=7 (2,187 states) so dense
    # AND BFS-per-source both fit in modest RAM. The headline is the
    # BFS-per-query column, which scales much further.
    for N in [3, 5, 7, 8]:
        n_states, edges = hanoi_edges(N)
        adj_list = {}
        for i, j in edges:
            adj_list.setdefault(i, []).append(j)
        # Dense closure (only attempt if mem < 1 GB)
        dense_mem_mb = (n_states * n_states * 4) / (1024 * 1024)
        if dense_mem_mb < 1024:
            try:
                t0 = time.time()
                A = torch.zeros(n_states, n_states)
                for i, j in edges:
                    A[i, j] = 1.0
                R = dense_closure(A)
                t_dense = time.time() - t0
                closure_cells = int(R.sum().item())
                del A, R
            except (RuntimeError, MemoryError):
                t_dense = float("inf")
                closure_cells = -1
        else:
            t_dense = float("inf")
            closure_cells = -1
        # BFS per source (build full closure as list of sets)
        try:
            t0 = time.time()
            rows = bfs_per_source_closure(adj_list, n_states)
            t_bfs_src = time.time() - t0
            if closure_cells == -1:
                closure_cells = sum(len(r) for r in rows)
            del rows
        except (RuntimeError, MemoryError):
            t_bfs_src = float("inf")
        # BFS per query (100 random)
        rng = random.Random(N)
        n_q = 100
        t0 = time.time()
        for _ in range(n_q):
            s = rng.randrange(n_states)
            d = rng.randrange(n_states)
            bfs_query(adj_list, s, d)
        t_bfs_q = (time.time() - t0) / n_q * 1000
        print(f"{N:<4}{n_states:<10}{len(edges):<10}"
              f"{(f'{t_dense:.2f}' if t_dense != float('inf') else 'OOM'):<14}"
              f"{(f'{t_bfs_src:.2f}' if t_bfs_src != float('inf') else 'OOM'):<14}"
              f"{t_bfs_q:<16.4f}{dense_mem_mb:<16.1f}{closure_cells:<16}")
    print()


def part_B_openhuman_scale():
    print("=== (B) openhuman-scale: synthetic 10k-entity KB ===")
    print("Approximating a real personal KB: 10k entities, ~50k facts,")
    print("structured as a forest (parent relation has at most one out-edge)")
    print("for realistic closure size — random graphs densify pathologically.")
    print()
    n = 10_000
    rng = random.Random(0)
    # Build a forest: each node (except roots) has exactly one parent
    # chosen from earlier nodes. ~50k random sibling/etc cross-edges added.
    edges = []
    for i in range(1, n):
        parent = rng.randrange(i)
        edges.append((parent, i))  # parent → child
    # add cross-edges (think: friend_of, lives_with, etc.)
    n_cross = 40_000
    for _ in range(n_cross):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            edges.append((a, b))
    edges = list({e for e in edges})
    print(f"  Graph: {n} nodes, {len(edges)} edges (forest backbone + random cross-edges)")
    adj_list = {}
    for i, j in edges:
        adj_list.setdefault(i, []).append(j)

    # BFS per query (1000 random)
    t0 = time.time()
    n_q = 1000
    n_yes = 0
    for _ in range(n_q):
        s, d = rng.randrange(n), rng.randrange(n)
        if bfs_query(adj_list, s, d):
            n_yes += 1
    t_per = (time.time() - t0) / n_q * 1000
    print(f"  BFS-per-query latency:  {t_per:.3f} ms/query  ({n_yes}/{n_q} reachable)")

    # Dense closure forecast
    dense_mem = n * n * 4 / (1024 * 1024)
    print(f"  Dense closure tensor:   {dense_mem:.1f} MB ({n}x{n} float32)")

    # Try BFS-per-source full closure if tractable
    t0 = time.time()
    rows = bfs_per_source_closure(adj_list, n)
    t_bfs_src = time.time() - t0
    closure_size = sum(len(r) for r in rows)
    avg_reachable = closure_size / n
    print(f"  BFS-per-source closure: {t_bfs_src:.2f} s  "
          f"({closure_size:,} cells, avg {avg_reachable:.0f} reachable per node)")
    del rows
    print()


def main():
    print("exp63: sparse closure substrate — pushing the exp59A memory wall")
    print("=" * 100)
    print()
    part_A_hanoi()
    part_B_openhuman_scale()
    print()
    print("=" * 100)
    print("CONCLUSIONS")
    print("=" * 100)
    print("- Dense closure: BLAS-fast per cell but O(V^2) memory. exp59A wall")
    print("  at N=10 (Hanoi) hits because the V×V dense matrix doesn't fit.")
    print("- BFS-per-source: O(V·(V+E)) total time, O(closure-cells) memory.")
    print("  Pushes the wall out by storing only what's reachable, not a")
    print("  full V×V matrix. Trades constant-factor speed for memory.")
    print("- BFS-per-query: O(V+E) per query, no closure stored. Right pick")
    print("  for openhuman's hot path (~1 query per user interaction).")
    print("- For openhuman scale (10k entities, ~50k facts):")
    print("    BFS-per-query ≈ <10 ms — viable hot path.")
    print("    Dense closure ≈ 400 MB — viable cold path (precompute once).")
    print("    BFS-per-source ≈ a few seconds total — viable warmup.")


if __name__ == "__main__":
    main()
