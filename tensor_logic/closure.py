from collections import deque

import torch


def dense_closure(A: torch.Tensor, max_iters: int | None = None, reflexive: bool = True) -> torch.Tensor:
    """Boolean transitive closure via thresholded matrix recurrence."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D tensor")
    n = A.shape[0]
    R = (A > 0).to(A.dtype)
    if reflexive:
        R = (R + torch.eye(n, dtype=A.dtype, device=A.device)).clamp(0, 1)
    max_iters = max_iters if max_iters is not None else n + 1
    for _ in range(max_iters):
        R_new = ((R @ R + R) > 0).to(A.dtype)
        if torch.equal(R_new, R):
            return R
        R = R_new
    return R


def bfs_per_source_closure(adj_list: dict[int, list[int] | tuple[int, ...]], n: int, reflexive: bool = True):
    """Build closure rows as sets without materializing an n x n tensor."""
    rows = []
    for src in range(n):
        seen = {src} if reflexive else set()
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj_list.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        rows.append(seen)
    return rows


def bfs_query(adj_list: dict[int, list[int] | tuple[int, ...]], src: int, dst: int, reflexive: bool = True) -> bool:
    """Reachability query over an adjacency list."""
    if reflexive and src == dst:
        return True
    seen = {src}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for v in adj_list.get(u, ()):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return False
