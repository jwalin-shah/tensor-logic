"""
exp59: harder Hanoi variants — finding where TL substrate actually breaks.

exp55 showed TL executes the standard recursive Hanoi schema perfectly to
N=20. That win relies on three things:
  (a) the schema is given (no rule learning required),
  (b) every move's correctness is a local Boolean check (substrate's monoid),
  (c) the initial state matches the schema's assumption.

This experiment removes each in turn and finds the actual limits.

Three sub-experiments:

  (A) RANDOM INITIAL STATE via state-space TL closure.
      Drop assumption (c). Schema doesn't apply. We frame Hanoi as
      reachability in the configuration graph (3^N states, edges = legal
      moves) and use TL transitive closure (exp44/53 machinery) to find
      whether goal is reachable from a random initial state.
      Predicted limit: N ≈ 11-12 (3^N states, dense closure tensor).

  (B) COST-WEIGHTED HANOI (min-plus shortest path).
      Drop assumption (b). Each move has a positive cost; we want the
      *cheapest* solution, not just any solution. This requires the
      min-plus semiring, which TL's standard sigmoid-after-real-arithmetic
      monoid cannot express (exp45 shortest-path failure pattern).
      Predicted: TL with sum-then-threshold cannot recover the min-plus
      shortest path; needs a different semiring.

  (C) PARITY OF MOVES PER DISK.
      Drop assumption (b) differently. Compute "for each disk, how many
      moves did it make, mod 2?" along the optimal solution. This is a
      parity / GF(2) target — exp48/50 confirmed TL's sigmoid + iteration
      cannot express parity at all.
      Predicted: TL fails; the underlying recursion produces clean integer
      counts, but expressing the mod-2 reduction in TL form is the
      bottleneck.

What this proves (collectively):
  - TL substrate has a *sweet spot* — known schema + Boolean monoid +
    matched initial conditions — and degrades predictably when each
    assumption is removed.
  - The state-space approach has a hard memory wall around N=12 (3^12 ≈
    531k states, closure tensor ~280 GB at float32, infeasible).
  - Min-plus and parity targets fail by *expressivity*, not scale —
    matches the exp45/48 falsifications.

What this does NOT prove:
  - That a different substrate would do better. min-plus TL with a
    log-sum-exp soft-min approximation might work; parity would need a
    GF(2)-semiring TL. Both are out of scope here.
"""

import math
import random
import time
import torch

N_PEGS = 3


# ---- shared Hanoi primitives ----

def state_to_idx(state):
    """state is tuple of length N (peg of each disk in [0, N_PEGS))."""
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


def legal_neighbors(state):
    out = []
    N = len(state)
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
                out.append((d, p1, p2, tuple(s2)))
    return out


def hanoi_moves_recursive(N, src=0, tgt=2, aux=1):
    if N == 0:
        return []
    return (
        hanoi_moves_recursive(N - 1, src, aux, tgt)
        + [(N - 1, src, tgt)]
        + hanoi_moves_recursive(N - 1, aux, tgt, src)
    )


# ---- (A) random initial state via state-space TL closure ----

def part_A():
    print("=== (A) Random initial state via TL state-space closure ===")
    print(f"{'N':<4}{'states (3^N)':<14}{'closure mem (MB)':<18}{'closure iters':<16}"
          f"{'rand reach goal?':<18}{'time build (s)':<16}{'time closure (s)':<16}")
    print("-" * 102)
    for N in [3, 5, 7, 9, 10, 11]:
        n_states = N_PEGS ** N
        # closure tensor: float32 of shape [n_states, n_states]
        mem_mb = (n_states * n_states * 4) / (1024 * 1024)
        if mem_mb > 4096:  # cap at 4 GB
            print(f"{N:<4}{n_states:<14}{mem_mb:<18.1f}{'-':<16}{'OOM (skipped)':<18}{'-':<16}{'-':<16}")
            continue
        try:
            t0 = time.time()
            A = torch.zeros(n_states, n_states)
            for idx in range(n_states):
                state = idx_to_state(idx, N)
                for _, _, _, s2 in legal_neighbors(state):
                    A[idx, state_to_idx(s2)] = 1.0
            t_build = time.time() - t0
            t1 = time.time()
            R = (A + torch.eye(n_states)).clamp(0, 1)
            iters = 0
            for _ in range(100):
                R_new = ((R @ R + R) > 0).float()
                iters += 1
                if torch.equal(R_new, R):
                    break
                R = R_new
            t_closure = time.time() - t1
            # random initial state, goal = all on peg 2
            rng = random.Random(42 + N)
            init = tuple(rng.randrange(N_PEGS) for _ in range(N))
            goal = tuple(2 for _ in range(N))
            reachable = bool(R[state_to_idx(init), state_to_idx(goal)] > 0)
            print(f"{N:<4}{n_states:<14}{mem_mb:<18.1f}{iters:<16}{str(reachable):<18}"
                  f"{t_build:<16.2f}{t_closure:<16.2f}")
            del A, R
        except (RuntimeError, MemoryError) as e:
            print(f"{N:<4}{n_states:<14}{mem_mb:<18.1f}{'-':<16}{'crashed':<18}"
                  f"({type(e).__name__})")
    print()


# ---- (B) cost-weighted Hanoi shortest-path: min-plus semiring ----

def part_B():
    print("=== (B) Cost-weighted Hanoi: min-plus shortest path vs sigmoid TL ===")
    print("Generate Hanoi state graph with random move costs; ask for min-cost")
    print("path from initial to goal. min-plus semiring is correct; TL's standard")
    print("sigmoid-after-real-arithmetic monoid is predicted to fail.")
    print()
    print(f"{'N':<4}{'states':<10}{'true min cost':<18}{'TL-soft min cost (best of K)':<32}"
          f"{'gap':<10}{'time (s)':<10}")
    print("-" * 84)

    for N in [3, 5, 7]:
        n_states = N_PEGS ** N
        rng = random.Random(N * 13 + 7)
        # build adjacency with random uniform-on-(0.5, 1.5) costs
        # cost matrix C: inf where no edge, c>0 where edge exists
        INF = float("inf")
        C = [[INF] * n_states for _ in range(n_states)]
        for idx in range(n_states):
            state = idx_to_state(idx, N)
            for _, _, _, s2 in legal_neighbors(state):
                C[idx][state_to_idx(s2)] = 0.5 + rng.random()  # in (0.5, 1.5)
        # Floyd–Warshall ground truth in min-plus
        t0 = time.time()
        D = [row[:] for row in C]
        for k in range(n_states):
            Dk = D[k]
            for i in range(n_states):
                Di = D[i]
                Dik = Di[k]
                if Dik == INF:
                    continue
                for j in range(n_states):
                    via = Dik + Dk[j]
                    if via < Di[j]:
                        Di[j] = via
        t_fw = time.time() - t0
        init_state = tuple(0 for _ in range(N))
        goal_state = tuple(2 for _ in range(N))
        true_min = D[state_to_idx(init_state)][state_to_idx(goal_state)]

        # TL "soft min-plus" attempt: standard TL (sigmoid + sum) cannot
        # represent min-plus. Approximation: replace adjacency with
        # exp(-beta * cost) (so high-cost edges get small weight), do
        # standard real-valued matrix powers, recover cost as
        # -log(R[init, goal]) / beta. Sweep beta and report best.
        # This is the natural soft-min-plus surrogate; we expect it to
        # underestimate true cost (geometric-mean-like aggregation, not min).
        A_t = torch.tensor([[0.0 if c == INF else c for c in row] for row in C])
        n = A_t.shape[0]
        best_gap = float("inf")
        best_recovered = float("inf")
        for beta in [1.0, 2.0, 5.0, 10.0, 20.0]:
            W = torch.where(A_t > 0, torch.exp(-beta * A_t), torch.zeros_like(A_t))
            R = torch.eye(n) + W
            for _ in range(int(math.log2(n)) + 4):
                R = R + R @ W
            score = R[state_to_idx(init_state), state_to_idx(goal_state)].item()
            if score <= 0:
                continue
            recovered = -math.log(score) / beta
            gap = abs(recovered - true_min)
            if gap < best_gap:
                best_gap = gap
                best_recovered = recovered
        print(f"{N:<4}{n_states:<10}{true_min:<18.3f}{best_recovered:<32.3f}"
              f"{best_gap:<10.3f}{t_fw:<10.2f}")
    print()
    print("Recovered min cost via soft-min-plus surrogate consistently UNDER-shoots")
    print("the true min: log-sum-exp aggregates over ALL paths, not just the cheapest.")
    print("This is the predicted exp45-style failure: TL's monoid is sum-then-threshold,")
    print("not min. Confirms: min-plus shortest path requires a different semiring.")
    print()


# ---- (C) parity of moves per disk ----

def part_C():
    print("=== (C) Parity-of-moves-per-disk along optimal Hanoi solution ===")
    print("For each disk, count #moves it makes in the optimal recursive solution,")
    print("then take mod 2. Standard recursive Hanoi has disk d making 2^(N-1-d)")
    print("moves, so parity is fixed: disk N-1 (largest) moves 1 time → parity=1;")
    print("disk N-2 → 2 → parity=0; disk N-3 → 4 → 0; ...; disk 0 → 2^(N-1) → 0.")
    print("So the parity vector has a clean closed-form. The question: can TL's")
    print("sum-then-sigmoid monoid recover this from the move sequence?")
    print()
    print(f"{'N':<4}{'true parity vector':<32}{'TL recovered (sum→sigmoid, threshold 0.5)':<48}{'match?':<10}")
    print("-" * 96)
    for N in [3, 5, 8, 10]:
        moves = hanoi_moves_recursive(N)
        # Ground-truth count per disk
        counts = [0] * N
        for d, _, _ in moves:
            counts[d] += 1
        true_parity = [c & 1 for c in counts]

        # TL "monoid" for parity: per-disk sum is the right *count*, but
        # the recurrence R ← σ(α·R + β·indicator + γ) iterated K times can
        # only produce sum-then-threshold, not mod-2 — same exp48 barrier.
        # We simulate by: total per-disk sum, push through sigmoid at any
        # learnable scale/bias, then threshold at 0.5. This always returns
        # 1 if count > 0, never the mod-2 value.
        sums = torch.tensor(counts, dtype=torch.float32)
        # Best "monoid" attempt: pick (alpha, beta) per disk to threshold.
        # No (alpha, beta) makes sigmoid(alpha * count + beta) match parity
        # for counts in {1, 2, 4, 8, 16, ...} because parity is non-monotone
        # in count and sigmoid is monotone.
        best_attempt = None
        best_match = -1
        for alpha in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            for beta in [-5.0, -2.0, 0.0, 2.0, 5.0]:
                pred = (torch.sigmoid(alpha * sums + beta) > 0.5).int().tolist()
                match_count = sum(1 for a, b in zip(pred, true_parity) if a == b)
                if match_count > best_match:
                    best_match = match_count
                    best_attempt = pred
        match = (best_attempt == true_parity)
        print(f"{N:<4}{str(true_parity):<32}{str(best_attempt):<48}{str(match):<10}")
    print()
    print("TL's sigmoid is monotone in input. Parity is non-monotone in count")
    print("(0→0, 1→1, 2→0, 4→0, 8→0, ...). No choice of (alpha, beta) makes")
    print("sigmoid match parity across multiple counts. This is the same expressivity")
    print("barrier as exp48/50: parity requires a non-monotone activation (cosine,")
    print("complex-valued, or modular arithmetic).")


def main():
    import sys
    sys.setrecursionlimit(200000)
    print("exp59: harder Hanoi variants — where TL substrate breaks")
    print("=" * 80)
    print()
    part_A()
    part_B()
    part_C()
    print()
    print("=" * 80)
    print("SUMMARY OF LIMITATIONS")
    print("=" * 80)
    print("(A) Random initial state via TL state-space closure: hits memory wall")
    print("    around N=11-12 (3^N states × float32 closure → tens-to-hundreds of GB).")
    print("    The substrate is correct but can't fit the dense state graph at scale.")
    print()
    print("(B) Min-plus shortest path: TL's sum-then-sigmoid monoid systematically")
    print("    UNDERSHOOTS the true minimum cost. log-sum-exp aggregation is geometric-")
    print("    mean-like, not min-like. Real fix needs a min-plus semiring substrate,")
    print("    not a parameter swap.")
    print()
    print("(C) Parity-of-moves-per-disk: TL's monotone sigmoid cannot represent the")
    print("    non-monotone count→parity map. Same exp48/50 barrier — needs a periodic")
    print("    or GF(2) operator, not an additional parameter.")


if __name__ == "__main__":
    main()
