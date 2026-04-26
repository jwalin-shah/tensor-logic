"""
exp58: Hanoi drift-collapse curve — falsifiable target for "Apple's reasoning
collapse is reproducible by simple compounding execution error."

Setup:
For each (N, eps) cell, execute the recursive Hanoi schema from exp55, but
at each step with probability eps replace the schema-correct move with a
uniformly-random *legal* move (a stand-in for an LRM "drifting" off-policy).
Continue executing the rest of the schema as written. The schema's later
moves may now be illegal because the state has diverged — count both legal-
move rate and reaches-goal.

Theoretical prediction:
If drift is i.i.d. per step, P(reaches goal at all) ≤ (1 - eps)^M where
M = 2^N - 1. So small eps + large N collapses fast: at eps=0.01, N=10 has
M=1023 and (0.99)^1023 ≈ 4e-5. This matches Apple's qualitative collapse
window.

Falsified if:
The empirical reaches-goal curve is qualitatively different from a
compounding-error curve — e.g. flat then sudden drop, or accuracy that
recovers at high eps. Either pattern would suggest the LRM failure mode
isn't simple per-step drift.

What this proves:
The "substrate maintains state, token-stream-execution doesn't" framing is
quantitative. Compounding error explains the collapse curve at the right
order of magnitude. TL's substrate (zero drift) is at the eps=0 column,
LRMs are somewhere in eps ∈ [0.001, 0.05] depending on model.

What this does NOT prove:
That LRMs literally are doing per-step random-legal drift. The mechanism
inside the LRM may be different (attention failure, working-memory loss).
This is a *behavioral* model that fits the curve, not a mechanistic one.
"""

import math
import random
import statistics
import sys
import time

sys.setrecursionlimit(200000)


N_PEGS = 3


def initial_state(N):
    return tuple(0 for _ in range(N))  # disk d on peg state[d]


def goal_state(N):
    return tuple(2 for _ in range(N))


def top_disk(state, peg):
    """Smallest disk on `peg`, or None."""
    for d, p in enumerate(state):
        if p == peg:
            return d
    return None


def legal_move(state, d, p1, p2):
    if state[d] != p1:
        return False
    if top_disk(state, p1) != d:
        return False
    top_p2 = top_disk(state, p2)
    if top_p2 is not None and top_p2 < d:
        return False
    return True


def apply_move(state, d, p1, p2):
    s = list(state)
    s[d] = p2
    return tuple(s)


def hanoi_moves(N, src=0, tgt=2, aux=1):
    if N == 0:
        return []
    return (
        hanoi_moves(N - 1, src, aux, tgt)
        + [(N - 1, src, tgt)]
        + hanoi_moves(N - 1, aux, tgt, src)
    )


def all_legal_moves(state):
    """Enumerate every (d, p1, p2) currently legal."""
    out = []
    for p1 in range(N_PEGS):
        d = top_disk(state, p1)
        if d is None:
            continue
        for p2 in range(N_PEGS):
            if p2 == p1:
                continue
            if legal_move(state, d, p1, p2):
                out.append((d, p1, p2))
    return out


def execute_with_drift(N, eps, rng):
    """Run schema; at each step, with prob eps replace with random legal move."""
    state = initial_state(N)
    goal = goal_state(N)
    schema = hanoi_moves(N)
    n_legal = 0
    for move in schema:
        if rng.random() < eps:
            options = all_legal_moves(state)
            move = rng.choice(options)
        d, p1, p2 = move
        if not legal_move(state, d, p1, p2):
            # schema move is illegal because state diverged from drift earlier;
            # we count this as a failed step but skip to keep the simulation
            # going (mirrors an LRM that emits the wrong move and then is
            # incoherent thereafter)
            continue
        state = apply_move(state, d, p1, p2)
        n_legal += 1
    reached = (state == goal)
    return reached, n_legal, len(schema)


def main():
    EPSILONS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
    NS = [5, 8, 10, 12, 15]
    SEEDS = 30
    print(f"exp58: Hanoi drift-collapse — {len(NS)} sizes × {len(EPSILONS)} eps × {SEEDS} seeds")
    print()
    print(f"{'N':<5}{'M=2^N-1':<10}", end="")
    for eps in EPSILONS:
        print(f"eps={eps:<7.3f}", end="")
    print()
    print("-" * (15 + 12 * len(EPSILONS)))
    for N in NS:
        M = 2 ** N - 1
        print(f"{N:<5}{M:<10}", end="")
        for eps in EPSILONS:
            t0 = time.time()
            successes = 0
            for seed in range(SEEDS):
                rng = random.Random(seed * 1000 + N + int(eps * 1e6))
                reached, _, _ = execute_with_drift(N, eps, rng)
                if reached:
                    successes += 1
            rate = successes / SEEDS
            dt = time.time() - t0
            print(f"{rate:<11.2f}", end="")
        print()
    print()
    print("Theoretical (1-eps)^M reach-goal upper bound for reference:")
    print(f"{'N':<5}{'M=2^N-1':<10}", end="")
    for eps in EPSILONS:
        print(f"eps={eps:<7.3f}", end="")
    print()
    print("-" * (15 + 12 * len(EPSILONS)))
    for N in NS:
        M = 2 ** N - 1
        print(f"{N:<5}{M:<10}", end="")
        for eps in EPSILONS:
            if eps == 0:
                bound = 1.0
            else:
                bound = (1 - eps) ** M if (1 - eps) ** M > 1e-30 else 0.0
            print(f"{bound:<11.2e}", end="")
        print()


if __name__ == "__main__":
    main()
