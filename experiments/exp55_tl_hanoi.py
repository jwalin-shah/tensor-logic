"""
exp55: Tower of Hanoi as a TL substrate — execute the algorithm at the N
where Apple's "Illusion of Thinking" reports LRM collapse.

Background:
Shojaee et al. (Apple, Jun 2025) find that frontier reasoning models
(o3-mini, DeepSeek-R1, Claude thinking) collapse to ~0% accuracy on Tower
of Hanoi beyond N≈8-10, even when given the explicit recursive algorithm
in the prompt. The Lawsen (Jun 2025) rebuttal attributes most of this to
context-window truncation of the 2^N-1 move enumeration: models write the
recursive function correctly when asked, they just can't *enumerate* from
it. Either way, the failure is at the execution interface, not the rule.

Hypothesis:
A TL-style substrate — where state is a tensor `S[disk, peg]`, the
legal-move predicate is a min-reduction, and a move is a deterministic
tensor update — executes the recursive Hanoi algorithm reliably to N=20
(2^20 − 1 = 1,048,575 moves). Zero learned parameters; substrate is
constitutionally deterministic. This separates two questions Apple
conflates:
  (1) Does the system *know* the algorithm? Both LRMs and TL: yes.
  (2) Does the system have a substrate to *execute* it without drift?
      LRM-as-enumerator: no, by construction (token-by-token, lossy).
      TL: yes, by construction (typed state + deterministic update).

This is the small, clean version of the OPENHUMAN_TL_MEMO claim that the
SLM should propose rules and the TL substrate should execute them — here
the "rule" is the 6-line recursive Hanoi schema and the "substrate" is
the tensor-state + tensor-update.

What this proves:
  - "Reasoning collapse" at large N on Hanoi is a substrate-of-execution
    problem, not a reasoning problem.
  - TL's typed-state recurrence is the right substrate for problems with
    deterministic, long-horizon execution where state must be maintained
    across many steps.

What this does NOT prove:
  - That LRMs cannot do this internally — they can write the function
    (Lawsen). The interesting research is the LM+TL composition.
  - Anything about *learning* the rule. The rule is given; only execution
    is tested. Learning the rule from demonstrations is a separate
    question (see exp44/53 for the closure analog).
"""

import time
import torch

NS = [3, 5, 8, 10, 12, 15, 18, 20]
N_PEGS = 3


def initial_state(N):
    """All N disks on peg 0. S[d, p] = 1 iff disk d is on peg p."""
    S = torch.zeros(N, N_PEGS, dtype=torch.int8)
    S[:, 0] = 1
    return S


def goal_state(N):
    """All N disks on peg 2."""
    G = torch.zeros(N, N_PEGS, dtype=torch.int8)
    G[:, 2] = 1
    return G


def top_disk(S, peg):
    """Smallest disk on `peg` (= the disk on top of the stack), or None.

    Pure TL primitive: arg-min over disk indices where S[:, peg] == 1.
    Equivalent to einsum('dp->d', S * mask) reduction with smallest-index
    selection. Implemented compactly via .nonzero() for clarity.
    """
    occupied = S[:, peg].nonzero(as_tuple=True)[0]
    if occupied.numel() == 0:
        return None
    return occupied.min().item()


def legal_move(S, d, p1, p2):
    """Move disk d from p1 to p2 is legal iff:
       (a) S[d, p1] == 1 (disk is actually on p1),
       (b) d is the top (smallest) disk on p1,
       (c) p2 is empty OR p2's top is larger than d.
    """
    if S[d, p1] != 1:
        return False
    if top_disk(S, p1) != d:
        return False
    top_p2 = top_disk(S, p2)
    if top_p2 is not None and top_p2 < d:
        return False
    return True


def apply_move(S, d, p1, p2):
    """Tensor update: S' = S - e(d, p1) + e(d, p2)."""
    S2 = S.clone()
    S2[d, p1] = 0
    S2[d, p2] = 1
    return S2


def hanoi_moves(N, src=0, tgt=2, aux=1):
    """Standard recursive schema. Returns flat list of (disk, from, to)."""
    if N == 0:
        return []
    return (
        hanoi_moves(N - 1, src, aux, tgt)
        + [(N - 1, src, tgt)]
        + hanoi_moves(N - 1, aux, tgt, src)
    )


def execute(N):
    """Generate move sequence and step the TL state through every move."""
    S = initial_state(N)
    G = goal_state(N)
    moves = hanoi_moves(N)
    n_legal = 0
    for d, p1, p2 in moves:
        if not legal_move(S, d, p1, p2):
            return False, n_legal, len(moves)
        S = apply_move(S, d, p1, p2)
        n_legal += 1
    reached = bool(torch.equal(S, G))
    return reached, n_legal, len(moves)


def main():
    import sys
    sys.setrecursionlimit(100000)
    print("exp55: TL substrate executes Tower of Hanoi where Apple's LRMs collapse")
    print("Apple (Shojaee et al. 2025): LRMs collapse to ~0 accuracy at N≈8-10")
    print()
    print(f"{'N':<5}{'optimal moves':<16}{'TL moves':<12}{'all legal?':<14}"
          f"{'reached goal?':<16}{'time (s)':<10}")
    print("-" * 72)
    rows = []
    for N in NS:
        t0 = time.time()
        ok_goal, n_legal, n_total = execute(N)
        dt = time.time() - t0
        opt = 2 ** N - 1
        legal_ok = (n_legal == n_total)
        rows.append((N, opt, n_total, legal_ok, ok_goal, dt))
        print(f"{N:<5}{opt:<16}{n_total:<12}"
              f"{('YES' if legal_ok else f'NO {n_legal}/{n_total}'):<14}"
              f"{('YES' if ok_goal else 'NO'):<16}"
              f"{dt:<10.3f}")
    print()
    all_perfect = all(r[3] and r[4] and r[2] == r[1] for r in rows)
    if all_perfect:
        print(f"RESULT: TL substrate executed all {len(NS)} sizes (max N={max(NS)}, "
              f"{2**max(NS)-1:,} moves) with 100% legal-move + goal-reached.")
    else:
        print("RESULT: at least one size failed — see table above.")
    print()
    print("Apple's curve for reference: o3-mini-high accuracy on Tower of Hanoi")
    print("crosses 0% near N=10. DeepSeek-R1 collapses earlier. Token usage")
    print("DROPS as N grows past the collapse point — the LRM gives up rather")
    print("than continues reasoning. TL has no analogous failure mode at this")
    print("scale because state is maintained in the substrate, not in the")
    print("token stream.")


if __name__ == "__main__":
    main()
