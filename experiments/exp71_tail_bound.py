"""
exp71: Empirical check of the GPT-5.4 Pro tail bound (Erdős #1196).

The April 2026 GPT-5.4 Pro theorem states: for every primitive set A ⊆ N

    sum_{a in A, a > x}  1/(a log a)  <=  1 + O(1/log x).

exp69 tested the *whole-N* version (extremality at primes for x = 2). This
file tests the *tail-restricted* version directly. For each x in a sweep we:

  1. Restrict the TL search to the tail: parameter theta is learned only on
     indices corresponding to integers in [x, N]; the head [2..x-1] is
     pinned with theta = -50 so sigmoid(theta) ≈ 0.
  2. Run cold-start lambda annealing (1e-3 -> 1e2, 3000 steps) — the same
     procedure exp69 PROBE A used.
  3. Round + greedy-reduce to a feasible primitive set A_x ⊆ [x, N].
  4. Compare f(A_x) against:
       - f(primes ∩ [x, N])   — a known feasible set in the tail
       - the conjectured ceiling  1 + C / log x   for several C

We then estimate the smallest constant C_emp such that f(A_x) ≤ 1 + C_emp/log x
holds across the swept x. Per the theorem, C_emp should stabilize at a finite
constant as x grows; concretely we expect f(A_x) << 1 for moderate x because
the tail of any primitive set has thin mass at 1/(n log n) weights.

Two caveats and why they don't kill the experiment:
  - We are bounded by N=2000 (dense divisibility tensor), so "tail" really
    means [x, 2000]. For x close to 2000 the search has almost nothing to
    work with — those rows are reported but should not anchor C_emp.
  - The search is heuristic; it can miss the true tail max. A miss would
    only *understate* C_emp, so a finite C_emp from this experiment is a
    lower bound on the true bound's tightness, not an upper bound on the
    theorem's content.

This is the cleanest "continue" of the exp68-57 line: it exercises the actual
tail-bound shape of the theorem rather than just the whole-N extremality.
"""

import math
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp68_erdos_primitive_sum as e68
import exp69_tl_primitive_search as e69


N = 2000
N_STEPS = 3000
LR = 5e-2
SEED = 0
LAM_LO = 1e-3
LAM_HI = 1e2
PIN_VALUE = -50.0  # theta value on indices outside the tail [x, N]


def head_mask(x, N):
    """Return a boolean mask of length N marking indices i (representing int i+2)
    that are STRICTLY BELOW x — i.e., the head we want to pin to s ≈ 0."""
    n = torch.arange(2, N + 2)
    return n < x


def search_tail(D, w, x, n_steps=N_STEPS, lr=LR, lam_lo=LAM_LO, lam_hi=LAM_HI, seed=SEED):
    torch.manual_seed(seed)
    head = head_mask(x, N)            # True on indices to pin
    theta = torch.randn(N, dtype=torch.float64) * 0.1
    theta[head] = PIN_VALUE
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    log_lo, log_hi = math.log(lam_lo), math.log(lam_hi)
    for t in range(n_steps):
        lam = math.exp(log_lo + (log_hi - log_lo) * (t / max(n_steps - 1, 1)))
        s = torch.sigmoid(theta)
        f = torch.einsum("a,a->", s, w)
        v = torch.einsum("a,ab,b->", s, D, s)
        loss = -f + lam * v
        opt.zero_grad()
        loss.backward()
        # zero out gradient on the pinned head so the projection is stable
        with torch.no_grad():
            theta.grad[head] = 0.0
        opt.step()
        # snap pinned head back if Adam state drifted (it shouldn't with grad=0,
        # but momentum can carry; explicit re-pin is cheap)
        with torch.no_grad():
            theta[head] = PIN_VALUE
    with torch.no_grad():
        s_final = torch.sigmoid(theta).detach()
    return s_final


def f_primes_tail(primes, x, N):
    return sum(1.0 / (p * math.log(p)) for p in primes if x <= p <= N + 1)


def main():
    print(f"=== exp71: tail-bound check  ∑_{{a in A, a>x}} 1/(a log a) <= 1 + C/log x ===")
    print(f"N = {N}   steps = {N_STEPS}   anneal {LAM_LO} -> {LAM_HI}   seed = {SEED}\n")

    D = e68.divisibility_tensor(N).to(torch.float64)
    w = e68.erdos_weights(N)
    primes_N = e68.primes_below(N + 1)

    xs = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

    print(f"  {'x':<6}{'1/log x':<10}{'f(primes∩[x,N])':<18}"
          f"{'f(A_x search)':<16}{'|A_x|':<8}{'prim?':<8}"
          f"{'f_greedy':<12}{'C_emp = (f-1)*log x':<22}")
    rows = []
    for x in xs:
        s = search_tail(D, w, x)
        idx_hi = (s > 0.5).nonzero(as_tuple=True)[0].tolist()
        rounded = [i + 2 for i in idx_hi]
        prim = e68.is_primitive_direct(rounded)
        greedy = e69.make_primitive_greedy(rounded)
        f_search = sum(1.0 / (n * math.log(n)) for n in rounded) if rounded else 0.0
        f_greedy = sum(1.0 / (n * math.log(n)) for n in greedy) if greedy else 0.0
        f_pri = f_primes_tail(primes_N, x, N)
        # Best feasible we know for the tail:
        f_best = max(f_greedy, f_pri)
        log_x = math.log(x)
        c_emp = (f_best - 1.0) * log_x
        rows.append((x, log_x, f_pri, f_search, len(rounded), prim, f_greedy, f_best, c_emp))
        print(f"  {x:<6}{1/log_x:<10.4f}{f_pri:<18.4f}{f_search:<16.4f}"
              f"{len(rounded):<8}{str(prim):<8}{f_greedy:<12.4f}"
              f"{c_emp:<22.4f}")

    # Report largest C_emp across the sweep (excluding x=2 which is the head case).
    c_max = max(r[8] for r in rows if r[0] >= 5)
    print(f"\nlargest C_emp over x in [5, {xs[-1]}] = {c_max:+.4f}")
    print("If C_emp is bounded as x grows, the theorem's 1 + O(1/log x) shape is")
    print("visible. C_emp <= 0 across the sweep means the bound has a comfortable")
    print("margin at this N — every tail-restricted primitive set we found stayed")
    print("strictly below 1, so the leading 1 in the ceiling is not approached.")

    # Spot-check: also print f_best vs. 1 + 1/log x.
    print(f"\n  {'x':<6}{'f_best':<10}{'1+1/log x':<14}{'gap (ceil-f)':<14}")
    for r in rows:
        x, log_x, _, _, _, _, _, f_best, _ = r
        ceiling = 1.0 + 1.0 / log_x
        print(f"  {x:<6}{f_best:<10.4f}{ceiling:<14.4f}{ceiling - f_best:<14.4f}")


if __name__ == "__main__":
    main()
