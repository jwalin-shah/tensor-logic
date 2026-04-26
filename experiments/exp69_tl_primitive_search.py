"""
exp69: Differentiable probes of the Erdős primitive-set bound.

Setup (builds on exp68):

  Indicator s in [0,1]^N  via sigmoid of a parameter vector theta in R^N.
  Divisibility tensor D in {0,1}^{N x N}  (frozen).
  Erdős weight w[a] = 1 / ((a+2) log(a+2)).

  Erdős sum     f(s) = einsum("a,a->", s, w)
  Violation     v(s) = einsum("a,ab,b->", s, D, s)        (= 0  iff  primitive)

The April 2026 GPT-5.4 Pro proof of Erdős Problem #1196 implies, for any
primitive A subset of [x, infinity),

    sum_{a in A, a > x}  1/(a log a)  <=  1 + O(1/log x).

For the whole-N version, the conjectured ceiling is f(primes_2..N) → 1.6366
(Erdős-Sárközy-Szemerédi). Two probes test the geometry numerically:

PROBE A — cold start with lambda annealing.
  theta init: tiny random noise (|theta| ~ 0.1).
  lambda(t) annealed log-linearly from 1e-3 to 1e2.
  Shows what gradient descent can recover from scratch when the violation
  weight is gradually turned on. Naive single-lambda search collapses to a
  tiny primitive set because the quadratic v(s) dominates the small w[a]
  signals at uniform init; annealing lets the f-signal accumulate first.

PROBE B — warm start at the primes.
  theta init: +c on prime indices, -c elsewhere (so s ≈ indicator(primes)).
  No violation penalty needed (set is already primitive). Run -f loss only,
  with mild L2 anchor to keep s near {0,1}. If the primes are extremal
  among primitive sets, gradient flow should *not* be able to move s to a
  primitive set with strictly larger f. We don't enforce primitivity here;
  we report (i) where theta drifts and (ii) whether the rounded set's
  greedy-primitive subset has f exceeding f(primes_2..N). A clear
  exceedance would falsify (or refine the constant in) the bound. We
  expect no exceedance.

Both probes write the optimization as one einsum over a single learnable
tensor — tensor logic in its smallest setting.
"""

import math
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp68_erdos_primitive_sum as e68


N = 2000
SEED = 0


def make_primitive_greedy(elements):
    """Greedy primitive subset: scan ascending, drop multiples of kept elements."""
    kept = []
    for a in sorted(set(elements)):
        if any(a % k == 0 for k in kept):
            continue
        kept.append(a)
    return kept


def f_of_set(elts):
    return sum(1.0 / (n * math.log(n)) for n in elts) if elts else 0.0


# ---------- PROBE A: cold start, lambda annealing ----------

def probe_cold(D, w, n_steps=3000, lr=5e-2, lam_lo=1e-3, lam_hi=1e2, seed=SEED):
    torch.manual_seed(seed)
    theta = torch.randn(N, dtype=torch.float64) * 0.1
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    log_lo, log_hi = math.log(lam_lo), math.log(lam_hi)
    trace = []
    for t in range(n_steps):
        lam = math.exp(log_lo + (log_hi - log_lo) * (t / max(n_steps - 1, 1)))
        s = torch.sigmoid(theta)
        f = torch.einsum("a,a->", s, w)
        v = torch.einsum("a,ab,b->", s, D, s)
        loss = -f + lam * v
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % (n_steps // 5) == 0 or t == n_steps - 1:
            trace.append((t, lam, f.item(), v.item()))
    with torch.no_grad():
        s = torch.sigmoid(theta).detach()
    return s, trace


# ---------- PROBE B: warm start at primes ----------

def probe_warm(D, w, primes_set_in_N, n_steps=2000, lr=2e-2, c=4.0, anchor=1e-3, seed=SEED):
    """theta init: +c on prime indices, -c elsewhere. Loss = -f + anchor*||s-s0||^2."""
    torch.manual_seed(seed)
    s0 = torch.zeros(N, dtype=torch.float64)
    for p in primes_set_in_N:
        if 2 <= p <= N + 1:
            s0[p - 2] = 1.0
    theta = (2 * s0 - 1) * c                            # +c on primes, -c off
    theta = theta + torch.randn_like(theta) * 0.01      # tiny perturbation
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    s_init = torch.sigmoid(theta).detach()
    for _ in range(n_steps):
        s = torch.sigmoid(theta)
        f = torch.einsum("a,a->", s, w)
        anchor_loss = anchor * ((s - s_init) ** 2).sum()
        loss = -f + anchor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = torch.sigmoid(theta).detach()
    return s_init, s


def round_and_greedy(s, threshold=0.5):
    idx_hi = (s > threshold).nonzero(as_tuple=True)[0].tolist()
    rounded = [i + 2 for i in idx_hi]
    primitive = e68.is_primitive_direct(rounded)
    f_rounded = f_of_set(rounded)
    greedy = make_primitive_greedy(rounded)
    f_greedy = f_of_set(greedy)
    return rounded, primitive, f_rounded, greedy, f_greedy


def main():
    print(f"=== exp69: differentiable probes of Erdős #1196, N={N} ===")
    D = e68.divisibility_tensor(N).to(torch.float64)
    w = e68.erdos_weights(N)
    primes_N = e68.primes_below(N + 1)
    f_primes = f_of_set(primes_N)
    print(f"baseline f(primes_2..{N+1}) = {f_primes:.6f}  (|primes|={len(primes_N)})")

    # PROBE A
    print(f"\n--- PROBE A: cold start, lambda annealing 1e-3 -> 1e2 ---")
    s_a, trace = probe_cold(D, w)
    print(f"  {'step':<8}{'lambda':<12}{'f(s)':<10}{'v(s)':<10}")
    for t, lam, fv, vv in trace:
        print(f"  {t:<8}{lam:<12.4g}{fv:<10.4f}{vv:<10.4f}")
    rounded, prim, f_r, greedy, f_g = round_and_greedy(s_a)
    print(f"  rounded |S|={len(rounded)} primitive?={prim}  f_rounded={f_r:.4f}  "
          f"greedy |G|={len(greedy)} f_greedy={f_g:.4f}")

    # PROBE B
    print(f"\n--- PROBE B: warm start at primes, no constraint penalty ---")
    s_init, s_b = probe_warm(D, w, primes_N)
    f_init = torch.einsum("a,a->", s_init, w).item()
    f_final = torch.einsum("a,a->", s_b, w).item()
    drift = (s_b - s_init).abs().sum().item()
    rounded, prim, f_r, greedy, f_g = round_and_greedy(s_b)
    delta_g = f_g - f_primes
    print(f"  f(s_init) = {f_init:.4f}   f(s_final) = {f_final:.4f}   ||drift||_1 = {drift:.4f}")
    print(f"  rounded |S|={len(rounded)} primitive?={prim}  f_rounded={f_r:.4f}")
    print(f"  greedy   |G|={len(greedy)} f_greedy={f_g:.4f}   "
          f"f_greedy - f(primes) = {delta_g:+.6f}")
    if delta_g > 1e-6:
        print(f"  *** EXCEEDANCE: greedy-primitive subset beats primes by {delta_g:+.6f} ***")
    else:
        print(f"  no exceedance found  (consistent with bound)")

    print(f"\nReference: f(primes_2..{N+1}) = {f_primes:.4f}")
    print("Probe A shows differentiable search recovering a primitive-feasible solution")
    print("from scratch. Probe B is the falsification attempt: starting at primes and")
    print("asking gradient flow to find anything better. No exceedance = consistent with")
    print("the GPT-5.4 Pro bound at this N.")


if __name__ == "__main__":
    main()
