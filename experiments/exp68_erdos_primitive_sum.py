"""
exp68: TL encoding of primitive sets and the Erdős sum.

The April 2026 GPT-5.4 Pro proof of Erdős Problem #1196 establishes

    sum_{a in A, a > x}  1/(a log a)  <=  1 + O(1/log x)

for every primitive set A subset of N. The proof uses a sub-Markov-chain with
visit mass proportional to 1/(n log n). That whole object is a tensor
recurrence — exactly the substrate this repo encodes (Domingos, Tensor Logic).

As the foundation for the rest of the primitive-set arc (exp69-70), this file
puts the static machinery into TL form:

  - Divisibility tensor  D[a,b] = 1  iff  a | b and a != b
  - "Is primitive" predicate:  einsum("a,ab,b->", S, D, S) == 0 for indicator S
  - Erdős weight  w[a] = 1 / (a log a)
  - Erdős sum     f(S) = einsum("a,a->", S, w)

Three sanity checks:

  (1) The einsum primitive predicate agrees with a direct two-loop check on
      half a dozen canonical sets at N=2000.

  (2) f(primes_2..M) increases monotonically toward the Erdős-Sárközy-Szemerédi
      constant ~ 1.6366 as M grows; reported at M in {1e3, 1e4, 1e5, 1e6}.

  (3) Other primitive sets we can construct (single primes, powers of 2,
      semiprimes p*q, squarefree odd numbers, etc.) all stay strictly below
      f(primes) — the primes are extremal among the things we tried.

This is not a proof of anything; it just verifies that the TL substrate
agrees with classical number-theoretic computation, so that exp69's
differentiable counter-example searcher is meaningful.
"""

import math
import torch


N_DENSE = 2000        # size of dense divisibility tensor for einsum checks
M_LARGE = [10**3, 10**4, 10**5, 10**6]  # prime-sum convergence range


# ---------- TL building blocks ----------

def divisibility_tensor(N):
    """D[a,b] = 1 iff (a+2) divides (b+2) and a != b. Index i represents int i+2."""
    D = torch.zeros(N, N)
    for a in range(2, N + 2):
        for b in range(2 * a, N + 2, a):
            D[a - 2, b - 2] = 1.0
    return D


def erdos_weights(N):
    """w[i] = 1 / ((i+2) log(i+2))."""
    n = torch.arange(2, N + 2, dtype=torch.float64)
    return 1.0 / (n * torch.log(n))


def indicator(elements, N):
    s = torch.zeros(N, dtype=torch.float64)
    for x in elements:
        if 2 <= x <= N + 1:
            s[x - 2] = 1.0
    return s


def is_primitive_einsum(s, D):
    return torch.einsum("a,ab,b->", s, D.to(s.dtype), s).item() <= 0.5


def is_primitive_direct(elements):
    elts = sorted(set(elements))
    for i, a in enumerate(elts):
        for b in elts[i + 1:]:
            if b % a == 0:
                return False
    return True


def erdos_sum_einsum(s, w):
    return torch.einsum("a,a->", s, w).item()


# ---------- canonical primitive sets ----------

def primes_below(M):
    sieve = bytearray([1]) * (M + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(M ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, M + 1, i):
                sieve[j] = 0
    return [i for i in range(2, M + 1) if sieve[i]]


def powers_of_2(M):
    out, k = [], 2
    while k <= M:
        out.append(k)
        k *= 2
    return out


def semiprimes_pq(M, primes):
    """Numbers p*q with p<=q both prime, p*q <= M. Primitive: no element divides another."""
    out = []
    for i, p in enumerate(primes):
        if p * p > M:
            break
        for q in primes[i:]:
            if p * q > M:
                break
            out.append(p * q)
    return sorted(set(out))


def squarefree_odd_in_band(lo, hi, primes):
    """Odd squarefree integers in [lo, hi). Not primitive in general — keep only those
    none of whose proper divisors lie in the same band."""
    pset = set(primes)
    out = []
    for n in range(max(lo, 3), hi, 2):
        m, sqfree = n, True
        for p in primes:
            if p * p > n:
                break
            if m % (p * p) == 0:
                sqfree = False
                break
        if sqfree:
            out.append(n)
    # filter to a primitive subset by the standard band trick: keep only those with
    # no proper divisor also in the kept list.
    kept_set = set()
    out_sorted = sorted(out)
    for n in out_sorted:
        ok = True
        for d in kept_set:
            if d != n and n % d == 0:
                ok = False
                break
        if ok:
            kept_set.add(n)
    return sorted(kept_set)


# ---------- main ----------

def main():
    print(f"=== exp68: TL encoding of primitive sets ===")
    print(f"dense divisibility tensor: N={N_DENSE} (memory ~{N_DENSE*N_DENSE*4/1e6:.1f} MB)")

    # (1) primitive predicate via einsum vs direct
    print("\n--- (1) einsum primitive predicate vs direct loop, N=2000 ---")
    D = divisibility_tensor(N_DENSE)
    primes_small = primes_below(N_DENSE + 1)
    cases = [
        ("primes <= 2001", primes_small),
        ("powers of 2 <= 2001", powers_of_2(N_DENSE + 1)),
        ("semiprimes p*q <= 2001", semiprimes_pq(N_DENSE + 1, primes_small)),
        ("not-primitive: {2, 4, 6}", [2, 4, 6]),
        ("not-primitive: {3, 9, 27}", [3, 9, 27]),
        ("primitive: {6, 10, 15}", [6, 10, 15]),
    ]
    print(f"  {'set':<32}{'|S|':<8}{'einsum':<10}{'direct':<10}{'match':<6}")
    all_match = True
    for name, elts in cases:
        s = indicator(elts, N_DENSE)
        e = is_primitive_einsum(s, D)
        d = is_primitive_direct(elts)
        match = "yes" if e == d else "NO"
        all_match = all_match and (e == d)
        print(f"  {name:<32}{len(elts):<8}{str(e):<10}{str(d):<10}{match:<6}")
    print(f"  all-match = {all_match}")

    # (2) f(primes_2..M) convergence
    print("\n--- (2) f(primes_2..M) convergence to Erdős-Sárközy-Szemerédi constant ---")
    print(f"  {'M':<10}{'#primes':<10}{'f(primes)':<14}")
    for M in M_LARGE:
        ps = primes_below(M)
        fM = sum(1.0 / (p * math.log(p)) for p in ps)
        print(f"  {M:<10}{len(ps):<10}{fM:<14.6f}")
    print("  (target constant ~ 1.6366 in the limit)")

    # (3) f compared across canonical primitive sets at N=10000
    print("\n--- (3) Erdős sum across canonical primitive sets, M=10000 ---")
    M = 10_000
    primes_M = primes_below(M)
    primitive_sets = [
        ("primes",                       primes_M),
        ("powers of 2",                  powers_of_2(M)),
        ("powers of 3",                  [3 ** k for k in range(1, int(math.log(M, 3)) + 1)]),
        ("semiprimes p*q",               semiprimes_pq(M, primes_M)),
        ("primes in [100, M]",           [p for p in primes_M if p >= 100]),
        ("primes in [1000, M]",          [p for p in primes_M if p >= 1000]),
        ("squarefree odd primitive",     squarefree_odd_in_band(3, M, primes_M)),
    ]
    print(f"  {'set':<32}{'|S|':<8}{'primitive?':<12}{'f(S)':<10}")
    for name, elts in primitive_sets:
        elts = [e for e in elts if 2 <= e <= M]
        f_val = sum(1.0 / (e * math.log(e)) for e in elts)
        prim = is_primitive_direct(elts)
        print(f"  {name:<32}{len(elts):<8}{str(prim):<12}{f_val:<10.4f}")

    print("\nIf (1) all match, (2) climbs toward ~1.637, and (3) primes top the list,")
    print("then the TL primitive-set substrate is consistent with classical NT and")
    print("ready for exp69's differentiable counter-example search.")


if __name__ == "__main__":
    main()
