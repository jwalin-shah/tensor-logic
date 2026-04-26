"""
exp72: Characterize the tail-extremal primitive sets that exp71's TL search
recovers — what shape are they?

exp71 found that for x >= 5 the differentiable search returns primitive sets
A_x in [x, N] with f(A_x) far above f(primes ∩ [x, N]) — by 1.79x at x=10
and 7.36x at x=1000. The natural hypothesis is that the search rediscovers
the *dyadic band* construction: integers in [x, 2x) are automatically
pairwise non-divisible (a < b in [x, 2x) ⟹ b/a < 2 ⟹ a does not divide b).

This file pulls A_x out of the search at three representative x values
{10, 100, 1000} and tests three structural hypotheses:

  H_band  — A_x is concentrated in the first dyadic band [x, 2x).
  H_SPF   — A_x is concentrated on integers whose smallest prime factor
            exceeds sqrt(x) (primitivity-by-prime-floor).
  H_freq  — The mass-weighted f(A_x) is well approximated by the natural
            "all of [x, 2x)" construction, modulo small adjustments.

For each x we report:

  - top 20 elements of A_x by integer value, plus their SPF
  - dyadic-band coverage:  |A_x ∩ [x, 2x)|, |A_x ∩ [2x, 4x)|, ...
  - SPF profile:  histogram of smallest prime factors among A_x
  - f comparisons against three explicit primitive sets:
        primes_tail   = primes ∩ [x, N]                  (Erdős extremum at x=2)
        dyadic_band   = integers in [x, 2x)              (primitive by construction)
        spf_floor     = {n in [x, N] : SPF(n) > sqrt(x)} (primitive when x >= 4)
        all_x_to_2x_n = integers in [x, 2x) ∩ [x, N]     (=dyadic_band ∩ search range)

The search and the explicit constructions both feed into the same TL einsum
substrate (D and w from exp68), so the comparison is apples-to-apples.

If H_band holds, exp60 (forward sub-Markov-chain construction) can be
designed knowing that the relevant high-mass primitive sets in each tail
are dyadic-band-shaped, which simplifies the chain's transition tensor.
"""

import math
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import exp68_erdos_primitive_sum as e68
import exp69_tl_primitive_search as e69
import exp71_tail_bound as e71


N = 2000
XS = [10, 100, 1000]


def smallest_prime_factor(n):
    if n < 2:
        return None
    if n % 2 == 0:
        return 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            return p
        p += 2
    return n  # n is prime


def f_of(elts):
    return sum(1.0 / (a * math.log(a)) for a in elts) if elts else 0.0


def dyadic_buckets(elts, x):
    """Return list of (lo, hi, count) for buckets [x, 2x), [2x, 4x), ..."""
    buckets = []
    lo = x
    while lo <= max(elts) if elts else 0:
        hi = lo * 2
        cnt = sum(1 for a in elts if lo <= a < hi)
        buckets.append((lo, hi, cnt))
        if not elts or hi > max(elts):
            break
        lo = hi
    return buckets


def spf_profile(elts, top_k=10):
    """Return dict spf -> count, with everything beyond top_k bucketed under 'other'."""
    counts = {}
    for a in elts:
        spf = smallest_prime_factor(a)
        counts[spf] = counts.get(spf, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])
    head = sorted_items[:top_k]
    other = sum(c for _, c in sorted_items[top_k:])
    return head, other


def maximal_primitive_extension_of_band(x, N):
    """The set [x, 2x) is automatically primitive. Extend it greedily within
    [x, N+1] by adding any integer n NOT divisible by any element of [x, 2x).
    This is the unique maximal primitive set in [x, N+1] containing [x, 2x).
    Conjectured to match what exp71's TL search recovers."""
    band = list(range(x, 2 * x))
    ext = list(band)
    for n in range(2 * x, N + 2):
        if all(n % a != 0 for a in band):
            ext.append(n)
    return ext


def main():
    print(f"=== exp72: characterize tail-extremal primitive sets ===")
    print(f"N = {N}, sweep x in {XS}\n")

    D = e68.divisibility_tensor(N).to(torch.float64)
    w = e68.erdos_weights(N)
    primes_N = e68.primes_below(N + 1)

    for x in XS:
        s = e71.search_tail(D, w, x)
        idx_hi = (s > 0.5).nonzero(as_tuple=True)[0].tolist()
        A_x = sorted(i + 2 for i in idx_hi)
        # Greedy reduction (search may return non-primitive due to soft optima)
        A_x_g = e69.make_primitive_greedy(A_x)
        is_prim = e68.is_primitive_direct(A_x_g)

        f_search = f_of(A_x)
        f_greedy = f_of(A_x_g)

        # Comparison primitive sets
        primes_tail = [p for p in primes_N if x <= p <= N + 1]
        dyadic_band_full = list(range(x, 2 * x))
        dyadic_band = [a for a in dyadic_band_full if 2 <= a <= N + 1]
        sqrt_x = math.sqrt(x)
        spf_floor = [a for a in range(max(x, 2), N + 2) if smallest_prime_factor(a) > sqrt_x]

        # Sanity: confirm dyadic_band and spf_floor are primitive
        prim_band = e68.is_primitive_direct(dyadic_band)
        prim_floor = e68.is_primitive_direct(spf_floor)

        f_pri = f_of(primes_tail)
        f_band = f_of(dyadic_band)
        f_floor = f_of(spf_floor)

        # Maximal primitive extension of the dyadic seed [x, 2x)
        ext = maximal_primitive_extension_of_band(x, N)
        f_ext = f_of(ext)
        prim_ext = e68.is_primitive_direct(ext)

        # Set agreement: what fraction of greedy(search) lies in ext, and vice versa
        ext_set = set(ext)
        gs_set = set(A_x_g)
        in_both = ext_set & gs_set
        only_search = gs_set - ext_set
        only_ext = ext_set - gs_set

        print(f"--- x = {x}  (sqrt(x) ≈ {sqrt_x:.2f}) ---")
        print(f"  search        : |A|={len(A_x):4d}  f={f_search:.4f}   primitive (raw)?={e68.is_primitive_direct(A_x)}")
        print(f"  greedy(search): |A|={len(A_x_g):4d}  f={f_greedy:.4f}   primitive?={is_prim}")
        print(f"  primes∩[x,N]  : |A|={len(primes_tail):4d}  f={f_pri:.4f}")
        print(f"  [x, 2x)       : |A|={len(dyadic_band):4d}  f={f_band:.4f}   primitive?={prim_band}")
        print(f"  SPF > sqrt(x) : |A|={len(spf_floor):4d}  f={f_floor:.4f}   primitive?={prim_floor}")
        print(f"  max-ext([x,2x)): |A|={len(ext):4d}  f={f_ext:.4f}   primitive?={prim_ext}")
        print(f"  search ∩ max-ext: {len(in_both)}    only-search: {len(only_search)}    only-ext: {len(only_ext)}")

        # Top 20 elements of greedy search set with their SPFs
        head_elems = A_x_g[:20]
        head_str = "  ".join(f"{a}(spf={smallest_prime_factor(a)})" for a in head_elems)
        print(f"  first 20 of greedy(search): {head_str}")

        # Dyadic bucket coverage
        print(f"  dyadic-band coverage of greedy(search):")
        for lo, hi, cnt in dyadic_buckets(A_x_g, x):
            frac = cnt / max(len(A_x_g), 1)
            print(f"    [{lo:5d}, {hi:5d}):  {cnt:4d}  ({frac*100:5.1f}%)")

        # SPF profile
        head, other = spf_profile(A_x_g, top_k=8)
        spf_str = "  ".join(f"spf={k}:{v}" for k, v in head)
        print(f"  SPF profile (top 8): {spf_str}   other={other}")

        print()

    print("Reading: H_band predicts heavy concentration of greedy(search) in [x, 2x).")
    print("H_SPF predicts SPF profile dominated by primes p > sqrt(x). f comparisons")
    print("against the explicit constructions (primes_tail, [x,2x), SPF > sqrt(x))")
    print("show which natural primitive set is closest to what the search recovered.")


if __name__ == "__main__":
    main()
