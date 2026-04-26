# QuickSilver

A pedagogical pure-Python implementation of the QuickSilver
zero-knowledge proof system from

> Yang, Sarkar, Weng, Wang — *QuickSilver: Efficient and Affordable
> Zero-Knowledge Proofs for Circuits and Polynomials over Any Field*,
> ACM CCS 2021. <https://eprint.iacr.org/2021/076>

The implementation focuses on clarity, not performance: ~600 lines of
Python, no third-party dependencies, no extension fields, no LPN-based
VOLE extension. It models the VOLE preprocessing as a trusted dealer
and then implements the QuickSilver protocol on top. Soundness error
is `~(m + d) / |F|` for `m` multiplication gates of total degree `d`,
which over the Mersenne prime `p = 2^127 - 1` is well below `2^-120`
for any circuit you can run in pure Python.

## What QuickSilver does

Designated-verifier zero-knowledge proofs for arbitrary arithmetic
circuits over a large prime field, with two key properties:

- **Linear operations are free.** Add, subtract, scale-by-constant
  cost no communication. They reduce to local operations on
  information-theoretic MACs.
- **Each multiplication gate costs ~2 field elements amortised.** A
  single batched check at the end of the circuit covers every mul
  gate at once.
- **Polynomial extension:** any degree-`d` polynomial relation between
  committed values can be checked in `d` field elements per batch,
  *without* introducing intermediate multiplication wires.

## How it works in one screen

For every committed wire value `x`, the prover holds `(x, M)` and the
verifier holds `K` such that

    K = M + Delta * x

where `Delta` is a global secret of the verifier sampled once at
setup. This is an information-theoretic MAC: the prover cannot change
`x` without knowing `Delta`, but `Delta` is hidden to the prover.

A single VOLE preprocessing element is a triple `(u, v, w)` with
`w = u * Delta + v`, distributed so the prover gets `(u, v)` and the
verifier gets `w`. To commit to a value `x`, the prover sends
`d = x - u` and both sides locally derive `(M, K)` with the IT-MAC
relation above (see `itmac.py`).

Linear gates compose IT-MACs trivially:

    (x + y, M_x + M_y)   <->   K_x + K_y
    (c * x, c * M_x)     <->   c * K_x
    (x + c, M_x)         <->   K_x + Delta * c

For multiplication `z = x * y`, the verifier can locally compute

    B = K_x * K_y - Delta * K_z
      = M_x * M_y + Delta * (x*M_y + y*M_x - M_z) + Delta^2 * (x*y - z)
      =: A_0     + Delta *  A_1                  + Delta^2 * (x*y - z)

If `z = x*y`, the `Delta^2` term vanishes and `B` is a linear
polynomial in `Delta` whose coefficients `(A_0, A_1)` only the prover
knows. The prover sends them; the verifier checks `B == A_0 + A_1 *
Delta`. To batch many multiplications and stay zero-knowledge:

1. Verifier picks a random challenge `chi`.
2. Prover and verifier aggregate `A_e := sum_i chi^i * A_{e,i}` and
   `B := sum_i chi^i * B_i`.
3. Prover sends `(U, V) = (A_0 + b, A_1 + a)` where `(a, b)` is one
   fresh masking VOLE element with verifier-side value
   `c = a*Delta + b`.
4. Verifier accepts iff `B + c == U + V * Delta`.

That is the entire protocol.

## Layout

    quicksilver/
        field.py            Mersenne-prime field arithmetic
        vole.py             Trusted-dealer VOLE preprocessing
        itmac.py            Information-theoretic MAC wires
        circuit.py          Tiny arithmetic-circuit DSL
        protocol.py         Prover and Verifier walkers, message types
        polynomial.py       Degree-d polynomial-zero check (Section 5)
        zk_reachability.py  Tensor-logic tie-in: ZK proof of graph reachability

    tests/test_quicksilver.py        completeness + soundness, 18 tests
    tests/test_zk_reachability.py    7 tests for the reachability circuit
    demos/quicksilver_demo.py        factorisation, x^3+x+5=y, batched polys
    demos/zk_graph_reachability.py   "I know a graph and a walk inside it"

## Usage

```python
from quicksilver import Circuit, run

# Prove knowledge of (p, q) with p * q == N, without revealing them.
N = 999985999949
c = Circuit()
p_w = c.input()
q_w = c.input()
c.assert_eq(c.mul(p_w, q_w), N)

assert run(c, [1_000_003, 999_983])   # honest prover -> verifier accepts
```

## Run

```bash
python -m pytest tests/ -v                      # 25 passing
python demos/quicksilver_demo.py                # ~1 ms total
python demos/zk_graph_reachability.py           # ~30 ms total
```

## Tensor-logic tie-in

The same einsum recurrence that defines transitive closure in
`demos/transitive_closure.py`,

    Path = step( Edge + einsum('xy,yz->xz', Path, Edge) ),

is also the natural ZK statement: "I know a graph and a walk inside
it." `zk_reachability.py` builds a QuickSilver circuit that
constrains, for committed adjacency `E` and a sequence of one-hot
frontier vectors `alpha_0, ..., alpha_k`, the einsum identity
`alpha_i^T E alpha_{i+1} = 1` at every step. Verifier learns nothing
about `E` or about the walk's interior beyond the public boundary
(n, k, source, target). The same operator, two semantics: deductive
forward chaining when run on cleartext tensors, ZK reachability proof
when run on IT-MAC-committed tensors.

## What this leaves out

- **Real VOLE.** Production QuickSilver would generate VOLE
  correlations via an LPN-based extension protocol (SoftSpoken,
  Wolverine's Ferret) bootstrapped from a small base OT. That phase
  is the cryptographically heavy part and is orthogonal to the
  protocol implemented here.
- **Binary / extension fields.** The paper handles `F_2`, `F_{2^k}`,
  and small prime fields via a "subspace VOLE" with MACs in an
  extension. Here we only do a single large prime field, where the
  base IT-MAC suffices.
- **Public verifiability / Fiat-Shamir.** The protocol is interactive
  designated-verifier as in the paper; non-interactive variants would
  derive `chi` from a transcript hash and add a commitment phase.
- **Performance.** Pure Python, no batching of field ops into vectors,
  no precomputation of `chi^i`. A serious implementation would use
  C/AVX field arithmetic and parallel VOLE expansion.
