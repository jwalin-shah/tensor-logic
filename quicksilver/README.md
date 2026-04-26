# QuickSilver

A pedagogical pure-Python implementation of the QuickSilver
zero-knowledge proof system from

> Yang, Sarkar, Weng, Wang — *QuickSilver: Efficient and Affordable
> Zero-Knowledge Proofs for Circuits and Polynomials over Any Field*,
> ACM CCS 2021. <https://eprint.iacr.org/2021/076>

About 2,500 lines of Python, no third-party dependencies. Eight
modules, 76 tests, six runnable demos. Soundness error is `~m / |F|`
for `m` multiplication gates: well below `2^-120` over the Mersenne
prime `2^127 - 1`, and below `2^-127` over `GF(2^128)` for boolean
circuits.

## What QuickSilver does

Designated-verifier zero-knowledge proofs for arbitrary circuits over
any field, with three signature properties:

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
relation above.

Linear gates compose IT-MACs trivially:

    (x + y, M_x + M_y)   <->   K_x + K_y
    (c * x, c * M_x)     <->   c * K_x
    (x + c, M_x)         <->   K_x + Delta * c

For multiplication `z = x * y`, the verifier locally computes

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

Fiat-Shamir replaces the verifier's challenge in step 1 with a hash
of the transcript, making the proof a single non-interactive object.

## Layout

    quicksilver/
        field.py            Mersenne-prime field arithmetic (2^127 - 1)
        gf2k.py             GF(2^128) arithmetic, GCM polynomial
        vole.py             Trusted-dealer VOLE preprocessing
        lpn_vole.py         LPN-based PCG: sparse base -> pseudorandom output
        itmac.py            Information-theoretic MAC wires (prime field)
        circuit.py          Tiny arithmetic-circuit DSL
        protocol.py         Prime-field prover and verifier
        polynomial.py       Degree-d polynomial-zero check (Section 5)
        boolean.py          Subspace-VOLE binary-circuit prover and verifier
        fiat_shamir.py      Non-interactive variant via SHA-256 transcript
        einsum.py           Compile tensor-logic einsum into a circuit
        zk_reachability.py  ZK proof of graph reachability

    tests/                  76 tests across 6 files
    demos/                  6 runnable demos

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

Same shape with the LPN-based VOLE PCG (drop-in for the trusted
dealer):

```python
from quicksilver.lpn_vole import LpnParams, lpn_vole_extend
from quicksilver.protocol import prove, verify
from quicksilver.field import F

p_share, v_share = lpn_vole_extend(LpnParams.default(n_out=c.vole_count()))
msg1, batched = prove(c, [1_000_003, 999_983], p_share)
chi = F.rand_nonzero()
msg2 = batched(chi)
assert verify(c, v_share, msg1, chi, msg2)
```

Boolean circuits (XOR free, AND charged) over `GF(2^128)`:

```python
from quicksilver.boolean import BoolCircuit, run

c = BoolCircuit()
a, b = c.input(), c.input()
c.assert_eq_const(c.and_(a, b), 1)   # prove AND of two secret bits is 1
assert run(c, [1, 1])
```

Tensor-logic einsum compiled to a ZK circuit:

```python
from quicksilver.circuit import Circuit
from quicksilver.einsum import (
    alloc_input_tensor, assert_einsum_equals,
    evaluate_einsum, flatten_row_major,
)
from quicksilver.protocol import run

A = [[1, 2], [3, 4]]; B = [[5, 6], [7, 8]]
expected = evaluate_einsum("ij,jk->ik", [A, B])
c = Circuit()
Aw = alloc_input_tensor(c, (2, 2))
Bw = alloc_input_tensor(c, (2, 2))
assert_einsum_equals(c, "ij,jk->ik", [Aw, Bw], [(2, 2), (2, 2)], expected)
witness = flatten_row_major(A) + flatten_row_major(B)
assert run(c, witness)
```

Fiat-Shamir non-interactive proof:

```python
from quicksilver.fiat_shamir import run_ni
ok, proof = run_ni(c, [1_000_003, 999_983])
# ``proof`` is a single object; pass to verify_ni later.
```

## Run

```bash
python -m pytest tests/ -v                       # 76 passing in <1s

# Prime-field demos
python demos/quicksilver_demo.py                 # factorisation, polys
python demos/zk_graph_reachability.py            # graph + walk in ZK
python demos/zk_einsum.py                        # matmul, grandparent rule

# Binary-field demos
python demos/quicksilver_boolean_demo.py         # 8-bit multiplier, mixer

# Real-cryptography demos
python demos/lpn_vole_demo.py                    # LPN-based PCG
```

## Tensor-logic tie-in

The repo's premise -- a Datalog rule head and an einsum are the same
operation -- gives a one-line ZK frontend. The recurrence that defines
boolean transitive closure in `demos/transitive_closure.py`,

    Path = step( Edge + einsum('xy,yz->xz', Path, Edge) ),

is also the natural ZK statement "I know a graph and a walk inside
it." `zk_reachability.py` constrains, for committed adjacency `E` and
one-hot frontier vectors `alpha_0, ..., alpha_k`, the einsum identity
`alpha_i^T E alpha_{i+1} = 1` at every step. The verifier learns
nothing about `E` or the walk's interior beyond `(n, k, source,
target)`.

`einsum.py` generalises this: any tensor-logic rule head expressible
as an einsum is automatically a QuickSilver circuit. One operator,
two semantics -- deductive forward chaining on cleartext tensors, ZK
proof on IT-MAC-committed tensors.

## What this still leaves out

- **Base OT / GGM-tree single-point VOLE** to remove the trusted base
  for `lpn_vole.py`. The LPN expansion step is the cryptographically
  characteristic part of the PCG; the base would still need OT-style
  primitives implemented separately.
- **Disjunctions and lookups** (Mac'n'Cheese, lookup arguments). All
  expressible on top of this protocol but each its own subproject.
- **Performance.** Pure Python, scalar arithmetic, no FFT, no
  vectorisation. A serious implementation would use CLMUL/CLMULNI for
  GF(2^128), a CRT-friendly prime + montgomery for the prime case,
  and parallel matrix-vector multiplies for LPN expansion.
