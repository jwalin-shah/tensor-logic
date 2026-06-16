#!/usr/bin/env python3
"""
tensor_logic_variadics.py
=========================
N-ary variadic tensor operations for tensor-logic programming,
implemented natively on Apple Silicon GPUs via the MLX framework.

What "tensor logic" needs, in one line: *every connective used in a
logic rule must accept an arbitrary number of tensor arguments and
broadcast over them*. This module supplies that primitive set, plus
a tiny DSL (`Rule`) that lets you write differentiable rules such as

    R(x,z) = sum_y   P(x,y) * Q(y,z)        # multiplicative rule
    S(x)   = max_y   A(x,y)  ∧ B(y)         # soft existential
    T(x)   = or_(    P(x), Q(x), R(x) )     # disjunctive blend

All ops run lazily on the Metal GPU through `mlx.core`; wrap any of
them in `mx.compile` for fusing.

Tested with:  mlx >= 0.20   (Apple Silicon, macOS 14+)
"""

from __future__ import annotations
import operator
from functools import reduce
from typing import Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

__version__ = "1.0.0"
__all__ = [
    "Tensor", "Rule",
    "vbroadcast", "vreduce",
    "vsum", "vprod", "vmax", "vmin", "vmean",
    "vand", "vor", "vsoftmax", "vmajority",
]

Tensor = mx.array
BinaryFn = Callable[[Tensor, Tensor], Tensor]


# ──────────────────────────────────────────────────────────────────────
#  Broadcasting helpers
# ──────────────────────────────────────────────────────────────────────
import itertools

def _broadcast_shape(shapes: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    """NumPy/MLX-compatible broadcast of N shapes."""
    if not shapes:
        return ()
    out: List[int] = []
    for dims in itertools.zip_longest(*(reversed(s) for s in shapes), fillvalue=1):
        m = 1
        for d in dims:
            if d == 1 or d == m or m == 1:
                m = max(m, d)
            else:
                raise ValueError(f"incompatible shapes for broadcast: {shapes}")
        out.append(m)
    return tuple(reversed(out))


def vbroadcast(*tensors: Tensor) -> List[Tensor]:
    """Broadcast N tensors to a common shape, returning a list."""
    if not tensors:
        return []
    target = _broadcast_shape([t.shape for t in tensors])
    return [mx.broadcast_to(t, target) for t in tensors]


def vreduce(fn: BinaryFn,
            *tensors: Tensor,
            identity: Optional[Tensor] = None) -> Tensor:
    """Left-fold `fn` over N tensors after broadcasting."""
    if not tensors:
        if identity is None:
            raise ValueError("vreduce needs ≥1 tensor or an identity")
        return identity
    return reduce(fn, vbroadcast(*tensors))


# ──────────────────────────────────────────────────────────────────────
#  Arithmetic N-ary variadics
# ──────────────────────────────────────────────────────────────────────
def vsum(*tensors: Tensor) -> Tensor:
    """Element-wise sum across N broadcasted tensors."""
    return vreduce(operator.add, *tensors, identity=mx.array(0.0))


def vprod(*tensors: Tensor) -> Tensor:
    """Element-wise product across N broadcasted tensors."""
    return vreduce(operator.mul, *tensors, identity=mx.array(1.0))


def vmax(*tensors: Tensor) -> Tensor:
    """Element-wise maximum across N broadcasted tensors."""
    if not tensors:
        raise ValueError("vmax requires ≥1 tensor")
    return reduce(mx.maximum, vbroadcast(*tensors))


def vmin(*tensors: Tensor) -> Tensor:
    """Element-wise minimum across N broadcasted tensors."""
    if not tensors:
        raise ValueError("vmin requires ≥1 tensor")
    return reduce(mx.minimum, vbroadcast(*tensors))


def vmean(*tensors: Tensor,
          weights: Optional[Sequence[float]] = None) -> Tensor:
    """Weighted (or uniform) mean across N tensors."""
    if not tensors:
        raise ValueError("vmean requires ≥1 tensor")
    if weights is None:
        return vsum(*tensors) / float(len(tensors))
    if len(weights) != len(tensors):
        raise ValueError("weights length must equal #tensors")
    w = mx.array(weights, dtype=tensors[0].dtype)
    bs = vbroadcast(*tensors)
    return vsum(*(w[i] * bs[i] for i in range(len(bs)))) / w.sum()


# ──────────────────────────────────────────────────────────────────────
#  Logical / fuzzy N-ary variadics
# ──────────────────────────────────────────────────────────────────────
def vand(*tensors: Tensor) -> Tensor:
    """Fuzzy AND (product t-norm) over N tensors."""
    return vprod(*tensors)


def vor(*tensors: Tensor) -> Tensor:
    """Fuzzy OR (probabilistic sum, in [0,1]) over N tensors."""
    if not tensors:
        raise ValueError("vor requires ≥1 tensor")
    acc = mx.zeros_like(tensors[0])
    for t in vbroadcast(*tensors):
        acc = acc + t - acc * t          # a ⊕ b = a + b − ab
    return acc


def vsoftmax(*tensors: Tensor,
             temperature: float = 1.0,
             axis: int = 0) -> Tensor:
    """Generalised softmax: stacks N tensors then softmaxes along `axis`."""
    if not tensors:
        raise ValueError("vsoftmax requires ≥1 tensor")
    stacked = mx.stack(vbroadcast(*tensors), axis=axis)
    return mx.softmax(stacked / temperature, axis=axis)


def vmajority(*tensors: Tensor, axis: int = 0) -> Tensor:
    """Soft majority: probability that the modal class dominates, per row."""
    if not tensors:
        raise ValueError("vmajority requires ≥1 tensor")
    stacked = mx.stack(vbroadcast(*tensors), axis=axis)   # (N, ...)
    p = mx.softmax(stacked, axis=axis)                    # normalised
    return mx.max(p, axis=axis)                           # top-1 mass


# ──────────────────────────────────────────────────────────────────────
#  Tiny tensor-logic DSL
# ──────────────────────────────────────────────────────────────────────
class Rule:
    """
    A differentiable, broadcastable tensor-logic rule.

    Example
    -------
    >>> Parent = mx.array([[0,1,0],[0,0,1],[0,0,0]])        # 3×3
    >>> Sibling = Rule("sibling", arity=2,
    ...                body=lambda p: vmax(p[0][:,None]*p[0][None,:],
    ...                                      p[1][:,None]*p[1][None,:]))
    """
    __slots__ = ("name", "arity", "body")

    def __init__(self, name: str, arity: int, body: Callable[..., Tensor]):
        self.name = name
        self.arity = arity
        self.body = body

    def __call__(self, *facts: Tensor) -> Tensor:
        if len(facts) != self.arity:
            raise TypeError(
                f"Rule {self.name!r} expects {self.arity} facts, got {len(facts)}"
            )
        return self.body(*facts)

    def __repr__(self) -> str:                      # pragma: no cover
        return f"Rule(name={self.name!r}, arity={self.arity})"


# ──────────────────────────────────────────────────────────────────────
#  Self-test / demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    a = mx.array([1.0, 2.0, 3.0])                   # (3,)
    b = mx.array([[10.0], [20.0], [30.0]])          # (3,1)
    c = mx.array(0.5)                               # scalar

    print("a       =", a)
    print("b       =\n", b)
    print("c       =", c)
    print("-" * 50)
    print("vsum    =\n", vsum(a, b, c))
    print("vprod   =\n", vprod(a, b, c))
    print("vmax    =\n", vmax(a, b, c))
    print("vmin    =\n", vmin(a, b, c))
    print("vmean(w=[1,2,3]) =\n",
          vmean(a, b, c, weights=[1.0, 2.0, 3.0]))
    print("vand    =\n", vand(a, b, c))
    print("vor     =\n", vor(a, b, c))
    print("vsoftmax(T=0.5) =\n", vsoftmax(a, b, c, temperature=0.5))
    print("vmajority      =", vmajority(a, b, c))

    # A toy tensor-logic rule
    P, Q = mx.array([[1, 0], [0, 1]]), mx.array([[0, 1], [1, 0]])
    R = Rule("or_rule", arity=2, body=lambda x, y: vor(x, y))
    print("Rule    =\n", R(P, Q))
