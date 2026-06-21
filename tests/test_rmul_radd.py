# TENSOR-LOGIC-OPS: commutative expression nodes must accept reflected
# operands so `2 * t == t * 2` (and likewise for `+`). Without these
# overrides, Python's `int.__mul__(2, t)` returns NotImplemented and
# raises TypeError before our `__mul__` ever runs.
from __future__ import annotations

from tensor_logic.language import (
    Domain,
    Product,
    Relation,
    Scalar,
    Sum,
    TensorRef,
)


def _two_domain_relation() -> Relation:
    d = Domain(("x", "y"))
    rel = Relation("R", d, d)
    return rel


def test_scalar_rmul_with_tensorref_returns_product():
    rel = _two_domain_relation()
    t = rel[("x", "y")]
    # 2 * t must succeed (no TypeError).
    expr = 2 * t
    assert isinstance(expr, Product)
    # the Product wraps (Scalar, TensorRef) in that order so the
    # semantic that `2 * t` lifts the scalar on the left is preserved.
    assert isinstance(expr.left, Scalar)
    assert isinstance(expr.right, TensorRef)


def test_tensorref_rmul_with_int_returns_product():
    rel = _two_domain_relation()
    t = rel[("x", "y")]
    expr = t * 3
    assert isinstance(expr, Product)
    assert isinstance(expr.left, TensorRef)
    assert isinstance(expr.right, Scalar)


def test_rmul_matches_mul_commutativity():
    rel = _two_domain_relation()
    t = rel[("x", "y")]
    # 2 * t == t * 2 in semantic terms: same Product shape, just swapped args.
    left = 2 * t
    right = t * 2
    assert type(left) is type(right)
    assert left.left == right.right
    assert left.right == right.left


def test_radd_matches_add_commutativity():
    rel = _two_domain_relation()
    t = rel[("x", "y")]
    left = 1 + t
    right = t + 1
    assert isinstance(left, Sum)
    assert isinstance(right, Sum)
    assert type(left) is type(right)
    assert left.left == right.right
    assert left.right == right.left


def test_product_rmul_with_scalar_is_product():
    rel = _two_domain_relation()
    t = rel[("x", "y")]
    inner = t * 2
    outer = 3 * inner
    assert isinstance(outer, Product)
    # the outermost Product has the scalar on the left, the inner Product on the right.
    assert isinstance(outer.left, Scalar)
    assert isinstance(outer.right, Product)


def test_sum_radd_with_scalar_is_sum():
    rel = _two_domain_relation()
    t = rel[("x", "y")]
    inner = t + 1
    outer = 4 + inner
    assert isinstance(outer, Sum)
    assert isinstance(outer.left, Scalar)
    assert isinstance(outer.right, Sum)
