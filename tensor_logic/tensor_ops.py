"""Fast tensor operators used by Personal Physics query plans."""

from __future__ import annotations

import torch

from .world_tensor import SparseWorldTensor


def sparse_binary_compose(
    left: SparseWorldTensor,
    right: SparseWorldTensor,
) -> torch.Tensor:
    """Compose two binary relations using sparse matrix multiplication.

    If left is A x B and right is B x C, return a sparse A x C tensor whose
    values are path weights/counts. Boolean entailment is obtained with > 0.
    """
    if len(left.axes) != 2 or len(right.axes) != 2:
        raise ValueError("binary composition requires two 2D tensors")
    if left.axes[1].symbols != right.axes[0].symbols:
        raise ValueError(
            "composition inner axes must have identical symbol ordering"
        )

    product = torch.sparse.mm(
        left.sparse(),
        right.sparse(),
    )
    if product.layout == torch.strided:
        return product.to_sparse().coalesce()
    return product.coalesce()


def booleanize_sparse(value: torch.Tensor) -> torch.Tensor:
    """Convert nonzero sparse values to 1 while preserving sparse indices."""
    value = value.coalesce()
    return torch.sparse_coo_tensor(
        value.indices(),
        torch.ones_like(value.values()),
        value.shape,
    ).coalesce()


def vectorized_weighted_score(
    features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Score N candidates against F visible feature weights."""
    if features.ndim != 2:
        raise ValueError("features must be [candidate, feature]")
    if weights.ndim != 1:
        raise ValueError("weights must be [feature]")
    if features.shape[1] != weights.shape[0]:
        raise ValueError("feature and weight dimensions do not match")
    return features @ weights


def topk_indices(
    scores: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0 < k <= scores.shape[0]:
        raise ValueError("k must be between 1 and candidate count")
    return torch.topk(scores, k)
