"""ACT-R-inspired memory activation over typed cognitive state.

This module intentionally implements a small, inspectable subset useful for
experiments. It is not a full ACT-R implementation.

Base-level activation:
    B_i = log(sum_j (t_now - t_ij)^(-d))

Context association is represented as an additive weighted signal:
    A_i = B_i + context_i + mismatch_i

Retrieval probability and latency use standard ACT-R-style monotonic forms:
    p(retrieve_i) = sigmoid((A_i - threshold) / noise_scale)
    latency_i = latency_factor * exp(-A_i)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class MemoryActivation:
    item_id: str
    base_level: float
    associative: float
    mismatch: float
    total: float
    retrieval_probability: float
    latency: float


def base_level_activation(
    use_times: Sequence[float],
    *,
    now: float,
    decay: float = 0.5,
    minimum_age: float = 1e-6,
) -> float:
    if not use_times:
        return float("-inf")
    if decay <= 0:
        raise ValueError("decay must be positive")
    strengths = []
    for use_time in use_times:
        age = max(now - float(use_time), minimum_age)
        if age <= 0:
            raise ValueError("use time cannot be in the future")
        strengths.append(age ** (-decay))
    return math.log(sum(strengths))


def retrieval_probability(
    activation: float,
    *,
    threshold: float = 0.0,
    noise_scale: float = 1.0,
) -> float:
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    x = (activation - threshold) / noise_scale
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def retrieval_latency(
    activation: float,
    *,
    latency_factor: float = 1.0,
) -> float:
    if latency_factor <= 0:
        raise ValueError("latency_factor must be positive")
    return latency_factor * math.exp(-activation)


def compute_activation(
    item_id: str,
    *,
    use_times: Sequence[float],
    now: float,
    associative: float = 0.0,
    mismatch: float = 0.0,
    decay: float = 0.5,
    threshold: float = 0.0,
    noise_scale: float = 1.0,
    latency_factor: float = 1.0,
) -> MemoryActivation:
    base = base_level_activation(
        use_times,
        now=now,
        decay=decay,
    )
    total = base + associative + mismatch
    return MemoryActivation(
        item_id=item_id,
        base_level=base,
        associative=associative,
        mismatch=mismatch,
        total=total,
        retrieval_probability=retrieval_probability(
            total,
            threshold=threshold,
            noise_scale=noise_scale,
        ),
        latency=retrieval_latency(
            total,
            latency_factor=latency_factor,
        ),
    )


def rank_memories(
    histories: Mapping[str, Sequence[float]],
    *,
    now: float,
    associative: Mapping[str, float] | None = None,
    mismatch: Mapping[str, float] | None = None,
    k: int | None = None,
    decay: float = 0.5,
    threshold: float = 0.0,
    noise_scale: float = 1.0,
    latency_factor: float = 1.0,
) -> tuple[MemoryActivation, ...]:
    if not histories:
        return ()
    if k is not None and not 0 < k <= len(histories):
        raise ValueError("k must be within memory count")

    associative = associative or {}
    mismatch = mismatch or {}
    rows = [
        compute_activation(
            item_id,
            use_times=times,
            now=now,
            associative=float(associative.get(item_id, 0.0)),
            mismatch=float(mismatch.get(item_id, 0.0)),
            decay=decay,
            threshold=threshold,
            noise_scale=noise_scale,
            latency_factor=latency_factor,
        )
        for item_id, times in histories.items()
    ]
    rows.sort(
        key=lambda row: (
            row.total,
            row.retrieval_probability,
            -row.latency,
            row.item_id,
        ),
        reverse=True,
    )
    if k is not None:
        rows = rows[:k]
    return tuple(rows)


def tensor_base_level_activation(
    ages: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    decay: float = 0.5,
) -> torch.Tensor:
    """Vectorized base-level activation for [item, exposure] age matrices."""
    if ages.ndim != 2 or valid_mask.shape != ages.shape:
        raise ValueError("ages and valid_mask must have shape [item, exposure]")
    if decay <= 0:
        raise ValueError("decay must be positive")
    safe_age = torch.clamp(ages, min=1e-6)
    strength = torch.where(
        valid_mask,
        safe_age.pow(-decay),
        torch.zeros_like(safe_age),
    )
    total = strength.sum(dim=1)
    return torch.where(
        total > 0,
        torch.log(total),
        torch.full_like(total, float("-inf")),
    )
