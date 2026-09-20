"""Unlabeled relation-discovery benchmark for Research OS v0.

The learner sees numeric pairwise observations only. Human-readable relation
names and truth values live exclusively in HiddenEvaluation and are never
passed to learner fitting functions.

This first slice is intentionally small: it establishes the zero-trust data
boundary, deterministic train/test splits, anonymous relation channels, and a
permutation-invariant evaluator before adding larger neural models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tensor_logic.research.synthetic_scene import (
    CANVAS,
    OBJ_IDX,
    OBJ_NAMES,
    PAIRS,
    RELATIONS,
    compute_pairwise_spatial_relations,
    generate_colored_object_sequence,
)


@dataclass(frozen=True)
class PairObservations:
    """Data that an unlabeled learner is allowed to observe."""

    features: torch.Tensor
    frame_ids: torch.Tensor
    pair_slots: torch.Tensor

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class HiddenEvaluation:
    """Evaluator-only truth. Never pass this object into learner fitting."""

    targets: torch.Tensor
    relation_names: tuple[str, ...]


@dataclass(frozen=True)
class HiddenWorldSplit:
    observations: PairObservations
    evaluation: HiddenEvaluation
    seed: int


@dataclass(frozen=True)
class PCARelationModel:
    mean: torch.Tensor
    scale: torch.Tensor
    components: torch.Tensor

    def transform(self, observations: PairObservations) -> torch.Tensor:
        standardized = (
            observations.features - self.mean
        ) / self.scale
        scores = standardized @ self.components
        return torch.sigmoid(scores)


@dataclass(frozen=True)
class KMeansRelationModel:
    mean: torch.Tensor
    scale: torch.Tensor
    centers: torch.Tensor

    def transform(self, observations: PairObservations) -> torch.Tensor:
        standardized = (
            observations.features - self.mean
        ) / self.scale
        distances = torch.cdist(standardized, self.centers)
        assignments = distances.argmin(dim=1)
        return torch.nn.functional.one_hot(
            assignments,
            num_classes=self.centers.shape[0],
        ).float()


def _pair_features(
    positions: np.ndarray,
    pair_slot: int,
) -> list[float]:
    """Raw geometric/context features with no semantic relation labels."""
    a, b = PAIRS[pair_slot]
    ai, bi = OBJ_IDX[a], OBJ_IDX[b]
    pa, pb = positions[ai], positions[bi]

    dx = float(pb[0] - pa[0]) / CANVAS
    dy = float(pb[1] - pa[1]) / CANVAS
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    chebyshev = max(abs_dx, abs_dy)
    euclidean = math.sqrt(dx * dx + dy * dy)
    pair_order = float(bi - ai) / max(len(OBJ_NAMES) - 1, 1)

    return [
        dx,
        dy,
        abs_dx,
        abs_dy,
        chebyshev,
        euclidean,
        pair_order,
    ]


def generate_hidden_world_split(
    *,
    n_frames: int,
    seed: int,
) -> HiddenWorldSplit:
    """Generate observations plus evaluator-only hidden relational truth."""
    rng = np.random.default_rng(seed)
    features: list[list[float]] = []
    targets: list[list[float]] = []
    frame_ids: list[int] = []
    pair_slots: list[int] = []

    for frame_id in range(n_frames):
        _, pos_hist = generate_colored_object_sequence(
            n_frames=1,
            rng=rng,
        )
        positions = pos_hist[0]
        hidden = compute_pairwise_spatial_relations(positions)

        for pair_slot, (a, b) in enumerate(PAIRS):
            features.append(_pair_features(positions, pair_slot))
            targets.append(
                [
                    float(hidden[(relation, a, b)])
                    for relation in RELATIONS
                ]
            )
            frame_ids.append(frame_id)
            pair_slots.append(pair_slot)

    return HiddenWorldSplit(
        observations=PairObservations(
            features=torch.tensor(features, dtype=torch.float32),
            frame_ids=torch.tensor(frame_ids, dtype=torch.long),
            pair_slots=torch.tensor(pair_slots, dtype=torch.long),
        ),
        evaluation=HiddenEvaluation(
            targets=torch.tensor(targets, dtype=torch.float32),
            relation_names=tuple(RELATIONS),
        ),
        seed=seed,
    )


def _standardization(
    observations: PairObservations,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = observations.features.mean(dim=0)
    scale = observations.features.std(dim=0).clamp_min(1e-6)
    standardized = (observations.features - mean) / scale
    return standardized, mean, scale


def fit_pca_relations(
    observations: PairObservations,
    *,
    n_channels: int = 4,
) -> PCARelationModel:
    """Learn anonymous relation channels using PCA, with no target labels."""
    standardized, mean, scale = _standardization(observations)
    covariance = (
        standardized.T @ standardized
        / max(standardized.shape[0] - 1, 1)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    n_channels = min(
        n_channels,
        standardized.shape[1],
    )
    components = eigenvectors[:, order[:n_channels]]
    return PCARelationModel(
        mean=mean,
        scale=scale,
        components=components,
    )


def _farthest_point_initialization(
    features: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """Deterministic center initialization for the clustering baseline."""
    distances_from_mean = torch.norm(
        features - features.mean(dim=0, keepdim=True),
        dim=1,
    )
    first = int(distances_from_mean.argmax().item())
    chosen = [first]

    while len(chosen) < n_clusters:
        current = features[chosen]
        min_distance = torch.cdist(features, current).min(dim=1).values
        min_distance[chosen] = -1
        chosen.append(int(min_distance.argmax().item()))

    return features[chosen].clone()


def fit_kmeans_relations(
    observations: PairObservations,
    *,
    n_channels: int = 4,
    max_iters: int = 30,
) -> KMeansRelationModel:
    """Deterministic raw-feature clustering baseline."""
    standardized, mean, scale = _standardization(observations)
    n_channels = min(n_channels, standardized.shape[0])
    centers = _farthest_point_initialization(
        standardized,
        n_channels,
    )

    for _ in range(max_iters):
        distances = torch.cdist(standardized, centers)
        assignments = distances.argmin(dim=1)
        next_centers = centers.clone()
        for index in range(n_channels):
            mask = assignments == index
            if mask.any():
                next_centers[index] = standardized[mask].mean(dim=0)
        if torch.allclose(next_centers, centers):
            break
        centers = next_centers

    return KMeansRelationModel(
        mean=mean,
        scale=scale,
        centers=centers,
    )


def random_relation_channels(
    observations: PairObservations,
    *,
    n_channels: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Random baseline with deterministic seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.rand(
        observations.n_samples,
        n_channels,
        generator=generator,
    )


def _binary_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction = prediction.float()
    target = target.float()
    tp = (prediction * target).sum().item()
    fp = (prediction * (1 - target)).sum().item()
    fn = ((1 - prediction) * target).sum().item()
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return 2 * precision * recall / max(
        precision + recall,
        1e-9,
    )


def evaluate_anonymous_channels(
    probabilities: torch.Tensor,
    hidden: HiddenEvaluation,
) -> dict:
    """Permutation-match anonymous channels to hidden relations for scoring only.

    Channel polarity is evaluator-only: PCA/eigendecomposition signs are
    arbitrary, so a channel and its complement represent the same latent split.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [samples, channels]")
    if probabilities.shape[0] != hidden.targets.shape[0]:
        raise ValueError("sample count does not match hidden evaluator")

    binary = (probabilities >= 0.5).float()
    n_channels = binary.shape[1]
    n_relations = hidden.targets.shape[1]
    scores = np.zeros((n_channels, n_relations), dtype=np.float64)
    inverted = np.zeros((n_channels, n_relations), dtype=bool)

    for channel in range(n_channels):
        for relation in range(n_relations):
            target = hidden.targets[:, relation]
            direct = _binary_f1(binary[:, channel], target)
            complement = _binary_f1(
                1 - binary[:, channel],
                target,
            )
            if complement > direct:
                scores[channel, relation] = complement
                inverted[channel, relation] = True
            else:
                scores[channel, relation] = direct

    rows, cols = linear_sum_assignment(-scores)
    matches = []
    for channel, relation in zip(rows.tolist(), cols.tolist()):
        matches.append(
            {
                "channel": f"z_rel_{channel}",
                "relation": hidden.relation_names[relation],
                "f1": float(scores[channel, relation]),
                "inverted": bool(inverted[channel, relation]),
            }
        )

    return {
        "mean_matched_f1": float(
            np.mean([match["f1"] for match in matches])
            if matches
            else 0.0
        ),
        "matches": matches,
        "score_matrix": scores.tolist(),
    }


def channels_to_world_tensors(
    probabilities: torch.Tensor,
    observations: PairObservations,
    *,
    n_frames: Optional[int] = None,
    threshold: float = 0.5,
) -> list[dict[str, torch.Tensor]]:
    """Convert anonymous channel predictions into TL-compatible relation tensors."""
    if probabilities.shape[0] != observations.n_samples:
        raise ValueError("probabilities and observations differ in sample count")

    if n_frames is None:
        n_frames = int(observations.frame_ids.max().item()) + 1

    n_objects = len(OBJ_NAMES)
    worlds = [
        {
            f"z_rel_{channel}": torch.zeros(
                n_objects,
                n_objects,
            )
            for channel in range(probabilities.shape[1])
        }
        for _ in range(n_frames)
    ]

    binary = probabilities >= threshold
    for sample in range(observations.n_samples):
        frame = int(observations.frame_ids[sample].item())
        pair_slot = int(observations.pair_slots[sample].item())
        a, b = PAIRS[pair_slot]
        ai, bi = OBJ_IDX[a], OBJ_IDX[b]
        for channel in range(probabilities.shape[1]):
            if bool(binary[sample, channel].item()):
                worlds[frame][f"z_rel_{channel}"][ai, bi] = 1.0

    return worlds



def mask_directional_information(
    observations: PairObservations,
) -> PairObservations:
    """Remove signed direction and pair-order cues for an identifiability control."""
    features = observations.features.clone()
    # dx, dy, and ordered-pair identity are the only directional cues.
    features[:, 0] = 0
    features[:, 1] = 0
    features[:, 6] = 0
    return PairObservations(
        features=features,
        frame_ids=observations.frame_ids.clone(),
        pair_slots=observations.pair_slots.clone(),
    )


def run_non_identifiable_control(
    train: HiddenWorldSplit,
    test: HiddenWorldSplit,
    *,
    n_channels: int = 4,
) -> dict:
    """Evaluator-declared control where directional predicates cannot be identified.

    The learner still receives no relation labels. The evaluator records why the
    intervention destroys identifiability rather than treating a forced
    permutation match as semantic recovery.
    """
    masked_train = mask_directional_information(train.observations)
    masked_test = mask_directional_information(test.observations)
    model = fit_pca_relations(masked_train, n_channels=n_channels)
    evaluation = evaluate_anonymous_channels(
        model.transform(masked_test),
        test.evaluation,
    )
    return {
        "status": "non_identifiable_by_construction",
        "intervention": {
            "removed_feature_indices": [0, 1, 6],
            "removed_information": [
                "signed_dx",
                "signed_dy",
                "ordered_pair_identity",
            ],
        },
        "expected_unidentifiable_relations": [
            "above",
            "left_of",
        ],
        "evaluation": evaluation,
    }


def run_unlabeled_benchmark(
    *,
    train_frames: int = 400,
    test_frames: int = 200,
    train_seed: int = 11,
    test_seed: int = 22,
    random_seed: int = 33,
    n_channels: int = 4,
) -> dict:
    """Run deterministic random, clustering, and PCA relation-discovery baselines."""
    if train_seed == test_seed:
        raise ValueError("train and test seeds must be disjoint")

    train = generate_hidden_world_split(
        n_frames=train_frames,
        seed=train_seed,
    )
    test = generate_hidden_world_split(
        n_frames=test_frames,
        seed=test_seed,
    )

    pca = fit_pca_relations(
        train.observations,
        n_channels=n_channels,
    )
    kmeans = fit_kmeans_relations(
        train.observations,
        n_channels=n_channels,
    )

    pca_probs = pca.transform(test.observations)
    kmeans_probs = kmeans.transform(test.observations)
    random_probs = random_relation_channels(
        test.observations,
        n_channels=n_channels,
        seed=random_seed,
    )

    pca_eval = evaluate_anonymous_channels(
        pca_probs,
        test.evaluation,
    )
    kmeans_eval = evaluate_anonymous_channels(
        kmeans_probs,
        test.evaluation,
    )
    random_eval = evaluate_anonymous_channels(
        random_probs,
        test.evaluation,
    )
    non_identifiable = run_non_identifiable_control(
        train,
        test,
        n_channels=n_channels,
    )
    weak_pca_matches = [
        match
        for match in pca_eval["matches"]
        if match["f1"] < 0.25
    ]

    return {
        "config": {
            "train_frames": train_frames,
            "test_frames": test_frames,
            "train_seed": train_seed,
            "test_seed": test_seed,
            "random_seed": random_seed,
            "n_channels": n_channels,
        },
        "methods": {
            "random": random_eval,
            "kmeans": kmeans_eval,
            "pca": pca_eval,
        },
        "pca_minus_random": (
            pca_eval["mean_matched_f1"]
            - random_eval["mean_matched_f1"]
        ),
        "diagnostics": {
            "weak_pca_matches": weak_pca_matches,
            "non_identifiable_control": non_identifiable,
        },
    }
