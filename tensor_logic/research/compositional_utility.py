"""Research OS v0-C: evaluate anonymous concepts by compositional utility.

Semantic relation labels are evaluator-only. Representation fitting receives only
PairObservations and never receives hidden relation truth or downstream targets.

The benchmark keeps isolated recovery and downstream utility separate instead of
collapsing them into one arbitrary score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_logic.research.anonymous_composition import (
    AnonymousCondition,
    anonymize_oracle_worlds,
    corrupt_worlds,
    derive_hidden_target,
    evaluate_body,
    hidden_relation_worlds,
    induce_condition,
)
from tensor_logic.research.synthetic_scene import N_OBJ
from tensor_logic.research.unlabeled_relations import (
    HiddenEvaluation,
    HiddenWorldSplit,
    PairObservations,
    channels_to_world_tensors,
    evaluate_anonymous_channels,
    fit_kmeans_relations,
    fit_pca_relations,
    generate_hidden_world_split,
    random_relation_channels,
)


TARGETS = {
    "same_side": ("left_of", "left_of"),
    "blocked_path": ("touching", "above"),
}


@dataclass(frozen=True)
class AutoencoderRelationModel:
    """Small deterministic reconstruction model with anonymous bottleneck channels."""

    mean: torch.Tensor
    scale: torch.Tensor
    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor

    def transform(self, observations: PairObservations) -> torch.Tensor:
        standardized = (
            observations.features - self.mean
        ) / self.scale
        latent = torch.tanh(
            F.linear(
                standardized,
                self.encoder_weight,
                self.encoder_bias,
            )
        )
        return torch.sigmoid(2.5 * latent)


def _standardize(
    observations: PairObservations,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = observations.features.mean(dim=0)
    scale = observations.features.std(dim=0).clamp_min(1e-6)
    standardized = (observations.features - mean) / scale
    return standardized, mean, scale


def fit_autoencoder_relations(
    observations: PairObservations,
    *,
    n_channels: int = 4,
    seed: int = 123,
    steps: int = 120,
    learning_rate: float = 0.03,
) -> AutoencoderRelationModel:
    """Fit an unlabeled bottleneck by reconstructing pairwise observations only."""
    standardized, mean, scale = _standardize(observations)

    torch.manual_seed(seed)
    encoder = nn.Linear(standardized.shape[1], n_channels)
    decoder = nn.Linear(n_channels, standardized.shape[1])
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
    )

    for _ in range(steps):
        latent = torch.tanh(encoder(standardized))
        reconstructed = decoder(latent)

        # Reconstruction learns context structure. The variance term only
        # discourages dead channels; it carries no semantic supervision.
        reconstruction_loss = F.mse_loss(
            reconstructed,
            standardized,
        )
        variance = latent.var(dim=0, unbiased=False)
        activity_penalty = torch.relu(0.15 - variance).mean()
        loss = reconstruction_loss + 0.02 * activity_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return AutoencoderRelationModel(
        mean=mean.detach().clone(),
        scale=scale.detach().clone(),
        encoder_weight=encoder.weight.detach().clone(),
        encoder_bias=encoder.bias.detach().clone(),
    )


def pca_whitened_probabilities(
    model,
    observations: PairObservations,
) -> torch.Tensor:
    """Whiten PCA channel scale before calibration, without semantic labels."""
    standardized = (
        observations.features - model.mean
    ) / model.scale
    raw = standardized @ model.components
    channel_scale = raw.std(dim=0).clamp_min(1e-6)
    return torch.sigmoid(raw / channel_scale)


def evaluator_calibration(
    probabilities: torch.Tensor,
    hidden: HiddenEvaluation,
) -> dict:
    """Evaluator-only Brier score after permutation/polarity alignment."""
    evaluation = evaluate_anonymous_channels(
        probabilities,
        hidden,
    )
    brier_scores: list[float] = []

    for match in evaluation["matches"]:
        channel = int(match["channel"].split("_")[-1])
        relation = hidden.relation_names.index(match["relation"])
        probability = probabilities[:, channel]
        if match["inverted"]:
            probability = 1.0 - probability
        target = hidden.targets[:, relation]
        brier_scores.append(
            float(((probability - target) ** 2).mean().item())
        )

    return {
        "mean_matched_brier": (
            sum(brier_scores) / len(brier_scores)
            if brier_scores
            else 1.0
        ),
        "matches": evaluation["matches"],
    }


def mapping_consistency(
    induction_probabilities: torch.Tensor,
    induction_hidden: HiddenEvaluation,
    heldout_probabilities: torch.Tensor,
    heldout_hidden: HiddenEvaluation,
) -> dict:
    """Compare evaluator-only channel meanings across independent worlds."""
    left = evaluate_anonymous_channels(
        induction_probabilities,
        induction_hidden,
    )["matches"]
    right = evaluate_anonymous_channels(
        heldout_probabilities,
        heldout_hidden,
    )["matches"]

    left_by_channel = {
        match["channel"]: (
            match["relation"],
            match["inverted"],
        )
        for match in left
    }
    right_by_channel = {
        match["channel"]: (
            match["relation"],
            match["inverted"],
        )
        for match in right
    }
    shared = sorted(set(left_by_channel) & set(right_by_channel))
    if not shared:
        return {
            "semantic_consistency": 0.0,
            "polarity_consistency": 0.0,
        }

    semantic = sum(
        left_by_channel[channel][0]
        == right_by_channel[channel][0]
        for channel in shared
    ) / len(shared)
    polarity = sum(
        left_by_channel[channel]
        == right_by_channel[channel]
        for channel in shared
    ) / len(shared)

    return {
        "semantic_consistency": semantic,
        "polarity_consistency": polarity,
    }


def _representation_probabilities(
    *,
    train: HiddenWorldSplit,
    isolation: HiddenWorldSplit,
    induction: HiddenWorldSplit,
    heldout: HiddenWorldSplit,
    random_seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fit on train only; return isolation, induction, and heldout projections."""
    pca = fit_pca_relations(train.observations)
    kmeans = fit_kmeans_relations(train.observations)
    autoencoder = fit_autoencoder_relations(
        train.observations,
        seed=123,
    )

    return {
        "pca": (
            pca.transform(isolation.observations),
            pca.transform(induction.observations),
            pca.transform(heldout.observations),
        ),
        "pca_whitened": (
            pca_whitened_probabilities(
                pca,
                isolation.observations,
            ),
            pca_whitened_probabilities(
                pca,
                induction.observations,
            ),
            pca_whitened_probabilities(
                pca,
                heldout.observations,
            ),
        ),
        "kmeans": (
            kmeans.transform(isolation.observations),
            kmeans.transform(induction.observations),
            kmeans.transform(heldout.observations),
        ),
        "autoencoder": (
            autoencoder.transform(isolation.observations),
            autoencoder.transform(induction.observations),
            autoencoder.transform(heldout.observations),
        ),
        "random": (
            random_relation_channels(
                isolation.observations,
                seed=random_seed,
            ),
            random_relation_channels(
                induction.observations,
                seed=random_seed + 1,
            ),
            random_relation_channels(
                heldout.observations,
                seed=random_seed + 2,
            ),
        ),
    }


def _remove_object(
    base: dict[str, torch.Tensor],
    object_index: int,
) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in base.items():
        out = tensor.clone()
        out[object_index, :] = 0
        out[:, object_index] = 0
        result[name] = out
    return result


def _binary_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction = (prediction > 0).float().reshape(-1)
    target = (target > 0).float().reshape(-1)
    tp = (prediction * target).sum().item()
    fp = (prediction * (1 - target)).sum().item()
    fn = ((1 - prediction) * target).sum().item()
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return 2 * precision * recall / max(
        precision + recall,
        1e-9,
    )


def counterfactual_retraction_metrics(
    *,
    candidate: list[str] | None,
    learned_worlds: list[dict[str, torch.Tensor]],
    hidden_worlds: list[dict[str, torch.Tensor]],
    target_body: tuple[str, str],
    object_index: int = 2,
) -> dict:
    """Score consequences after removing one object, including the retraction delta.

    Delta F1 avoids the sparse-negative inflation of raw accuracy: it asks whether
    the consequences that disappear after intervention are the consequences that
    should disappear.
    """
    if not candidate:
        return {
            "retracted_f1": 0.0,
            "delta_f1": 0.0,
            "active_retraction_worlds": 0,
        }

    predicted_retracted = []
    target_retracted = []
    predicted_delta = []
    target_delta = []
    active_worlds = 0

    for learned, hidden in zip(learned_worlds, hidden_worlds):
        full_prediction = _apply_candidate(
            candidate,
            learned,
        )
        full_target = _apply_candidate(
            list(target_body),
            hidden,
        )
        learned_retracted = _remove_object(
            learned,
            object_index,
        )
        hidden_retracted = _remove_object(
            hidden,
            object_index,
        )
        retracted_prediction = _apply_candidate(
            candidate,
            learned_retracted,
        )
        retracted_target = _apply_candidate(
            list(target_body),
            hidden_retracted,
        )

        p_delta = (
            full_prediction != retracted_prediction
        ).float()
        t_delta = (
            full_target != retracted_target
        ).float()
        if bool(t_delta.any().item()):
            active_worlds += 1

        predicted_retracted.append(retracted_prediction)
        target_retracted.append(retracted_target)
        predicted_delta.append(p_delta)
        target_delta.append(t_delta)

    return {
        "retracted_f1": _binary_f1(
            torch.block_diag(*predicted_retracted),
            torch.block_diag(*target_retracted),
        ),
        "delta_f1": _binary_f1(
            torch.block_diag(*predicted_delta),
            torch.block_diag(*target_delta),
        ),
        "active_retraction_worlds": active_worlds,
    }


def _apply_candidate(
    body: list[str],
    base: dict[str, torch.Tensor],
) -> torch.Tensor:
    from tensor_logic.research.utils import apply_body

    result = apply_body(body, base)
    result.fill_diagonal_(0)
    return result


def _worlds_from_probabilities(
    probabilities: torch.Tensor,
    split: HiddenWorldSplit,
) -> list[dict[str, torch.Tensor]]:
    return channels_to_world_tensors(
        probabilities,
        split.observations,
    )


def _noise_stability(
    *,
    candidate: list[str] | None,
    heldout_worlds: list[dict[str, torch.Tensor]],
    heldout_target: torch.Tensor,
    admission_threshold: float,
    seed_offset: int,
) -> dict:
    curve = {}
    failure_point = None
    baseline_passes = False

    for index, noise in enumerate((0.0, 0.05, 0.10, 0.20, 0.30)):
        noisy_worlds = corrupt_worlds(
            heldout_worlds,
            noise=noise,
            seed=seed_offset + index,
        )
        evaluation = evaluate_body(
            candidate,
            _block_diagonal(noisy_worlds),
            heldout_target,
            threshold=admission_threshold,
        )
        curve[f"{noise:.2f}"] = {
            "heldout_f1": evaluation["heldout_f1"],
            "accepted": evaluation["accepted"],
        }
        if noise == 0.0:
            baseline_passes = evaluation["accepted"]
        elif (
            baseline_passes
            and failure_point is None
            and not evaluation["accepted"]
        ):
            failure_point = noise

    return {
        "curve": curve,
        "baseline_passes": baseline_passes,
        "failure_point_after_clean_pass": failure_point,
    }


def _block_diagonal(
    worlds: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not worlds:
        raise ValueError("at least one world is required")
    return {
        name: torch.block_diag(
            *[world[name] for world in worlds]
        )
        for name in worlds[0]
    }


def _oracle_condition(
    *,
    induction: HiddenWorldSplit,
    heldout: HiddenWorldSplit,
    mask_seed: int,
) -> AnonymousCondition:
    induction_worlds, mapping = anonymize_oracle_worlds(
        hidden_relation_worlds(induction),
        seed=mask_seed,
    )
    heldout_worlds, heldout_mapping = anonymize_oracle_worlds(
        hidden_relation_worlds(heldout),
        seed=mask_seed,
    )
    if mapping != heldout_mapping:
        raise AssertionError("oracle mapping drifted")
    return AnonymousCondition(
        name="oracle",
        induction_base=_block_diagonal(induction_worlds),
        heldout_base=_block_diagonal(heldout_worlds),
        evaluator_mapping=mapping,
    )


def run_compositional_utility_benchmark(
    *,
    train_frames: int = 200,
    isolation_frames: int = 100,
    induction_frames: int = 80,
    heldout_frames: int = 80,
    train_seed: int = 11,
    isolation_seed: int = 22,
    induction_seed: int = 33,
    heldout_seed: int = 44,
    random_seed: int = 33,
    admission_threshold: float = 0.70,
) -> dict:
    """Measure isolated and compositional utility as separate evidence vectors."""
    if len(
        {
            train_seed,
            isolation_seed,
            induction_seed,
            heldout_seed,
        }
    ) != 4:
        raise ValueError(
            "train, isolation, induction, and heldout seeds must differ"
        )

    train = generate_hidden_world_split(
        n_frames=train_frames,
        seed=train_seed,
    )
    isolation = generate_hidden_world_split(
        n_frames=isolation_frames,
        seed=isolation_seed,
    )
    induction = generate_hidden_world_split(
        n_frames=induction_frames,
        seed=induction_seed,
    )
    heldout = generate_hidden_world_split(
        n_frames=heldout_frames,
        seed=heldout_seed,
    )

    probability_triples = _representation_probabilities(
        train=train,
        isolation=isolation,
        induction=induction,
        heldout=heldout,
        random_seed=random_seed,
    )
    hidden_heldout_worlds = hidden_relation_worlds(heldout)

    representation_results = {}
    for method_index, (
        method,
        (
            isolation_probabilities,
            induction_probabilities,
            heldout_probabilities,
        ),
    ) in enumerate(probability_triples.items()):
        # This split deliberately reproduces exp84's train/test seeds for the
        # PCA/k-means isolated-recovery comparison.
        isolated = evaluate_anonymous_channels(
            isolation_probabilities,
            isolation.evaluation,
        )
        calibration = evaluator_calibration(
            isolation_probabilities,
            isolation.evaluation,
        )
        consistency = mapping_consistency(
            induction_probabilities,
            induction.evaluation,
            heldout_probabilities,
            heldout.evaluation,
        )
        induction_worlds = _worlds_from_probabilities(
            induction_probabilities,
            induction,
        )
        heldout_worlds = _worlds_from_probabilities(
            heldout_probabilities,
            heldout,
        )
        condition = AnonymousCondition(
            name=method,
            induction_base=_block_diagonal(
                induction_worlds,
            ),
            heldout_base=_block_diagonal(
                heldout_worlds,
            ),
        )

        target_results = {}
        accepted_rules = 0
        counterfactual_delta_scores = []
        counterfactual_retracted_scores = []
        noise_failure_points = []

        for target_name, target_body in TARGETS.items():
            induction_target = derive_hidden_target(
                induction,
                target_body,
            )
            heldout_target = derive_hidden_target(
                heldout,
                target_body,
            )
            composition = induce_condition(
                condition,
                induction_target,
                heldout_target,
                admission_threshold=admission_threshold,
            )
            if composition["accepted"]:
                accepted_rules += 1

            counterfactual = counterfactual_retraction_metrics(
                candidate=composition["candidate"],
                learned_worlds=heldout_worlds,
                hidden_worlds=hidden_heldout_worlds,
                target_body=target_body,
            )
            counterfactual_delta_scores.append(
                counterfactual["delta_f1"]
            )
            counterfactual_retracted_scores.append(
                counterfactual["retracted_f1"]
            )

            noise = _noise_stability(
                candidate=composition["candidate"],
                heldout_worlds=heldout_worlds,
                heldout_target=heldout_target,
                admission_threshold=admission_threshold,
                seed_offset=1000 + 100 * method_index,
            )
            if (
                noise["failure_point_after_clean_pass"]
                is not None
            ):
                noise_failure_points.append(
                    noise["failure_point_after_clean_pass"]
                )

            target_results[target_name] = {
                "target_body_evaluator_only": list(target_body),
                **composition,
                "counterfactual_retraction": counterfactual,
                "noise_stability": noise,
            }

        representation_results[method] = {
            "isolated_matched_f1": isolated["mean_matched_f1"],
            "matched_brier": calibration["mean_matched_brier"],
            "cross_world_consistency": consistency,
            "targets": target_results,
            "accepted_rule_count": accepted_rules,
            "mean_compositional_f1": sum(
                item["heldout_f1"]
                for item in target_results.values()
            ) / len(target_results),
            "mean_counterfactual_delta_f1": (
                sum(counterfactual_delta_scores)
                / len(counterfactual_delta_scores)
            ),
            "mean_counterfactual_retracted_f1": (
                sum(counterfactual_retracted_scores)
                / len(counterfactual_retracted_scores)
            ),
            "earliest_noise_failure_after_clean_pass": (
                min(noise_failure_points)
                if noise_failure_points
                else None
            ),
        }

    oracle = _oracle_condition(
        induction=induction,
        heldout=heldout,
        mask_seed=7,
    )
    oracle_targets = {}
    for target_name, target_body in TARGETS.items():
        oracle_targets[target_name] = induce_condition(
            oracle,
            derive_hidden_target(induction, target_body),
            derive_hidden_target(heldout, target_body),
            admission_threshold=admission_threshold,
        )

    isolated_order = sorted(
        representation_results,
        key=lambda name: representation_results[name][
            "isolated_matched_f1"
        ],
        reverse=True,
    )
    compositional_order = sorted(
        representation_results,
        key=lambda name: representation_results[name][
            "mean_compositional_f1"
        ],
        reverse=True,
    )

    pca_isolated = representation_results["pca"][
        "isolated_matched_f1"
    ]
    kmeans_isolated = representation_results["kmeans"][
        "isolated_matched_f1"
    ]
    pca_composition = representation_results["pca"][
        "mean_compositional_f1"
    ]
    kmeans_composition = representation_results["kmeans"][
        "mean_compositional_f1"
    ]

    ordering_disagreement = (
        (kmeans_isolated > pca_isolated)
        and (pca_composition > kmeans_composition)
    )

    autoencoder = representation_results["autoencoder"]
    pca = representation_results["pca"]
    blocked_path_f1 = pca["targets"]["blocked_path"][
        "heldout_f1"
    ]
    if (
        autoencoder["mean_compositional_f1"]
        < pca["mean_compositional_f1"]
    ):
        next_hypothesis = (
            "Reconstruction-only latent learning is insufficient. The next "
            "representation objective should preserve anonymous algebraic "
            "structure (inverse/transposition and multi-hop closure) while "
            "also retaining sparse local-contact structure: PCA composes "
            f"directional relations but reaches only {blocked_path_f1:.3f} "
            "F1 on touching-to-above composition."
        )
    else:
        next_hypothesis = (
            "Learned reconstruction improves compositional utility; test "
            "object-centric slot features under the same zero-label evaluator "
            "to separate object binding from relation learning."
        )

    return {
        "config": {
            "train_frames": train_frames,
            "isolation_frames": isolation_frames,
            "induction_frames": induction_frames,
            "heldout_frames": heldout_frames,
            "train_seed": train_seed,
            "isolation_seed": isolation_seed,
            "induction_seed": induction_seed,
            "heldout_seed": heldout_seed,
            "random_seed": random_seed,
            "admission_threshold": admission_threshold,
            "semantic_labels_visible_to_representation_fit": False,
        },
        "targets": {
            name: list(body)
            for name, body in TARGETS.items()
        },
        "representations": representation_results,
        "oracle_targets": oracle_targets,
        "isolated_recovery_order": isolated_order,
        "compositional_utility_order": compositional_order,
        "pca_kmeans_ordering_disagreement": ordering_disagreement,
        "next_hypothesis_from_measured_failure": next_hypothesis,
    }
