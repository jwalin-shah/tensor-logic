"""Composition benchmark over anonymous learned predicates.

Research OS v0-B asks whether rules can be induced over predicates whose names
and semantics are hidden from the conjecturer. The evaluator knows the hidden
world only to score held-out consequences and to construct falsification
controls.

No evaluator semantic mapping is passed into Conjecturer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from tensor_logic.research.rule_induction import (
    Config,
    Conjecturer,
    candidate_relation_names,
)
from tensor_logic.research.synthetic_scene import (
    N_OBJ,
    OBJ_IDX,
    PAIRS,
    RELATIONS,
)
from tensor_logic.research.unlabeled_relations import (
    HiddenWorldSplit,
    channels_to_world_tensors,
    fit_kmeans_relations,
    fit_pca_relations,
    generate_hidden_world_split,
    random_relation_channels,
)
from tensor_logic.research.utils import Schema, apply_body, f1 as compute_f1


DEFAULT_TARGET_NAME = "same_side"
DEFAULT_TARGET_BODY = ("left_of", "left_of")


@dataclass(frozen=True)
class AnonymousCondition:
    """One proposer-visible predicate world plus evaluator-only metadata."""

    name: str
    induction_base: dict[str, torch.Tensor]
    heldout_base: dict[str, torch.Tensor]
    evaluator_mapping: dict[str, str] | None = None


def hidden_relation_worlds(
    split: HiddenWorldSplit,
) -> list[dict[str, torch.Tensor]]:
    """Reconstruct semantic relation tensors for evaluator use only."""
    n_frames = int(split.observations.frame_ids.max().item()) + 1
    worlds = [
        {
            relation: torch.zeros(N_OBJ, N_OBJ)
            for relation in RELATIONS
        }
        for _ in range(n_frames)
    ]

    for sample in range(split.observations.n_samples):
        frame = int(split.observations.frame_ids[sample].item())
        pair_slot = int(split.observations.pair_slots[sample].item())
        a, b = PAIRS[pair_slot]
        ai, bi = OBJ_IDX[a], OBJ_IDX[b]
        for relation_idx, relation in enumerate(RELATIONS):
            if bool(split.evaluation.targets[sample, relation_idx].item()):
                worlds[frame][relation][ai, bi] = 1.0

    return worlds


def block_diagonal_worlds(
    worlds: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Combine independent small worlds without inventing cross-world edges."""
    if not worlds:
        raise ValueError("at least one world is required")
    names = tuple(worlds[0])
    return {
        name: torch.block_diag(
            *[world[name] for world in worlds]
        )
        for name in names
    }


def derive_hidden_target(
    split: HiddenWorldSplit,
    body: Iterable[str] = DEFAULT_TARGET_BODY,
) -> torch.Tensor:
    """Compute evaluator-only target consequences from hidden semantics."""
    body = list(body)
    per_world = []
    for world in hidden_relation_worlds(split):
        derived = apply_body(body, world)
        derived.fill_diagonal_(0)
        per_world.append(derived)
    return torch.block_diag(*per_world)


def anonymize_oracle_worlds(
    worlds: list[dict[str, torch.Tensor]],
    *,
    seed: int = 7,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, str]]:
    """Mask semantic names behind a fixed permutation of anonymous channels."""
    rng = np.random.default_rng(seed)
    anonymous_names = [f"z_rel_{idx}" for idx in range(len(RELATIONS))]
    permutation = rng.permutation(len(RELATIONS)).tolist()

    semantic_to_anon = {
        relation: anonymous_names[permutation[idx]]
        for idx, relation in enumerate(RELATIONS)
    }
    anon_to_semantic = {
        anonymous: semantic
        for semantic, anonymous in semantic_to_anon.items()
    }

    anonymous_worlds = []
    for world in worlds:
        anonymous_worlds.append(
            {
                semantic_to_anon[relation]: tensor.clone()
                for relation, tensor in world.items()
            }
        )
    return anonymous_worlds, anon_to_semantic


def learned_worlds(
    train: HiddenWorldSplit,
    split: HiddenWorldSplit,
    *,
    method: str,
) -> list[dict[str, torch.Tensor]]:
    """Fit only on observations and emit anonymous predicate tensors."""
    if method == "pca":
        model = fit_pca_relations(train.observations)
    elif method == "kmeans":
        model = fit_kmeans_relations(train.observations)
    else:
        raise ValueError(f"unknown learned method: {method}")

    probabilities = model.transform(split.observations)
    return channels_to_world_tensors(
        probabilities,
        split.observations,
    )


def random_worlds(
    split: HiddenWorldSplit,
    *,
    seed: int,
    n_channels: int = 4,
) -> list[dict[str, torch.Tensor]]:
    probabilities = random_relation_channels(
        split.observations,
        n_channels=n_channels,
        seed=seed,
    )
    return channels_to_world_tensors(
        probabilities,
        split.observations,
    )


def corrupt_worlds(
    worlds: list[dict[str, torch.Tensor]],
    *,
    noise: float,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Flip off-diagonal predicate bits for a controlled perception-noise test."""
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    rng = np.random.default_rng(seed)
    result = []
    for world in worlds:
        corrupted = {}
        for name, tensor in world.items():
            out = tensor.clone()
            for row in range(out.shape[0]):
                for col in range(out.shape[1]):
                    if row == col:
                        out[row, col] = 0
                        continue
                    if rng.random() < noise:
                        out[row, col] = 1.0 - out[row, col]
            corrupted[name] = out
        result.append(corrupted)
    return result


def within_world_examples(
    target: torch.Tensor,
    *,
    block_size: int = N_OBJ,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Use only within-world pairs, avoiding trivial cross-world negatives."""
    positive: list[tuple[int, int]] = []
    negative: list[tuple[int, int]] = []
    n_blocks = target.shape[0] // block_size

    for block in range(n_blocks):
        offset = block * block_size
        for row in range(block_size):
            for col in range(block_size):
                if row == col:
                    continue
                pair = (offset + row, offset + col)
                if target[pair[0], pair[1]] > 0:
                    positive.append(pair)
                else:
                    negative.append(pair)
    return positive, negative


def _config_for_base(
    name: str,
    base: dict[str, torch.Tensor],
) -> Config:
    n_entities = next(iter(base.values())).shape[0]
    schema = Schema(
        name,
        {
            relation: ("person", "person")
            for relation in base
        },
    )
    return Config(
        name=name,
        schema=schema,
        n_pos=1000,
        n_neg=1000,
        noise=0.0,
        n_entities=n_entities,
        max_steps=1,
        min_equiv=0.95,
        f1_threshold=0.70,
        n_attempts=1,
        vote_threshold=0.50,
        min_support=1,
    )


def first_counterexample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    block_size: int = N_OBJ,
) -> dict | None:
    """Return one within-world disagreement as a concrete falsifier."""
    n_blocks = target.shape[0] // block_size
    for block in range(n_blocks):
        offset = block * block_size
        for row in range(block_size):
            for col in range(block_size):
                if row == col:
                    continue
                i, j = offset + row, offset + col
                predicted = int(prediction[i, j] > 0)
                expected = int(target[i, j] > 0)
                if predicted != expected:
                    return {
                        "world": block,
                        "src": row,
                        "dst": col,
                        "predicted": predicted,
                        "expected": expected,
                    }
    return None


def evaluate_body(
    body: list[str] | None,
    heldout_base: dict[str, torch.Tensor],
    heldout_target: torch.Tensor,
    *,
    threshold: float = 0.70,
) -> dict:
    """Deterministic held-out admission gate."""
    if not body:
        return {
            "accepted": False,
            "heldout_f1": 0.0,
            "counterexample": {"reason": "no_candidate"},
        }

    prediction = apply_body(body, heldout_base)
    prediction.fill_diagonal_(0)
    heldout_f1 = compute_f1(prediction, heldout_target)
    return {
        "accepted": heldout_f1 >= threshold,
        "heldout_f1": heldout_f1,
        "counterexample": first_counterexample(
            prediction,
            heldout_target,
        ),
    }


def induce_condition(
    condition: AnonymousCondition,
    induction_target: torch.Tensor,
    heldout_target: torch.Tensor,
    *,
    admission_threshold: float = 0.70,
) -> dict:
    """Run the reusable #22 Conjecturer, then independently score held-out data."""
    cfg = _config_for_base(condition.name, condition.induction_base)
    relation_names = candidate_relation_names(cfg.schema)
    conjecturer = Conjecturer(
        cfg,
        condition.induction_base,
        relation_names,
        max_len=2,
    )
    positive, negative = within_world_examples(induction_target)
    proposal = conjecturer.propose_from_examples(
        positive,
        negative,
        target=induction_target,
    )
    heldout = evaluate_body(
        proposal["body"],
        condition.heldout_base,
        heldout_target,
        threshold=admission_threshold,
    )

    evaluator_semantics = None
    if condition.evaluator_mapping and proposal["body"]:
        evaluator_semantics = [
            condition.evaluator_mapping.get(
                relation.rstrip("^T"),
                "unknown",
            )
            + ("^T" if relation.endswith("^T") else "")
            for relation in proposal["body"]
        ]

    return {
        "candidate": proposal["body"],
        "candidate_semantics_evaluator_only": evaluator_semantics,
        "induction_f1": proposal["f1"],
        "example_f1": proposal["example_f1"],
        **heldout,
    }


def _condition_from_worlds(
    name: str,
    induction_worlds: list[dict[str, torch.Tensor]],
    heldout_worlds: list[dict[str, torch.Tensor]],
    *,
    evaluator_mapping: dict[str, str] | None = None,
) -> AnonymousCondition:
    return AnonymousCondition(
        name=name,
        induction_base=block_diagonal_worlds(induction_worlds),
        heldout_base=block_diagonal_worlds(heldout_worlds),
        evaluator_mapping=evaluator_mapping,
    )


def run_anonymous_composition_benchmark(
    *,
    train_frames: int = 200,
    induction_frames: int = 80,
    heldout_frames: int = 80,
    train_seed: int = 11,
    induction_seed: int = 22,
    heldout_seed: int = 44,
    oracle_mask_seed: int = 7,
    random_seed: int = 99,
    admission_threshold: float = 0.70,
    target_body: tuple[str, str] = DEFAULT_TARGET_BODY,
) -> dict:
    """Compare oracle, learned, random, and noisy predicates under one rule gate."""
    if len({train_seed, induction_seed, heldout_seed}) != 3:
        raise ValueError("train, induction, and heldout seeds must differ")

    train = generate_hidden_world_split(
        n_frames=train_frames,
        seed=train_seed,
    )
    induction = generate_hidden_world_split(
        n_frames=induction_frames,
        seed=induction_seed,
    )
    heldout = generate_hidden_world_split(
        n_frames=heldout_frames,
        seed=heldout_seed,
    )

    induction_target = derive_hidden_target(
        induction,
        target_body,
    )
    heldout_target = derive_hidden_target(
        heldout,
        target_body,
    )

    induction_hidden = hidden_relation_worlds(induction)
    heldout_hidden = hidden_relation_worlds(heldout)
    oracle_induction, mapping = anonymize_oracle_worlds(
        induction_hidden,
        seed=oracle_mask_seed,
    )
    oracle_heldout, heldout_mapping = anonymize_oracle_worlds(
        heldout_hidden,
        seed=oracle_mask_seed,
    )
    if mapping != heldout_mapping:
        raise AssertionError("oracle anonymization drifted across splits")

    conditions = {
        "oracle": _condition_from_worlds(
            "oracle",
            oracle_induction,
            oracle_heldout,
            evaluator_mapping=mapping,
        ),
        "pca": _condition_from_worlds(
            "pca",
            learned_worlds(train, induction, method="pca"),
            learned_worlds(train, heldout, method="pca"),
        ),
        "kmeans": _condition_from_worlds(
            "kmeans",
            learned_worlds(train, induction, method="kmeans"),
            learned_worlds(train, heldout, method="kmeans"),
        ),
        "random": _condition_from_worlds(
            "random",
            random_worlds(induction, seed=random_seed),
            random_worlds(heldout, seed=random_seed + 1),
        ),
    }

    results = {
        name: induce_condition(
            condition,
            induction_target,
            heldout_target,
            admission_threshold=admission_threshold,
        )
        for name, condition in conditions.items()
    }

    noise_results = {}
    for index, noise in enumerate((0.0, 0.05, 0.10, 0.20, 0.30)):
        noisy_induction = corrupt_worlds(
            oracle_induction,
            noise=noise,
            seed=500 + index,
        )
        noisy_heldout = corrupt_worlds(
            oracle_heldout,
            noise=noise,
            seed=700 + index,
        )
        condition = _condition_from_worlds(
            f"noise_{noise:.2f}",
            noisy_induction,
            noisy_heldout,
            evaluator_mapping=mapping,
        )
        noise_results[f"{noise:.2f}"] = induce_condition(
            condition,
            induction_target,
            heldout_target,
            admission_threshold=admission_threshold,
        )

    semantic_to_anon = {
        semantic: anonymous
        for anonymous, semantic in mapping.items()
    }
    deliberately_false = [
        semantic_to_anon["touching"],
        semantic_to_anon["touching"],
    ]
    false_result = evaluate_body(
        deliberately_false,
        conditions["oracle"].heldout_base,
        heldout_target,
        threshold=admission_threshold,
    )

    return {
        "config": {
            "train_frames": train_frames,
            "induction_frames": induction_frames,
            "heldout_frames": heldout_frames,
            "train_seed": train_seed,
            "induction_seed": induction_seed,
            "heldout_seed": heldout_seed,
            "oracle_mask_seed": oracle_mask_seed,
            "random_seed": random_seed,
            "admission_threshold": admission_threshold,
            "target_name": DEFAULT_TARGET_NAME,
            "target_body_evaluator_only": list(target_body),
        },
        "conditions": results,
        "noise_sweep": noise_results,
        "false_rule_control": {
            "body": deliberately_false,
            **false_result,
        },
        "evaluator_only_oracle_mapping": mapping,
    }
