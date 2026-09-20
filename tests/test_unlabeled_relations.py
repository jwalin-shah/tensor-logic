import torch

from tensor_logic.research.unlabeled_relations import (
    channels_to_world_tensors,
    fit_pca_relations,
    generate_hidden_world_split,
    run_unlabeled_benchmark,
)


def test_learner_boundary_exposes_only_observations():
    split = generate_hidden_world_split(n_frames=8, seed=11)
    observations = split.observations

    assert set(observations.__dataclass_fields__) == {
        "features",
        "frame_ids",
        "pair_slots",
    }
    assert observations.features.shape[1] == 7
    assert observations.n_samples == 8 * 6


def test_train_and_test_worlds_are_seed_separated_and_deterministic():
    train_a = generate_hidden_world_split(n_frames=10, seed=11)
    train_b = generate_hidden_world_split(n_frames=10, seed=11)
    test = generate_hidden_world_split(n_frames=10, seed=22)

    assert torch.equal(
        train_a.observations.features,
        train_b.observations.features,
    )
    assert torch.equal(
        train_a.evaluation.targets,
        train_b.evaluation.targets,
    )
    assert not torch.equal(
        train_a.observations.features,
        test.observations.features,
    )


def test_pca_channels_are_fixed_seed_reproducible():
    train = generate_hidden_world_split(n_frames=40, seed=11)
    test = generate_hidden_world_split(n_frames=20, seed=22)

    left = fit_pca_relations(train.observations).transform(
        test.observations
    )
    right = fit_pca_relations(train.observations).transform(
        test.observations
    )
    assert torch.equal(left, right)


def test_anonymous_channels_convert_to_tensor_logic_relations():
    train = generate_hidden_world_split(n_frames=30, seed=11)
    test = generate_hidden_world_split(n_frames=5, seed=22)
    probs = fit_pca_relations(train.observations).transform(
        test.observations
    )

    worlds = channels_to_world_tensors(
        probs,
        test.observations,
        n_frames=5,
    )

    assert len(worlds) == 5
    assert set(worlds[0]) == {
        "z_rel_0",
        "z_rel_1",
        "z_rel_2",
        "z_rel_3",
    }
    for tensor in worlds[0].values():
        assert tensor.shape == (3, 3)


def test_unlabeled_pca_beats_random_baseline_on_hidden_world():
    result = run_unlabeled_benchmark(
        train_frames=200,
        test_frames=100,
        train_seed=11,
        test_seed=22,
        random_seed=33,
    )

    pca = result["methods"]["pca"]["mean_matched_f1"]
    random = result["methods"]["random"]["mean_matched_f1"]

    assert pca > random
    assert result["pca_minus_random"] >= 0.10
