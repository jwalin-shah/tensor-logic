from tensor_logic.research.compositional_utility import (
    fit_autoencoder_relations,
    run_compositional_utility_benchmark,
)
from tensor_logic.research.unlabeled_relations import (
    generate_hidden_world_split,
)


def _small_benchmark():
    return run_compositional_utility_benchmark(
        train_frames=200,
        isolation_frames=100,
        induction_frames=60,
        heldout_frames=60,
        train_seed=11,
        isolation_seed=22,
        induction_seed=33,
        heldout_seed=44,
        random_seed=33,
        admission_threshold=0.70,
    )


def test_unlabeled_autoencoder_fit_is_deterministic():
    split = generate_hidden_world_split(
        n_frames=40,
        seed=11,
    )
    left = fit_autoencoder_relations(
        split.observations,
        steps=40,
        seed=123,
    )
    right = fit_autoencoder_relations(
        split.observations,
        steps=40,
        seed=123,
    )

    assert left.transform(split.observations).equal(
        right.transform(split.observations)
    )


def test_two_independent_oracle_compositions_are_recoverable():
    result = _small_benchmark()

    assert set(result["oracle_targets"]) == {
        "same_side",
        "blocked_path",
    }
    for target in result["oracle_targets"].values():
        assert target["accepted"] is True
        assert target["heldout_f1"] >= 0.95


def test_representation_report_separates_isolated_and_compositional_order():
    result = _small_benchmark()

    assert set(result["representations"]) == {
        "pca",
        "pca_whitened",
        "kmeans",
        "autoencoder",
        "random",
    }
    assert set(result["isolated_recovery_order"]) == set(
        result["representations"]
    )
    assert set(result["compositional_utility_order"]) == set(
        result["representations"]
    )

    for metrics in result["representations"].values():
        assert 0.0 <= metrics["isolated_matched_f1"] <= 1.0
        assert 0.0 <= metrics["mean_compositional_f1"] <= 1.0
        assert 0.0 <= metrics[
            "mean_counterfactual_delta_f1"
        ] <= 1.0
        assert 0.0 <= metrics[
            "mean_counterfactual_retracted_f1"
        ] <= 1.0
        assert "matched_brier" in metrics
        assert "cross_world_consistency" in metrics
        assert "earliest_noise_failure_after_clean_pass" in metrics


def test_counterfactual_metric_tracks_retraction_delta_not_sparse_accuracy():
    result = _small_benchmark()

    for metrics in result["representations"].values():
        for target in metrics["targets"].values():
            retraction = target["counterfactual_retraction"]
            assert set(retraction) == {
                "retracted_f1",
                "delta_f1",
                "active_retraction_worlds",
            }
            assert 0.0 <= retraction["delta_f1"] <= 1.0
            assert retraction["active_retraction_worlds"] >= 0


def test_semantic_labels_stay_out_of_representation_fit():
    result = _small_benchmark()

    assert result["config"][
        "semantic_labels_visible_to_representation_fit"
    ] is False

    for metrics in result["representations"].values():
        for target in metrics["targets"].values():
            candidate = target["candidate"]
            if candidate is None:
                continue
            assert all(
                relation.rstrip("^T").startswith("z_rel_")
                for relation in candidate
            )
            assert target[
                "candidate_semantics_evaluator_only"
            ] is None


def test_compositional_utility_benchmark_is_reproducible():
    first = _small_benchmark()
    second = _small_benchmark()

    assert first == second
