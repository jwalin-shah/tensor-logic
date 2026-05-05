import torch

from experiments.exp84_support_data import generate_dataset
from experiments.exp86_support_baselines import (
    DeepSetsBaseline,
    PaddedMLP,
    masked_accuracy,
    run_baselines,
    tensorize_scenes,
    train_model,
)


def test_tensorize_scenes_pads_and_masks_variable_object_counts():
    scenes = [
        generate_dataset(1, split="id", seed=1, interventions=("none",))[0],
        generate_dataset(1, split="ood", seed=2, interventions=("remove",))[0],
    ]

    batch = tensorize_scenes(scenes)

    assert batch.features.shape[0] == 2
    assert batch.features.shape[2] == 5
    assert batch.labels.shape == batch.mask.shape
    assert batch.mask[0].sum().item() == len(scenes[0]["objects"])
    assert batch.mask[1].sum().item() == len(scenes[1]["objects"])
    assert torch.all(batch.labels[~batch.mask] == -100)


def test_models_return_per_object_logits():
    scenes = generate_dataset(2, split="id", seed=3, interventions=("none",))
    batch = tensorize_scenes(scenes, max_objects=3)

    mlp = PaddedMLP(max_objects=3)
    deepsets = DeepSetsBaseline()

    assert mlp(batch.features, batch.mask).shape == (2, 3, 2)
    assert deepsets(batch.features, batch.mask).shape == (2, 3, 2)


def test_masked_accuracy_ignores_padding():
    logits = torch.tensor([[[0.0, 2.0], [3.0, 0.0], [5.0, 0.0]]])
    labels = torch.tensor([[1, 1, -100]])
    mask = torch.tensor([[True, True, False]])

    metrics = masked_accuracy(logits, labels, mask)

    assert metrics == {"accuracy": 0.5, "correct": 1, "total": 2}


def test_training_reduces_loss_on_tiny_batch():
    scenes = generate_dataset(4, split="id", seed=4, interventions=("none", "remove"))
    batch = tensorize_scenes(scenes, max_objects=3)
    model = PaddedMLP(max_objects=3)

    losses = train_model(model, batch, epochs=8, lr=0.02)

    assert losses[-1] <= losses[0]


def test_run_baselines_quick_emits_expected_metric_schema():
    results = run_baselines(quick=True)

    assert results["quick"] is True
    for model_name in ("mlp", "deepsets"):
        assert set(results[model_name]) == {"id", "ood", "loss_start", "loss_end"}
        for split in ("id", "ood"):
            metrics = results[model_name][split]
            assert set(metrics) == {"accuracy", "correct", "total"}
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert metrics["total"] > 0
