from experiments.exp85_support_tl import extract_primitives, infer_stability
from experiments.exp88_support_noisy_relations import (
    NoiseConfig,
    flip_primitives,
    infer_labels_from_primitives,
    object_accuracy,
    perturb_objects,
    run_noise_evaluation,
)


def test_perturb_objects_is_deterministic_and_keeps_ids():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]

    assert perturb_objects(objects, delta=0.0, seed=1) == objects
    a = perturb_objects(objects, delta=0.1, seed=7, axes=("x", "y"))
    b = perturb_objects(objects, delta=0.1, seed=7, axes=("x", "y"))
    c = perturb_objects(objects, delta=0.1, seed=8, axes=("x", "y"))

    assert a == b
    assert a != c
    assert [obj["id"] for obj in a] == ["o0", "o1"]
    assert a[0]["w"] == objects[0]["w"]
    assert a[0]["h"] == objects[0]["h"]


def test_flip_primitives_has_controlled_zero_and_one_probability():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    primitives = extract_primitives(objects)

    assert flip_primitives(primitives, ["o0", "o1"], probability=0.0, seed=1) == primitives

    flipped = flip_primitives(primitives, ["o0", "o1"], probability=1.0, seed=1)
    assert ("o1", "o0") not in flipped.touching
    assert ("o0", "o1") in flipped.touching
    assert "o0" not in flipped.on_ground
    assert "o1" in flipped.on_ground
    assert flipped.removed == primitives.removed


def test_infer_labels_from_primitives_matches_clean_engine_without_flips():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    primitives = extract_primitives(objects)

    assert infer_labels_from_primitives(objects, primitives) == infer_stability(objects).labels


def test_object_accuracy_counts_per_object_labels():
    scenes = [
        {"labels": {"o0": "stable", "o1": "falls"}},
        {"labels": {"o0": "stable"}},
    ]
    predictions = [
        {"o0": "stable", "o1": "stable"},
        {"o0": "stable"},
    ]

    assert object_accuracy(scenes, predictions) == {"accuracy": 2 / 3, "correct": 2, "total": 3}


def test_run_noise_evaluation_schema_and_first_drop(tmp_path):
    output = tmp_path / "results.json"
    config = NoiseConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        geometry_deltas=(0.0, 0.001),
        relation_flip_probabilities=(0.0, 1.0),
    )

    results = run_noise_evaluation(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp88_support_noisy_relations"
    assert set(results["geometry_noise"]["modes"]) == {"x_only", "y_only", "xy"}
    assert results["geometry_noise"]["modes"]["y_only"]["first_below_100"]["all"]["delta"] == 0.001
    assert results["relation_flip_noise"]["first_below_100"]["all"]["probability"] == 1.0
