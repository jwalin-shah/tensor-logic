import pytest

from experiments.exp84_support_data import compute_labels, generate_dataset


def _check_non_overlapping(scene):
    objs = scene["objects"]
    for i, a in enumerate(objs):
        for b in objs[i + 1 :]:
            x_overlap = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            y_overlap = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
            assert not (x_overlap > 0 and y_overlap > 0)


def test_id_generation_is_deterministic_and_valid():
    a = generate_dataset(num_scenes=5, split="id", seed=7)
    b = generate_dataset(num_scenes=5, split="id", seed=7)
    assert a == b
    for scene in a:
        assert len(scene["objects"]) in {2, 3}
        _check_non_overlapping(scene)
        assert set(scene["labels"].values()) <= {"stable", "falls"}


def test_ood_generation_shapes_and_ranges():
    ds = generate_dataset(num_scenes=8, split="ood", seed=11)
    for scene in ds:
        n = len(scene["objects"])
        assert 4 <= n <= 8
        _check_non_overlapping(scene)
        widths = [o["w"] for o in scene["objects"]]
        heights = [o["h"] for o in scene["objects"]]
        assert any(w not in {2.0, 3.0, 2.5, 1.5} for w in widths)
        assert any(h != 1.0 for h in heights)


def test_remove_intervention_retracts_support_chain():
    scene = {
        "objects": [
            {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
            {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
            {"id": "o2", "x": 0.0, "y": 2.0, "w": 2.0, "h": 1.0},
        ]
    }
    base = compute_labels(scene["objects"], {"type": "none"})
    assert base == {"o0": "stable", "o1": "stable", "o2": "stable"}
    removed = compute_labels(scene["objects"], {"type": "remove", "object_id": "o1"})
    assert removed["o1"] == "falls"
    assert removed["o2"] == "falls"
    assert removed["o0"] == "stable"


def test_branch_support_union_is_stable_then_falls_when_removed():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 1.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    labels = compute_labels(objects, {"type": "none"})
    assert labels["o2"] == "stable"
    labels2 = compute_labels(objects, {"type": "remove", "object_id": "o1"})
    assert labels2["o2"] == "falls"


def test_generated_intervention_samples_do_not_alias_objects():
    ds = generate_dataset(num_scenes=1, split="id", seed=1, interventions=("none", "remove"))

    assert len(ds) == 2
    assert ds[0]["objects"] is not ds[1]["objects"]
    assert ds[0]["objects"][0] is not ds[1]["objects"][0]

    original = ds[1]["objects"][0]["x"]
    ds[0]["objects"][0]["x"] = 999.0
    assert ds[1]["objects"][0]["x"] == original


def test_unknown_remove_target_rejected():
    objects = [{"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}]

    with pytest.raises(ValueError, match="unknown remove target"):
        compute_labels(objects, {"type": "remove", "object_id": "missing"})
