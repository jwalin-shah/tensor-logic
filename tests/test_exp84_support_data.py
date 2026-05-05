import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp84_support_data import generate_dataset


def _interior_overlap(a, b) -> bool:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    x_overlap = min(ax2, bx2) - max(a["x"], b["x"])
    y_overlap = min(ay2, by2) - max(a["y"], b["y"])
    return x_overlap > 0 and y_overlap > 0


def test_id_split_shape_and_reproducibility():
    d1 = generate_dataset(n_scenes=12, split="id", intervention="none", seed=11)
    d2 = generate_dataset(n_scenes=12, split="id", intervention="none", seed=11)
    assert d1 == d2

    for scene in d1:
        objs = scene["objects"]
        assert 2 <= len(objs) <= 3
        for obj in objs:
            assert set(obj.keys()) == {"id", "x", "y", "w", "h"}
            assert obj["w"] >= 2 and obj["w"] <= 4
            assert obj["h"] >= 1 and obj["h"] <= 2


def test_ood_split_shape_and_unseen_sizes():
    data = generate_dataset(n_scenes=40, split="ood", intervention="none", seed=3)
    saw_large_width = False
    saw_large_height = False

    for scene in data:
        objs = scene["objects"]
        assert 4 <= len(objs) <= 8
        for obj in objs:
            if obj["w"] > 4:
                saw_large_width = True
            if obj["h"] > 2:
                saw_large_height = True

    assert saw_large_width
    assert saw_large_height


def test_remove_intervention_produces_consistent_labels():
    data = generate_dataset(n_scenes=20, split="ood", intervention="remove", seed=7)
    for scene in data:
        iv = scene["intervention"]
        assert iv["kind"] == "remove"
        assert iv["object_id"] is not None
        removed = iv["object_id"]
        object_ids = {o["id"] for o in scene["objects"]}
        assert removed in object_ids
        assert removed not in scene["labels"]
        assert set(scene["labels"].values()) <= {"stable", "falls"}


def test_rectangles_do_not_overlap_interiorly():
    data = generate_dataset(n_scenes=60, split="ood", intervention="none", seed=99)
    for scene in data:
        objs = scene["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                assert not _interior_overlap(objs[i], objs[j])
