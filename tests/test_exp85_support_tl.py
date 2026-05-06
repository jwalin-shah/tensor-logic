from experiments.exp84_support_data import compute_labels, generate_dataset
from experiments.exp85_support_tl import SupportTolerance, extract_primitives, infer_stability


def test_relation_extractor_hand_built_near_misses():
    objects = [
        {"id": "ground", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "top", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
        {"id": "gap", "x": 0.0, "y": 2.1, "w": 2.0, "h": 1.0},
        {"id": "side", "x": 3.0, "y": 1.0, "w": 1.0, "h": 1.0},
        {"id": "edge", "x": 2.0, "y": 1.0, "w": 1.0, "h": 1.0},
    ]

    rels = extract_primitives(objects)

    assert "ground" in rels.on_ground
    assert ("top", "ground") in rels.touching
    assert ("top", "ground") in rels.above
    assert ("top", "ground") in rels.horiz_overlap
    assert ("gap", "top") not in rels.touching
    assert ("side", "ground") not in rels.horiz_overlap
    assert ("side", "ground") not in rels.touching
    assert ("edge", "ground") not in rels.horiz_overlap
    assert ("edge", "ground") not in rels.touching


def test_single_block_on_ground_is_stable():
    objects = [{"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}]

    result = infer_stability(objects)

    assert result.labels == {"o0": "stable"}
    assert "on_ground" in result.proof_for("stable", "o0").format()


def test_simple_vertical_stack_and_deeper_chain_are_stable():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 2.0, "w": 2.0, "h": 1.0},
    ]

    result = infer_stability(objects)

    assert result.labels == {"o0": "stable", "o1": "stable", "o2": "stable"}
    assert ("o0", "o1") in result.supports
    assert ("o1", "o2") in result.supports
    assert "supports" in result.proof_for("stable", "o2").format()


def test_unsupported_floating_block_falls():
    objects = [{"id": "o0", "x": 0.0, "y": 2.0, "w": 1.0, "h": 1.0}]

    result = infer_stability(objects)

    assert result.labels == {"o0": "falls"}
    assert "not stable" in result.proof_for("falls", "o0").format()


def test_branching_support_requires_full_width_union():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 1.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]

    result = infer_stability(objects)
    retracted = infer_stability(objects, {"type": "remove", "object_id": "o1"})

    assert result.labels["o2"] == "stable"
    assert {("o0", "o2"), ("o1", "o2")} <= result.supports
    assert retracted.labels["o2"] == "falls"
    assert retracted.labels["o1"] == "falls"


def test_generator_parity_for_id_and_ood_samples():
    for split in ("id", "ood"):
        scenes = generate_dataset(num_scenes=8, split=split, seed=17)
        for scene in scenes:
            result = infer_stability(scene["objects"], scene["intervention"])
            assert result.labels == scene["labels"]


def test_generator_parity_on_removal_retraction_samples():
    scenes = generate_dataset(num_scenes=6, split="ood", seed=23, interventions=("remove",))
    for scene in scenes:
        result = infer_stability(scene["objects"], scene["intervention"])
        expected = compute_labels(scene["objects"], scene["intervention"])
        assert result.labels == expected
        removed_id = scene["intervention"]["object_id"]
        assert removed_id in result.primitives.removed
        assert result.labels[removed_id] == "falls"


def test_tolerant_extraction_recovers_tiny_vertical_contact_jitter():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.00005, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.00010, "w": 2.0, "h": 1.0},
    ]

    strict = infer_stability(objects)
    tolerant = infer_stability(objects, tolerance=SupportTolerance(contact=0.001, horizontal=0.001))

    assert strict.labels == {"o0": "falls", "o1": "falls"}
    assert tolerant.labels == {"o0": "stable", "o1": "stable"}
    assert ("o1", "o0") in tolerant.primitives.touching


def test_tolerant_extraction_recovers_tiny_horizontal_support_gap():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 0.9999, "h": 1.0},
        {"id": "o1", "x": 1.0001, "y": 0.0, "w": 0.9999, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]

    strict = infer_stability(objects)
    tolerant = infer_stability(objects, tolerance=SupportTolerance(contact=0.001, horizontal=0.001))

    assert strict.labels["o2"] == "falls"
    assert tolerant.labels["o2"] == "stable"
    assert {("o0", "o2"), ("o1", "o2")} <= tolerant.supports


def test_tolerant_extraction_rejects_obvious_gap():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 0.8, "h": 1.0},
        {"id": "o1", "x": 1.2, "y": 0.0, "w": 0.8, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]

    result = infer_stability(objects, tolerance=SupportTolerance(contact=0.001, horizontal=0.05))

    assert result.labels["o2"] == "falls"
