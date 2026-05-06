from experiments.exp92_pixel_abstain_recover import detect_object_table, render_scene
from experiments.exp93_detector_calibration_stress import (
    DetectorStressConfig,
    apply_detector_stress,
    evaluate_stress_scene,
)
from experiments.exp94_object_hypothesis_layer import (
    ObjectHypothesisConfig,
    evaluate_object_hypothesis_scene,
    generate_object_hypotheses,
    run_object_hypothesis_layer,
)


def _stable_stack_scene():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    return {
        "objects": objects,
        "split": "unit",
        "intervention": {"type": "none"},
        "labels": {"o0": "stable", "o1": "stable"},
    }


def _falling_top_scene():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    return {
        "objects": objects,
        "split": "unit",
        "intervention": {"type": "remove", "object_id": "o0"},
        "labels": {"o0": "falls", "o1": "falls"},
    }


def _detected(scene):
    rendered = render_scene(scene["objects"], scale=100, padding=0.0)
    return detect_object_table(
        rendered,
        scene["objects"],
        localization_delta=0.0,
        uncertainty_multiplier=1.0,
        axes=("x", "y"),
        seed=123,
    )


def test_missing_hypothesis_restores_missing_source_object():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "missing", seed=1)
    hypotheses = generate_object_hypotheses(scene, stressed)
    repaired = next(hypothesis for hypothesis in hypotheses if hypothesis.name == "restore_missing")

    assert stressed.affected_ids == ("o0",)
    assert {str(obj["id"]) for obj in repaired.objects} == {"o0", "o1"}


def test_merge_hypothesis_splits_merged_object_into_sources():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "merge", seed=1)
    hypotheses = generate_object_hypotheses(scene, stressed)
    repaired = next(hypothesis for hypothesis in hypotheses if hypothesis.name == "split_merge")

    assert {str(obj["id"]) for obj in stressed.objects} == {"merged_o0_o1"}
    assert {str(obj["id"]) for obj in repaired.objects} == {"o0", "o1"}


def test_false_positive_hypothesis_drops_detector_spurious_support():
    scene = _falling_top_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "false_positive", seed=1)
    hypotheses = generate_object_hypotheses(scene, stressed)
    repaired = next(hypothesis for hypothesis in hypotheses if hypothesis.name == "drop_false_positive")

    assert stressed.false_positive_ids
    assert not set(stressed.false_positive_ids) & {str(obj["id"]) for obj in repaired.objects}


def test_hypothesis_route_recovers_where_guarded_route_abstains():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "missing", seed=1)
    exp93_records, _ = evaluate_stress_scene(scene, stressed, DetectorStressConfig(quick=True))
    records, _ = evaluate_object_hypothesis_scene(scene, stressed, ObjectHypothesisConfig(quick=True))
    o1_exp93 = next(record for record in exp93_records if record["object_id"] == "o1")
    o1 = next(record for record in records if record["object_id"] == "o1")

    assert o1_exp93["naive_accept"]
    assert not o1_exp93["guarded_accept"]
    assert o1["selected_hypothesis"] == "restore_missing"
    assert o1["hypothesis_accept"]
    assert o1["hypothesis_label"] == "stable"


def test_false_positive_hypothesis_rejects_false_stable_observation():
    scene = _falling_top_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "false_positive", seed=1)
    records, _ = evaluate_object_hypothesis_scene(scene, stressed, ObjectHypothesisConfig(quick=True))
    o1 = next(record for record in records if record["object_id"] == "o1")

    assert o1["observed_accept"]
    assert o1["observed_label"] == "stable"
    assert o1["selected_hypothesis"] == "drop_false_positive"
    assert o1["hypothesis_accept"]
    assert o1["hypothesis_label"] == "falls"


def test_run_object_hypothesis_layer_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = ObjectHypothesisConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        localization_deltas=(0.0,),
        uncertainty_multipliers=(1.0,),
        failure_modes=("missing", "merge", "false_positive"),
    )

    results = run_object_hypothesis_layer(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp94_object_hypothesis_layer"
    assert set(results["object_hypothesis_layer"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["object_hypothesis_layer"]["modes"]["xy"]["all"]
    assert {row["failure_mode"] for row in xy_rows} == {
        "missing",
        "merge",
        "false_positive",
    }
    first = xy_rows[0]["uncertainty"][0]
    assert "observed_naive" in first
    assert "guarded_structural_abstain" in first
    assert "object_hypothesis" in first
    assert "hypothesis_recovered_structural" in first["triage"]
