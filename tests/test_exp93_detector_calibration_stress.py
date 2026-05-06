from experiments.exp92_pixel_abstain_recover import detect_object_table, render_scene
from experiments.exp93_detector_calibration_stress import (
    DetectorStressConfig,
    apply_detector_stress,
    evaluate_stress_scene,
    run_detector_stress,
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


def test_missing_detector_failure_abstains_with_guard():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "missing", seed=1)
    records, _ = evaluate_stress_scene(scene, stressed, DetectorStressConfig(quick=True))
    by_status = {record["source_status"] for record in records}
    o1 = next(record for record in records if record["expected"] == "stable" and record["source_status"] == "detected")

    assert "missing_detector" in by_status
    assert o1["hard"] == "falls"
    assert o1["naive_accept"]
    assert not o1["guarded_accept"]


def test_merge_detector_failure_marks_source_ids():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "merge", seed=1)

    assert stressed.structural_failure
    assert stressed.affected_ids == ("o0", "o1")
    assert {record["id"] for record in stressed.objects} == {"merged_o0_o1"}
    assert stressed.source_status["o0"] == "merged_source"
    assert stressed.source_status["o1"] == "merged_source"


def test_false_positive_can_create_false_stable_naively():
    scene = _falling_top_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "false_positive", seed=1)
    records, _ = evaluate_stress_scene(scene, stressed, DetectorStressConfig(quick=True))
    o1 = next(record for record in records if record["object_id"] == "o1")

    assert stressed.false_positive_ids
    assert o1["hard"] == "stable"
    assert o1["naive_accept"]
    assert not o1["guarded_accept"]


def test_run_detector_stress_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = DetectorStressConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        localization_deltas=(0.0,),
        uncertainty_multipliers=(1.0,),
        failure_modes=("coordinate", "missing", "merge", "false_positive"),
    )

    results = run_detector_stress(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp93_detector_calibration_stress"
    assert set(results["detector_stress"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["detector_stress"]["modes"]["xy"]["all"]
    assert {row["failure_mode"] for row in xy_rows} == {
        "coordinate",
        "missing",
        "merge",
        "false_positive",
    }
    first = xy_rows[0]["uncertainty"][0]
    assert "naive_abstain_recover" in first
    assert "guarded_abstain_recover" in first
    assert "naive_false_stable" in first["triage"]
