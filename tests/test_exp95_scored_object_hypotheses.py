from dataclasses import replace

from experiments.exp92_pixel_abstain_recover import detect_object_table, render_scene
from experiments.exp93_detector_calibration_stress import apply_detector_stress
from experiments.exp95_scored_object_hypotheses import (
    ScoredHypothesisConfig,
    evaluate_scored_hypothesis_scene,
    generate_non_oracle_hypotheses,
    rank_non_oracle_hypotheses,
    run_scored_object_hypotheses,
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


def test_missing_candidates_do_not_need_affected_ids():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "missing", seed=1)
    no_oracle = replace(stressed, affected_ids=())

    hypotheses = generate_non_oracle_hypotheses(scene, no_oracle, ScoredHypothesisConfig(quick=True))

    assert any(hypothesis.repair_kind == "missing_support_added" for hypothesis in hypotheses)
    assert all("o0" not in {str(obj["id"]) for obj in hypothesis.objects} for hypothesis in hypotheses)


def test_merge_candidates_do_not_need_source_ids():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "merge", seed=1)
    no_oracle_objects = [
        {key: value for key, value in obj.items() if key != "source_ids"}
        for obj in stressed.objects
    ]
    no_oracle = replace(stressed, objects=no_oracle_objects, affected_ids=())

    hypotheses = generate_non_oracle_hypotheses(scene, no_oracle, ScoredHypothesisConfig(quick=True))
    split = next(hypothesis for hypothesis in hypotheses if hypothesis.repair_kind == "merge_split_candidate")

    assert {str(obj["id"]) for obj in split.objects} == {
        "hyp_split_merged_o0_o1_lower",
        "hyp_split_merged_o0_o1_upper",
    }


def test_false_positive_candidates_do_not_need_false_positive_ids():
    scene = _falling_top_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "false_positive", seed=1)
    no_oracle = replace(stressed, false_positive_ids=(), affected_ids=())

    hypotheses = generate_non_oracle_hypotheses(scene, no_oracle, ScoredHypothesisConfig(quick=True))

    assert any(hypothesis.repair_kind == "false_positive_drop_candidate" for hypothesis in hypotheses)
    assert len(hypotheses) > 2


def test_scored_missing_candidate_recovers_supported_object_without_source_identity():
    scene = _stable_stack_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "missing", seed=1)
    ranked = rank_non_oracle_hypotheses(scene, stressed, ScoredHypothesisConfig(quick=True))
    records, _ = evaluate_scored_hypothesis_scene(scene, stressed, ScoredHypothesisConfig(quick=True))
    o0 = next(record for record in records if record["object_id"] == "o0")
    o1 = next(record for record in records if record["object_id"] == "o1")

    assert ranked[0].hypothesis.repair_kind == "missing_support_added"
    assert not o0["scored_accept"]
    assert o1["scored_accept"]
    assert o1["scored_label"] == "stable"


def test_scored_false_positive_candidate_ranks_spurious_support_drop():
    scene = _falling_top_scene()
    stressed = apply_detector_stress(scene, _detected(scene), "false_positive", seed=1)
    ranked = rank_non_oracle_hypotheses(scene, stressed, ScoredHypothesisConfig(quick=True))
    records, _ = evaluate_scored_hypothesis_scene(scene, stressed, ScoredHypothesisConfig(quick=True))
    o1 = next(record for record in records if record["object_id"] == "o1")

    assert ranked[0].hypothesis.repair_kind == "false_positive_drop_candidate"
    assert o1["observed_accept"]
    assert o1["observed_label"] == "stable"
    assert o1["scored_accept"]
    assert o1["scored_label"] == "falls"


def test_run_scored_object_hypotheses_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = ScoredHypothesisConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        localization_deltas=(0.0,),
        uncertainty_multipliers=(1.0,),
        failure_modes=("missing", "merge", "false_positive"),
    )

    results = run_scored_object_hypotheses(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp95_scored_object_hypotheses"
    assert set(results["scored_object_hypotheses"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["scored_object_hypotheses"]["modes"]["xy"]["all"]
    assert {row["failure_mode"] for row in xy_rows} == {
        "missing",
        "merge",
        "false_positive",
    }
    first = xy_rows[0]["uncertainty"][0]
    assert "oracle_object_hypothesis" in first
    assert "scored_non_oracle" in first
    assert "scored_recovery_gap_vs_oracle" in first["triage"]
