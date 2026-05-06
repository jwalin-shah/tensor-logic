from experiments.exp85_support_tl import SupportTolerance
from experiments.exp89_support_primitive_confidence import (
    ConfidenceConfig,
    prediction_confidences,
    run_confidence_evaluation,
    stability_confidences,
)


def test_stability_confidence_is_deterministic_and_margin_sensitive():
    clean = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    jittered = [
        {"id": "o0", "x": 0.0, "y": 0.0008, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0015, "w": 2.0, "h": 1.0},
    ]
    tolerance = SupportTolerance(contact=0.001, horizontal=0.001)

    clean_scores = stability_confidences(clean, tolerance=tolerance)
    jittered_a = stability_confidences(jittered, tolerance=tolerance)
    jittered_b = stability_confidences(jittered, tolerance=tolerance)

    assert clean_scores == {"o0": 1.0, "o1": 1.0}
    assert jittered_a == jittered_b
    assert 0.0 < jittered_a["o0"] < clean_scores["o0"]
    assert 0.0 < jittered_a["o1"] < clean_scores["o1"]


def test_prediction_confidence_keeps_obvious_fall_high_confidence():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 2.0, "y": 1.0, "w": 1.0, "h": 1.0},
    ]

    confidences = prediction_confidences(
        objects,
        tolerance=SupportTolerance(contact=0.001, horizontal=0.001),
    )

    assert confidences["o0"] == 1.0
    assert confidences["o1"] == 1.0


def test_run_confidence_evaluation_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = ConfidenceConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        geometry_deltas=(0.0, 0.0001),
    )

    results = run_confidence_evaluation(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp89_support_primitive_confidence"
    assert set(results["confidence_noise"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["confidence_noise"]["modes"]["xy"]["all"]
    assert xy_rows[0]["delta"] == 0.0
    assert xy_rows[0]["overall"]["accuracy"] == 1.0
    assert xy_rows[0]["buckets"]["high"]["total"] > 0
    assert xy_rows[0]["coverage_at_thresholds"][0]["threshold"] == 0.0
    assert xy_rows[1]["delta"] == 0.0001
    assert set(xy_rows[1]["buckets"]) == {"low", "medium", "high"}
