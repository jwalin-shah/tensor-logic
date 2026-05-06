from experiments.exp85_support_tl import SupportTolerance, infer_stability
from experiments.exp91_interval_support_uncertainty import (
    IntervalConfig,
    object_bands,
    possible_stability,
    run_interval_evaluation,
)


def test_object_bands_are_deterministic_and_axis_specific():
    objects = [{"id": "o0", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}]

    bands = object_bands(objects, x_radius=0.1, y_radius=0.2)

    assert bands["o0"].x == (0.9, 1.1)
    assert bands["o0"].y == (1.8, 2.2)
    assert bands["o0"].top == (5.8, 6.2)


def test_possible_stability_flags_contact_and_gap_uncertainty():
    tolerance = SupportTolerance(contact=0.001, horizontal=0.001)

    contact_jitter = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.01, "w": 2.0, "h": 1.0},
    ]
    assert infer_stability(contact_jitter, tolerance=tolerance).labels["o1"] == "falls"
    assert possible_stability(contact_jitter, y_radius=0.01, tolerance=tolerance)["o1"]

    support_gap = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 1.01, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    assert infer_stability(support_gap, tolerance=tolerance).labels["o2"] == "falls"
    assert possible_stability(support_gap, x_radius=0.01, tolerance=tolerance)["o2"]

    obvious_fall = [{"id": "o0", "x": 0.0, "y": 3.0, "w": 1.0, "h": 1.0}]
    assert not possible_stability(obvious_fall, y_radius=0.01, tolerance=tolerance)["o0"]


def test_run_interval_evaluation_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = IntervalConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        geometry_deltas=(0.0, 0.001),
        band_multipliers=(1.0, 2.0),
    )

    results = run_interval_evaluation(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp91_interval_support_uncertainty"
    assert set(results["interval_noise"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["interval_noise"]["modes"]["xy"]["all"]
    assert xy_rows[0]["delta"] == 0.0
    assert xy_rows[0]["bands"][0]["point"]["accuracy"] == 1.0
    assert xy_rows[0]["bands"][0]["interval_feasibility"]["coverage"] == 1.0
    assert xy_rows[1]["delta"] == 0.001
    assert {row["band_multiplier"] for row in xy_rows[1]["bands"]} == {1.0, 2.0}
    assert "interval_wrong_coverage" in xy_rows[1]["bands"][0]["triage"]
