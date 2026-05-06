from experiments.exp85_support_tl import SupportTolerance, infer_stability
from experiments.exp92_pixel_abstain_recover import (
    PixelBenchmarkConfig,
    detect_object_table,
    render_scene,
    run_pixel_benchmark,
)
from experiments.exp91_interval_support_uncertainty import possible_stability


def test_render_scene_produces_label_pixels_and_boxes():
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 1.0, "h": 1.0},
    ]

    rendered = render_scene(objects, scale=8, padding=0.0, include_pixels=True)

    assert rendered.width == 8
    assert rendered.height == 16
    assert len(rendered.boxes) == 2
    assert rendered.pixels is not None
    labels = {value for row in rendered.pixels for value in row}
    assert labels == {1, 2}


def test_detector_emits_calibrated_uncertainty_bands():
    objects = [{"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}]
    rendered = render_scene(objects, scale=10, padding=0.0)

    detected = detect_object_table(
        rendered,
        objects,
        localization_delta=0.05,
        uncertainty_multiplier=2.0,
        axes=("x",),
        seed=123,
    )

    assert detected.objects[0]["id"] == "o0"
    assert detected.x_radius == 0.2
    assert detected.y_radius == 0.1
    assert detected.pixel_radius == 0.1


def test_interval_route_flags_pixel_localization_false_fall():
    tolerance = SupportTolerance(contact=0.001, horizontal=0.001)
    objects = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    rendered = render_scene(objects, scale=100, padding=0.0)
    detected = detect_object_table(
        rendered,
        objects,
        localization_delta=0.01,
        uncertainty_multiplier=1.0,
        axes=("y",),
        seed=1,
    )

    labels = infer_stability(detected.objects, tolerance=tolerance).labels
    possible = possible_stability(
        detected.objects,
        x_radius=detected.x_radius,
        y_radius=detected.y_radius,
        tolerance=tolerance,
    )

    assert labels["o1"] == "falls"
    assert possible["o1"]


def test_run_pixel_benchmark_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = PixelBenchmarkConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        localization_deltas=(0.0, 0.001),
        uncertainty_multipliers=(1.0,),
    )

    results = run_pixel_benchmark(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp92_pixel_abstain_recover"
    assert set(results["pixel_noise"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["pixel_noise"]["modes"]["xy"]["all"]
    assert xy_rows[0]["delta"] == 0.0
    assert xy_rows[0]["uncertainty"][0]["hard"]["accuracy"] < 1.0
    assert xy_rows[0]["uncertainty"][0]["interval_feasibility"]["accuracy"] == 1.0
    assert xy_rows[0]["uncertainty"][0]["triage"]["interval_wrong_coverage"] == 1.0
    assert xy_rows[1]["delta"] == 0.001
    assert "pipeline_accepted_wrong" in xy_rows[1]["uncertainty"][0]["triage"]
