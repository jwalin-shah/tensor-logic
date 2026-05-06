from experiments.exp85_support_tl import SupportTolerance, infer_stability
from experiments.exp90_support_repair_sweep import (
    RepairConfig,
    classify_failures,
    repair_objects,
    run_repair_evaluation,
)


def _config() -> RepairConfig:
    return RepairConfig(
        quick=True,
        eval_scenes=2,
        repair_contact_radius=0.02,
        repair_horizontal_radius=0.02,
    )


def test_classify_false_fall_missed_ground_contact_and_support_gap():
    config = _config()
    tolerance = SupportTolerance(contact=0.001, horizontal=0.001)

    missed_ground = [{"id": "o0", "x": 0.0, "y": 0.005, "w": 1.0, "h": 1.0}]
    ground_labels = infer_stability(missed_ground, tolerance=tolerance).labels
    assert classify_failures(missed_ground, None, {"o0": "stable"}, ground_labels, config) == {
        "o0": "false_fall_missed_ground"
    }

    missed_contact = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.005, "w": 2.0, "h": 1.0},
    ]
    contact_labels = infer_stability(missed_contact, tolerance=tolerance).labels
    assert classify_failures(
        missed_contact,
        None,
        {"o0": "stable", "o1": "stable"},
        contact_labels,
        config,
    )["o1"] == "false_fall_missed_contact"

    support_gap = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 1.01, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]
    gap_labels = infer_stability(support_gap, tolerance=tolerance).labels
    assert classify_failures(
        support_gap,
        None,
        {"o0": "stable", "o1": "stable", "o2": "stable"},
        gap_labels,
        config,
    )["o2"] == "false_fall_support_gap"


def test_repair_objects_recovers_contact_jitter_and_support_gap():
    config = _config()
    tolerance = SupportTolerance(contact=0.001, horizontal=0.001)
    contact_jitter = [
        {"id": "o0", "x": 0.0, "y": 0.005, "w": 2.0, "h": 1.0},
        {"id": "o1", "x": 0.0, "y": 1.010, "w": 2.0, "h": 1.0},
    ]

    repaired = repair_objects(contact_jitter, config=config)
    repaired_labels = infer_stability(repaired.objects, tolerance=tolerance).labels

    assert repaired_labels == {"o0": "stable", "o1": "stable"}
    assert [action.kind for action in repaired.actions] == ["snap_to_ground", "snap_to_support_top"]

    support_gap = [
        {"id": "o0", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o1", "x": 1.01, "y": 0.0, "w": 1.0, "h": 1.0},
        {"id": "o2", "x": 0.0, "y": 1.0, "w": 2.0, "h": 1.0},
    ]

    repaired_gap = repair_objects(support_gap, config=config)
    repaired_gap_labels = infer_stability(repaired_gap.objects, tolerance=tolerance).labels

    assert repaired_gap_labels["o2"] == "stable"
    assert "bridge_support_gap" in {action.kind for action in repaired_gap.actions}


def test_run_repair_evaluation_schema_and_zero_noise(tmp_path):
    output = tmp_path / "results.json"
    config = RepairConfig(
        quick=True,
        seed=123,
        eval_scenes=2,
        geometry_deltas=(0.0, 0.001),
    )

    results = run_repair_evaluation(config, output_path=output)

    assert output.exists()
    assert results["experiment"] == "exp90_support_repair_sweep"
    assert set(results["repair_noise"]["modes"]) == {"x_only", "y_only", "xy"}

    xy_rows = results["repair_noise"]["modes"]["xy"]["all"]
    assert xy_rows[0]["delta"] == 0.0
    assert xy_rows[0]["baseline"]["accuracy"] == 1.0
    assert xy_rows[0]["repaired"]["accuracy"] == 1.0
    assert xy_rows[0]["recovery"]["regressions"] == 0
    assert xy_rows[1]["delta"] == 0.001
    assert "taxonomy" in xy_rows[1]
    assert "repair_actions" in xy_rows[1]
