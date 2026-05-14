import ast
from pathlib import Path

from experiments.runtime_paths import default_runtime_result_path


SUPPORT_RESULT_SCRIPTS = [
    Path("experiments/exp87_support_eval.py"),
    Path("experiments/exp88_support_noisy_relations.py"),
    Path("experiments/exp89_support_primitive_confidence.py"),
    Path("experiments/exp90_support_repair_sweep.py"),
    Path("experiments/exp91_interval_support_uncertainty.py"),
    Path("experiments/exp92_pixel_abstain_recover.py"),
    Path("experiments/exp93_detector_calibration_stress.py"),
    Path("experiments/exp94_object_hypothesis_layer.py"),
    Path("experiments/exp95_scored_object_hypotheses.py"),
]


def _assigns_default_runtime_path(script: Path) -> bool:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "output_path" for target in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == "default_runtime_result_path":
                return True
    return False


def test_default_runtime_result_path_stays_out_of_fixture_dirs():
    fixture_dir = Path("experiments/exp87_support_data")

    assert default_runtime_result_path(fixture_dir, quick=False) == Path(
        ".runtime/experiments/exp87_support_data/results.json"
    ).resolve()
    assert default_runtime_result_path(fixture_dir, quick=True) == Path(
        ".runtime/experiments/exp87_support_data/results_quick.json"
    ).resolve()


def test_support_experiment_defaults_use_ignored_runtime_paths():
    missing = [str(script) for script in SUPPORT_RESULT_SCRIPTS if not _assigns_default_runtime_path(script)]

    assert missing == []
