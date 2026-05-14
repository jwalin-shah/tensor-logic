from pathlib import Path
import subprocess

import experiments.exp78_rule_induction as exp78
import experiments.exp79_lewm_tl as exp79
import experiments.exp87_support_eval as exp87
import experiments.exp88_support_noisy_relations as exp88
import experiments.exp89_support_primitive_confidence as exp89
import experiments.exp90_support_repair_sweep as exp90
import experiments.exp91_interval_support_uncertainty as exp91
import experiments.exp92_pixel_abstain_recover as exp92
import experiments.exp93_detector_calibration_stress as exp93
import experiments.exp94_object_hypothesis_layer as exp94
import experiments.exp95_scored_object_hypotheses as exp95
from experiments.runtime_paths import RUNTIME_OUTPUT_ROOT, result_path


def test_runtime_output_root_is_git_ignored():
    candidate = RUNTIME_OUTPUT_ROOT / "example" / "results.json"

    completed = subprocess.run(
        ["git", "check-ignore", str(candidate)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "runtime_outputs/" in completed.stdout


def test_default_runtime_paths_do_not_target_tracked_fixtures():
    experiment_names = [
        exp78.EXPERIMENT_NAME,
        exp87.EXPERIMENT_NAME,
        exp88.EXPERIMENT_NAME,
        exp89.EXPERIMENT_NAME,
        exp90.EXPERIMENT_NAME,
        exp91.EXPERIMENT_NAME,
        exp92.EXPERIMENT_NAME,
        exp93.EXPERIMENT_NAME,
        exp94.EXPERIMENT_NAME,
        exp95.EXPERIMENT_NAME,
    ]

    for experiment_name in experiment_names:
        for quick in (False, True):
            path = result_path(experiment_name, quick=quick)
            assert path.is_relative_to(RUNTIME_OUTPUT_ROOT)
            assert "_data" not in path.parts


def test_training_experiment_output_dirs_default_to_runtime_tree():
    for output_dir in (Path(exp79.DATA_DIR),):
        assert output_dir.is_relative_to(RUNTIME_OUTPUT_ROOT)
        assert "_data" not in output_dir.parts

    exp83_source = Path("experiments/exp83_slot_attention.py").read_text(encoding="utf-8")
    assert 'DATA_DIR = str(experiment_output_dir("exp83_slot_attention"))' in exp83_source


def test_data_generators_default_to_runtime_tree():
    default_markers = {
        "experiments/exp60a_kinship_traces.py": 'experiment_output_dir("exp60a_kinship_traces")',
        "experiments/exp60d_sft.py": 'experiment_output_dir("exp60d_sft")',
        "experiments/exp76a_rule_traces.py": 'experiment_output_dir("exp76a_rule_traces")',
        "experiments/exp76b_hard_eval.py": 'experiment_output_dir("exp76b_hard_eval")',
        "experiments/exp76c_paraphrase_train.py": 'experiment_output_dir("exp76c_paraphrase_train")',
    }

    for script, marker in default_markers.items():
        assert marker in Path(script).read_text(encoding="utf-8")
