import json

import pytest

import experiments.exp87_support_eval as exp87
from experiments.exp87_support_eval import EvalConfig, make_splits, run_evaluation


def test_make_splits_covers_required_eval_distributions():
    config = EvalConfig(quick=True, train_scenes=2, eval_scenes=2, epochs=1, hidden_dim=8)
    train, eval_splits = make_splits(config)

    assert train
    assert set(eval_splits) == {"id", "larger_ood", "deeper_ood", "branching_ood", "counterfactual"}
    assert {scene["intervention"]["type"] for scene in eval_splits["counterfactual"]} == {"remove"}
    assert max(len(scene["objects"]) for scene in eval_splits["id"]) <= 3
    assert min(len(scene["objects"]) for scene in eval_splits["larger_ood"]) >= 4
    assert min(len(scene["objects"]) for scene in eval_splits["deeper_ood"]) >= 5
    assert min(len(scene["objects"]) for scene in eval_splits["branching_ood"]) >= 5


def test_run_evaluation_writes_expected_results_schema(tmp_path):
    output = tmp_path / "results.json"
    config = EvalConfig(quick=True, train_scenes=3, eval_scenes=2, epochs=2, hidden_dim=8)

    results = run_evaluation(config, output_path=output)
    on_disk = json.loads(output.read_text())

    assert on_disk == results
    assert results["experiment"] == "exp87_support_eval"
    assert set(results["tl"]) == {"id", "larger_ood", "deeper_ood", "branching_ood", "counterfactual"}
    for model in ("mlp", "deepsets"):
        assert "loss_start" in results[model]
        assert "loss_end" in results[model]
        for split in results["tl"]:
            metrics = results[model][split]
            assert set(metrics) == {"accuracy", "correct", "total"}
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert metrics["total"] > 0

    gates = results["gates"]
    assert gates["tl_deterministic_label_accuracy_100"]["passed"] is True
    assert gates["tl_counterfactual_retraction_accuracy_100"]["passed"] is True
    assert set(gates["tl_ood_margin_vs_best_neural_at_least_10pp"]["splits"]) == {"larger_ood", "deeper_ood"}


def test_run_evaluation_raises_when_ood_margin_gate_fails(monkeypatch, tmp_path):
    output = tmp_path / "results.json"
    config = EvalConfig(quick=True, train_scenes=2, eval_scenes=1, epochs=1, hidden_dim=8)

    def fake_train_and_eval_baselines(train, eval_splits, config):
        del train, config
        baseline = {"loss_start": 0.0, "loss_end": 0.0}
        for split, scenes in eval_splits.items():
            total = sum(len(scene["objects"]) for scene in scenes)
            baseline[split] = {"accuracy": 1.0, "correct": total, "total": total}
        return {"mlp": dict(baseline), "deepsets": dict(baseline), "metadata": {"max_objects": 8}}

    def fake_build_gates(tl, baselines):
        del tl, baselines
        return {
            "tl_deterministic_label_accuracy_100": {
                "passed": True,
                "accuracy": 1.0,
                "correct": 1,
                "total": 1,
                "threshold": 1.0,
            },
            "tl_counterfactual_retraction_accuracy_100": {
                "passed": True,
                "accuracy": 1.0,
                "correct": 1,
                "total": 1,
                "threshold": 1.0,
            },
            "tl_ood_margin_vs_best_neural_at_least_10pp": {
                "passed": False,
                "threshold": 0.10,
                "splits": {
                    "larger_ood": {
                        "tl_accuracy": 1.0,
                        "best_neural": "mlp",
                        "best_neural_accuracy": 0.95,
                        "margin": 0.05,
                        "passed": False,
                    },
                    "deeper_ood": {
                        "tl_accuracy": 1.0,
                        "best_neural": "deepsets",
                        "best_neural_accuracy": 0.85,
                        "margin": 0.15,
                        "passed": True,
                    },
                },
            },
            "v1_passed": False,
        }

    monkeypatch.setattr(exp87, "_train_and_eval_baselines", fake_train_and_eval_baselines)
    monkeypatch.setattr(exp87, "_build_gates", fake_build_gates)

    with pytest.raises(RuntimeError, match=r"OOD margin.*larger_ood.*margin=0\.050.*threshold=0\.100"):
        run_evaluation(config, output_path=output)

    assert not output.exists()
