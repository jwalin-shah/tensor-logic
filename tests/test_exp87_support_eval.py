import json

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
