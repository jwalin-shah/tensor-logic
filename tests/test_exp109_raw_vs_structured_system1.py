from experiments.exp109_raw_vs_structured_system1 import run


def test_exp109_reports_paired_representation_metrics_without_assuming_winner():
    result = run(
        seed=59,
        iid_count=50,
        ood_per_family=3,
        epochs=15,
        raw_dim=96,
    )

    assert result["pair_count"] == 59
    assert len(result["dataset_digest"]) == 64

    for split in (
        "test",
        "ood_compositional",
        "ood_source_failure",
        "ood_confidence_shift",
    ):
        payload = result["splits"][split]
        assert "structured" in payload
        assert "raw_hashed" in payload
        assert "delta_structured_minus_raw" in payload
        assert 0.0 <= payload["structured"]["accuracy"] <= 1.0
        assert 0.0 <= payload["raw_hashed"]["accuracy"] <= 1.0
