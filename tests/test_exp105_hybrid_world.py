from experiments.exp105_hybrid_software_world import run


def test_hybrid_world_pipeline_derives_error_symbol_from_same_evidence():
    result = run()

    assert result["event_count"] == 2
    assert result["graph_nodes"] > 0
    assert result["graph_edges"] > 0
    assert result["tensor_coordinates"] > 0
    assert result["derived_error_symbol"] == [
        ("err:timeout", "sym:query")
    ]
    assert result["source_evidence"] == ["span:db"]
