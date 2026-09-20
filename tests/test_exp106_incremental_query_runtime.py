from experiments.exp106_incremental_query_runtime import run


def test_exp106_selects_subset_and_incremental_matches_full():
    result = run(seed=3, n=80)

    query = result["query"]
    incremental = result["incremental"]

    assert query["selected_primitive_count"] < query["total_primitive_count"]
    assert query["selected_view_count"] < query["total_view_count"]
    assert query["excluded_primitive_fraction"] > 0.0

    assert incremental["affected_output_cells"] > 0
    assert incremental["matches_full_on_affected_cells"] is True
    assert incremental["output_nnz"] > 0
