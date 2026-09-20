from experiments.exp107_materialized_dag_reuse import run


def test_exp107_materialization_avoids_most_repeated_derivations():
    result = run(iterations=40)

    assert result["stateless_derivations"] == 160
    assert result["materialized_derivations"] < result["stateless_derivations"]
    assert result["derivations_avoided"] > 0
    assert result["derivation_reduction_fraction"] > 0.50
    assert result["cache_hits"] > result["cache_misses"]
    assert result["last_output"] != result["first_output"]
