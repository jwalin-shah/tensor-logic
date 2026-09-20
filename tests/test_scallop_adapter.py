from tensor_logic.scallop_adapter import (
    SCALLOP_UPSTREAM_COMMIT,
    debug_fact_specs,
    edge_path_program,
    run_unit_transitive_closure,
    scallopy_available,
    to_difftopk_debug_facts,
)


def test_adapter_pins_exact_upstream_commit_and_recursive_program():
    assert len(SCALLOP_UPSTREAM_COMMIT) == 40
    program = edge_path_program()
    assert "path(a, c)" in program
    assert "path(a, b)" in program
    assert "edge(b, c)" in program


def test_debug_fact_specs_assign_contiguous_ids_from_one():
    specs = debug_fact_specs(
        [("a", "b"), ("b", "c"), ("a", "d")],
        [0.9, 0.8, 0.2],
    )

    assert tuple(spec.fact_id for spec in specs) == (1, 2, 3)
    assert specs[0].fact == ("a", "b")
    assert specs[1].probability == 0.8


def test_debug_fact_conversion_matches_documented_tag_shape():
    specs = debug_fact_specs(
        [("a", "b"), ("b", "c")],
        [0.9, 0.8],
    )
    facts = to_difftopk_debug_facts(specs)

    assert len(facts) == 2
    tag, fact = facts[0]
    probability, fact_id = tag

    assert abs(float(probability.item()) - 0.9) < 1e-6
    assert fact_id == 1
    assert fact == ("a", "b")


def test_noncontiguous_debug_ids_fail_closed():
    specs = list(
        debug_fact_specs(
            [("a", "b"), ("b", "c")],
            [0.9, 0.8],
        )
    )
    specs[1] = type(specs[1])(
        fact_id=3,
        probability=specs[1].probability,
        fact=specs[1].fact,
    )

    try:
        to_difftopk_debug_facts(specs)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("noncontiguous debug IDs should fail")


def test_unit_runtime_is_real_upstream_or_fails_loudly_when_not_installed():
    if scallopy_available():
        result = run_unit_transitive_closure(
            [("a", "b"), ("b", "c")]
        )
        assert ("a", "c") in result.relation
        assert result.provenance == "unit"
    else:
        try:
            run_unit_transitive_closure(
                [("a", "b"), ("b", "c")]
            )
        except RuntimeError as exc:
            assert "scallopy is not installed" in str(exc)
        else:
            raise AssertionError("missing Scallop runtime should fail loudly")
