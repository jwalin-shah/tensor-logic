import pytest

import tensor_logic
from tensor_logic import core
from tensor_logic.program import Atom


def test_package_root_exports_supported_core_api_only():
    assert tensor_logic.__all__ == core.__all__
    assert set(tensor_logic.__all__) == {
        "Domain",
        "Relation",
        "Program",
        "FactSource",
        "Proof",
        "NegativeProof",
        "prove",
        "prove_negative",
        "prove_with_do",
        "prove_binary_relation_result",
        "format_proof_result",
        "bfs_query",
        "bfs_per_source_closure",
        "dense_closure",
        "evaluate_expr",
        "facts",
    }


def test_supported_root_exports_are_importable():
    for name in tensor_logic.__all__:
        assert getattr(tensor_logic, name) is getattr(core, name)


def test_wildcard_import_excludes_adapter_and_research_helpers():
    namespace: dict[str, object] = {}
    exec("from tensor_logic import *", namespace)

    assert "Program" in namespace
    assert "prove" in namespace
    assert "execute_command" not in namespace
    assert "load_tl" not in namespace
    assert "parse_rule" not in namespace
    assert "evaluate_with_provenance" not in namespace


def test_module_scoped_helpers_remain_available_at_owning_modules():
    from tensor_logic.execution import execute_command
    from tensor_logic.file_format import load_tl
    from tensor_logic.rules import evaluate_rule, parse_rule, query_relation

    assert callable(execute_command)
    assert callable(load_tl)
    assert callable(parse_rule)
    assert callable(evaluate_rule)
    assert callable(query_relation)


def test_atom_args_are_variadic_but_left_right_are_binary_only():
    atom = Atom("ternary", ("x", "y", "z"))

    assert atom.args == ("x", "y", "z")
    with pytest.raises(ValueError, match="only defined for binary atoms"):
        _ = atom.left
    with pytest.raises(ValueError, match="only defined for binary atoms"):
        _ = atom.right
