import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from experiments.exp81_optimize_rule_induction import (
    make_proposer,
    explain_miss,
    parse_artifact,
)


def test_parse_artifact_valid():
    art = json.dumps({"relevant_relations": ["parent", "sibling"], "max_len": 2})
    result = parse_artifact(art)
    assert result["relevant_relations"] == ["parent", "sibling"]
    assert result["max_len"] == 2


def test_parse_artifact_invalid_returns_none():
    assert parse_artifact("not json") is None
    assert parse_artifact(json.dumps({"wrong_key": []})) is None


def test_explain_miss_chain_breaks_at_hop0():
    # 2-entity world: entity 0 has no parent edges at all
    base = {
        "parent": torch.zeros(3, 3),
        "sibling": torch.zeros(3, 3),
    }
    msg = explain_miss(["parent", "parent"], base, src=0, dst=2)
    assert "hop 0" in msg
    assert "parent" in msg


def test_explain_miss_chain_breaks_at_hop1():
    # entity 0→1 exists via parent, but 1 has no parent edge
    base = {
        "parent": torch.zeros(3, 3),
        "sibling": torch.zeros(3, 3),
    }
    base["parent"][0, 1] = 1.0  # 0→1 exists
    # no 1→2 edge → breaks at hop 1
    msg = explain_miss(["parent", "parent"], base, src=0, dst=2)
    assert "hop 1" in msg


def test_explain_miss_dst_not_in_final_reachable():
    base = {"parent": torch.zeros(3, 3)}
    base["parent"][0, 1] = 1.0
    base["parent"][1, 0] = 1.0  # loops back to 0, never reaches 2
    msg = explain_miss(["parent", "parent"], base, src=0, dst=2)
    assert "2" in msg  # dst mentioned


def test_make_proposer_returns_valid_json():
    """Smoke test: proposer with feedback="" returns parseable JSON."""
    from experiments.exp78_rule_induction import Schema
    schema = Schema(
        target="grandparent",
        primitives=["parent", "sibling"],
        gold_body=["parent", "parent"],
    )
    positive = [(0, 2), (1, 3)]
    negative = [(0, 1), (2, 3)]
    propose = make_proposer(schema=schema, target_rel="grandparent",
                            positive=positive, negative=negative)
    art = propose("")
    parsed = parse_artifact(art)
    assert parsed is not None
    assert "relevant_relations" in parsed
    assert isinstance(parsed["relevant_relations"], list)
