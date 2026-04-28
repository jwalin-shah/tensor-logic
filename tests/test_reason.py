import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tensor_logic.reason import obs_to_facts, facts_to_tl_source, _extract_obs_ids


_SAMPLE_OBS = [
    {"id": 0, "kind": "decision", "project": "tensor", "text": "switch to Borda voting",
     "ts": "2026-04-28T07:10:00Z", "sha": "d570543"},
    {"id": 1, "kind": "fix", "project": "tensor", "text": "probe architecture fix",
     "ts": "2026-04-28T09:48:00Z"},
    {"id": 2, "kind": "decision", "project": "inbox", "text": "drop asyncio",
     "ts": "2026-04-27T10:00:00Z"},
]


def test_obs_to_facts_kind():
    facts = obs_to_facts(_SAMPLE_OBS[0])
    assert 'obs_kind(0, "decision").' in facts


def test_obs_to_facts_project():
    facts = obs_to_facts(_SAMPLE_OBS[0])
    assert 'obs_project(0, "tensor").' in facts


def test_obs_to_facts_missing_field_omitted():
    obs = {"id": 5, "kind": "fix", "project": "tensor"}
    facts = obs_to_facts(obs)
    assert "obs_sha" not in facts
    assert "obs_file" not in facts


def test_facts_to_tl_source_contains_all_obs():
    source = facts_to_tl_source(_SAMPLE_OBS)
    assert 'obs_kind(0, "decision").' in source
    assert 'obs_kind(1, "fix").' in source
    assert 'obs_project(2, "inbox").' in source


def test_extract_obs_ids():
    source = facts_to_tl_source(_SAMPLE_OBS)
    ids = _extract_obs_ids(source)
    assert set(ids) == {0, 1, 2}
