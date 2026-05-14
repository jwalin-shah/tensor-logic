import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_logic.research import synthetic_scene as scene
from experiments import exp79_lewm_tl, exp83_slot_attention


def test_experiments_use_canonical_spatial_scene_helpers():
    assert exp79_lewm_tl.generate_sequence is scene.generate_colored_object_sequence
    assert exp83_slot_attention.generate_sequence is scene.generate_colored_object_sequence
    assert exp79_lewm_tl.compute_relations is scene.compute_pairwise_spatial_relations
    assert exp83_slot_attention.compute_relations is scene.compute_pairwise_spatial_relations
    assert exp79_lewm_tl.build_tl_state is scene.build_spatial_tl_state
    assert exp83_slot_attention.build_tl_state is scene.build_spatial_tl_state
    assert exp79_lewm_tl.remove_object is scene.remove_object_facts
    assert exp83_slot_attention.remove_object is scene.remove_object_facts
    assert exp79_lewm_tl.compute_gt_derived is scene.compute_ground_truth_spatial_derivations
    assert exp83_slot_attention.compute_gt_derived is scene.compute_ground_truth_spatial_derivations


def test_canonical_pairwise_spatial_relations_cover_all_object_pairs():
    pos = np.array([[10.0, 10.0], [32.0, 54.0], [20.0, 10.0]])
    rels = scene.compute_pairwise_spatial_relations(pos)

    assert len(rels) == len(scene.RELATIONS) * len(scene.PAIRS)
    assert rels[("above", "R", "G")]
    assert rels[("above", "B", "G")]
    assert not rels[("above", "G", "R")]
    assert rels[("left_of", "R", "G")]
    assert not rels[("left_of", "G", "R")]


def test_canonical_spatial_tl_retraction_matches_geometry():
    pos = np.array([[10.0, 10.0], [32.0, 54.0], [20.0, 10.0]])
    facts = scene.compute_pairwise_spatial_relations(pos)
    facts_no_b = scene.remove_object_facts(facts, "B")

    state_no_b = scene.build_spatial_tl_state(facts_no_b)
    gt_no_b = scene.compute_ground_truth_spatial_derivations(pos, exclude_obj="B")

    assert torch.equal(state_no_b["blocked_path"].round(), gt_no_b["blocked_path"])
    assert torch.equal(state_no_b["same_side"].round(), gt_no_b["same_side"])
    assert torch.equal(state_no_b["clear_above"].round(), gt_no_b["clear_above"])
