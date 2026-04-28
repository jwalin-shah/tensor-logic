import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import unittest

from experiments.exp79_lewm_tl import (
    generate_sequence, compute_relations,
    OBJ_NAMES, CANVAS, OBJ_SIZE, PAIRS, PAIR_INDEX, RELATIONS,
)


class TestDataGen(unittest.TestCase):
    def test_generate_sequence_shapes(self):
        rng = np.random.default_rng(0)
        frames, pos = generate_sequence(n_frames=8, rng=rng)
        self.assertEqual(frames.shape, (8, 3, 64, 64))
        self.assertEqual(pos.shape, (8, 3, 2))

    def test_frames_in_range(self):
        rng = np.random.default_rng(1)
        frames, _ = generate_sequence(rng=rng)
        self.assertGreaterEqual(frames.min().item(), 0.0)
        self.assertLessEqual(frames.max().item(), 1.0)

    def test_positions_in_bounds(self):
        rng = np.random.default_rng(2)
        _, pos = generate_sequence(rng=rng)
        half = OBJ_SIZE // 2
        self.assertTrue((pos >= half).all())
        self.assertTrue((pos <= CANVAS - half).all())

    def test_compute_relations_above(self):
        # R at y=10, G at y=50: R is above G
        pos = np.array([[32.0, 10.0], [32.0, 50.0], [10.0, 30.0]])
        rels = compute_relations(pos)
        self.assertTrue(rels[("above", "R", "G")])
        self.assertFalse(rels[("above", "G", "R")])

    def test_compute_relations_left_of(self):
        pos = np.array([[10.0, 32.0], [50.0, 32.0], [30.0, 10.0]])
        rels = compute_relations(pos)
        self.assertTrue(rels[("left_of", "R", "G")])
        self.assertFalse(rels[("left_of", "G", "R")])

    def test_compute_relations_touching(self):
        # R at (20,20), G at (25,20): gap_x = |20-25| - 10 = -5 < 2 → touching
        pos = np.array([[20.0, 20.0], [25.0, 20.0], [55.0, 55.0]])
        rels = compute_relations(pos)
        self.assertTrue(rels[("touching", "R", "G")])
        self.assertFalse(rels[("touching", "R", "B")])

    def test_compute_relations_all_keys_present(self):
        pos = np.array([[10.0, 10.0], [32.0, 32.0], [55.0, 55.0]])
        rels = compute_relations(pos)
        for rel in RELATIONS:
            for a in OBJ_NAMES:
                for b in OBJ_NAMES:
                    if a != b:
                        self.assertIn((rel, a, b), rels)

    def test_pairs_count_and_index(self):
        self.assertEqual(len(PAIRS), 6)
        self.assertIn(("R", "G"), PAIRS)
        self.assertNotIn(("R", "R"), PAIRS)
        self.assertIn(("R", "G"), PAIR_INDEX)


from experiments.exp79_lewm_tl import Encoder, Predictor, jepa_loss, generate_sequence


class TestJEPA(unittest.TestCase):
    def test_encoder_output_shape(self):
        enc = Encoder(latent_dim=64)
        x = torch.zeros(4, 3, 64, 64)
        z = enc(x)
        self.assertEqual(z.shape, (4, 64))

    def test_predictor_output_shape(self):
        pred = Predictor(latent_dim=64)
        z = torch.zeros(4, 64)
        out = pred(z)
        self.assertEqual(out.shape, (4, 64))

    def test_jepa_loss_is_scalar_nonneg(self):
        enc = Encoder(64)
        pred = Predictor(64)
        frames = torch.zeros(2, 8, 3, 64, 64)
        loss = jepa_loss(enc, pred, frames, "cpu")
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_jepa_loss_backward(self):
        enc = Encoder(64)
        pred = Predictor(64)
        frames = torch.rand(2, 4, 3, 64, 64)
        loss = jepa_loss(enc, pred, frames, "cpu")
        loss.backward()
        for p in list(enc.parameters()) + list(pred.parameters()):
            self.assertIsNotNone(p.grad)

    def test_jepa_loss_training_stable(self):
        """Training steps run, params update, loss stays finite."""
        torch.manual_seed(0)
        enc = Encoder(64)
        pred = Predictor(64)
        params = list(enc.parameters()) + list(pred.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        rng = np.random.default_rng(0)
        seqs = [generate_sequence(n_frames=4, rng=rng)[0] for _ in range(8)]
        frames = torch.stack(seqs)
        params_before = [p.data.clone() for p in params]
        for _ in range(5):
            opt.zero_grad()
            jepa_loss(enc, pred, frames, "cpu").backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        self.assertTrue(any(not torch.equal(pb, p.data) for pb, p in zip(params_before, params)))
        with torch.no_grad():
            loss_after = jepa_loss(enc, pred, frames, "cpu")
        self.assertTrue(torch.isfinite(loss_after))


from experiments.exp79_lewm_tl import (
    Encoder, RelationProbe, make_probe_dataset, eval_probe,
    generate_labeled_frames,
)


class TestProbe(unittest.TestCase):
    def setUp(self):
        self.enc = Encoder(64)
        self.probe = RelationProbe()
        self.device = "cpu"

    def test_probe_forward_shape(self):
        z = torch.zeros(8, 64)
        pair_idx = torch.zeros(8, dtype=torch.long)
        logits = self.probe(z, pair_idx)
        self.assertEqual(set(logits.keys()), {"above", "left_of", "touching", "occluded"})
        for v in logits.values():
            self.assertEqual(v.shape, (8,))

    def test_make_probe_dataset_shapes(self):
        frames = generate_labeled_frames(n=10, seed=7)
        z, pi, lab = make_probe_dataset(frames, self.enc, self.device)
        # 10 frames × 6 pairs = 60 samples
        self.assertEqual(z.shape, (60, 64))
        self.assertEqual(pi.shape, (60,))
        self.assertEqual(lab.shape, (60, 4))

    def test_make_probe_dataset_labels_binary(self):
        frames = generate_labeled_frames(n=5, seed=8)
        _, _, lab = make_probe_dataset(frames, self.enc, self.device)
        self.assertTrue(((lab == 0) | (lab == 1)).all())

    def test_eval_probe_returns_acc_dict(self):
        frames = generate_labeled_frames(n=20, seed=9)
        acc = eval_probe(self.probe, frames, self.enc, self.device)
        self.assertEqual(set(acc.keys()), set(RELATIONS))
        for v in acc.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


from experiments.exp79_lewm_tl import (
    build_tl_state, remove_object, compute_gt_derived,
    compute_relations, OBJ_IDX,
)


class TestTLWiring(unittest.TestCase):
    def _simple_facts(self):
        # R at top-left, G at bottom-center, B at top-right
        pos = np.array([[10.0, 10.0], [32.0, 54.0], [20.0, 10.0]])
        return compute_relations(pos), pos

    def test_build_tl_state_keys(self):
        facts, _ = self._simple_facts()
        state = build_tl_state(facts)
        self.assertIn("above", state)
        self.assertIn("blocked_path", state)
        self.assertIn("same_side", state)
        self.assertIn("clear_above", state)

    def test_build_tl_state_shapes(self):
        facts, _ = self._simple_facts()
        state = build_tl_state(facts)
        self.assertEqual(state["above"].shape, (3, 3))
        self.assertEqual(state["blocked_path"].shape, (3, 3))
        self.assertEqual(state["clear_above"].shape, (3,))

    def test_build_tl_state_above_values(self):
        # R at y=10, G at y=54, B at y=10 → above(R,G)=1, above(B,G)=1
        pos = np.array([[10.0, 10.0], [32.0, 54.0], [20.0, 10.0]])
        facts = compute_relations(pos)
        state = build_tl_state(facts)
        ri, gi, bi = OBJ_IDX["R"], OBJ_IDX["G"], OBJ_IDX["B"]
        self.assertEqual(state["above"][ri, gi].item(), 1.0)
        self.assertEqual(state["above"][bi, gi].item(), 1.0)
        self.assertEqual(state["above"][gi, ri].item(), 0.0)

    def test_remove_object_removes_both_directions(self):
        facts, _ = self._simple_facts()
        filtered = remove_object(facts, "B")
        for k in filtered:
            self.assertNotEqual(k[1], "B")
            self.assertNotEqual(k[2], "B")

    def test_remove_object_preserves_others(self):
        facts, _ = self._simple_facts()
        filtered = remove_object(facts, "B")
        self.assertIn(("above", "R", "G"), filtered)

    def test_compute_gt_derived_shapes(self):
        _, pos = self._simple_facts()
        gt = compute_gt_derived(pos)
        self.assertEqual(gt["blocked_path"].shape, (3, 3))
        self.assertEqual(gt["same_side"].shape, (3, 3))
        self.assertEqual(gt["clear_above"].shape, (3,))

    def test_tl_only_retraction_deterministic(self):
        # With ground-truth facts, TL retraction must match ground-truth geometry exactly
        pos = np.array([[10.0, 10.0], [32.0, 54.0], [20.0, 10.0]])
        facts = compute_relations(pos)
        state_full = build_tl_state(facts)
        facts_no_b = remove_object(facts, "B")
        state_no_b = build_tl_state(facts_no_b)
        gt_no_b = compute_gt_derived(pos, exclude_obj="B")
        self.assertTrue(torch.equal(state_no_b["blocked_path"].round(), gt_no_b["blocked_path"]))
        self.assertTrue(torch.equal(state_no_b["same_side"].round(), gt_no_b["same_side"]))
        self.assertTrue(torch.equal(state_no_b["clear_above"].round(), gt_no_b["clear_above"]))


if __name__ == "__main__":
    unittest.main()
