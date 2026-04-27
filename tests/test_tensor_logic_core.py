import unittest

import torch

from tensor_logic import (
    Domain,
    Program,
    Relation,
    bfs_per_source_closure,
    bfs_query,
    dense_closure,
    evaluate_rule,
    evaluate_with_provenance,
    parse_rule,
    proof_score,
    query_relation,
    load_tl,
    prove,
    prove_negative,
)


GRAPH = {
    "parent": [
        ["alice", "bob"],
        ["alice", "carol"],
        ["bob", "dave"],
        ["bob", "eve"],
        ["carol", "frank"],
        ["carol", "grace"],
    ],
    "sibling": [
        ["bob", "carol"],
        ["carol", "bob"],
        ["dave", "eve"],
        ["eve", "dave"],
        ["frank", "grace"],
        ["grace", "frank"],
    ],
}


class TensorLogicCoreTest(unittest.TestCase):
    def test_dense_and_bfs_closure_match(self):
        A = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        R = dense_closure(A)
        self.assertEqual(R[0, 3].item(), 1.0)
        adj = {0: [1], 1: [2], 2: [3]}
        rows = bfs_per_source_closure(adj, 4)
        self.assertIn(3, rows[0])
        self.assertTrue(bfs_query(adj, 0, 3))
        self.assertFalse(bfs_query(adj, 3, 0))

    def test_positive_rule_join(self):
        rule = parse_rule('<tl_rule head="uncle(X, Y)" body="sibling(X, P), parent(P, Y)"></tl_rule>')
        result, err = evaluate_rule(GRAPH, rule)
        self.assertIsNone(err)
        self.assertTrue(query_relation(result, "carol", "dave"))
        self.assertFalse(query_relation(result, "alice", "dave"))

    def test_stratified_negation(self):
        rule = parse_rule(
            '<tl_rule head="sibling_not_parent(X, Y)" body="sibling(X, Y), !parent(X, Y)"></tl_rule>'
        )
        result, err = evaluate_rule(GRAPH, rule)
        self.assertIsNone(err)
        self.assertTrue(query_relation(result, "bob", "carol"))
        self.assertFalse(query_relation(result, "alice", "bob"))

    def test_provenance_and_ranking(self):
        rules = [
            parse_rule('<tl_rule head="grandparent(X, Y)" body="parent(X, Z), parent(Z, Y)"></tl_rule>'),
            parse_rule('<tl_rule head="uncle(X, Y)" body="sibling(X, P), parent(P, Y)"></tl_rule>'),
        ]
        proofs = evaluate_with_provenance(GRAPH, rules, "grandparent", "alice", "dave")
        self.assertEqual(len(proofs), 1)
        self.assertEqual(proof_score(proofs[0]), (1, 3))
        primitive = evaluate_with_provenance(GRAPH, rules, "parent", "alice", "bob")
        self.assertEqual(proof_score(primitive[0]), (0, 1))

    def test_gf2_semiring_parity_control(self):
        A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        B = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        got = gf2_matmul(A, B)
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        self.assertTrue(torch.equal(got, expected))

    def test_named_index_language_join(self):
        people = Domain(["alice", "bob", "carol", "dave"])
        parent = Relation("parent", people, people)
        grandparent = Relation("grandparent", people, people)
        parent["alice", "bob"] = 1
        parent["bob", "dave"] = 1
        parent["alice", "carol"] = 1

        grandparent["x", "z"] = (parent["x", "y"] * parent["y", "z"]).step()

        self.assertEqual(grandparent.value("alice", "dave"), 1.0)
        self.assertEqual(grandparent.value("bob", "dave"), 0.0)

    def test_named_index_recursive_fixpoint(self):
        nodes = Domain(["a", "b", "c", "d"])
        edge = Relation("edge", nodes, nodes)
        path = Relation("path", nodes, nodes)
        edge["a", "b"] = 1
        edge["b", "c"] = 1
        edge["c", "d"] = 1

        path["x", "z"] = (edge["x", "z"] + path["x", "y"] * edge["y", "z"]).step()
        closure = path.fixpoint()

        self.assertEqual(closure[nodes.id("a"), nodes.id("d")].item(), 1.0)
        self.assertEqual(closure[nodes.id("d"), nodes.id("a")].item(), 0.0)

    def test_program_string_rules(self):
        program = Program()
        program.domain("Person", ["alice", "bob", "carol", "dave"])
        program.relation("parent", "Person", "Person")
        program.relation("grandparent", "Person", "Person")
        program.relation("ancestor", "Person", "Person")
        program.fact("parent", "alice", "bob")
        program.fact("parent", "bob", "carol")
        program.fact("parent", "carol", "dave")

        program.rule("grandparent(x,z) := (parent(x,y) * parent(y,z)).step()")
        program.rule("ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()")

        self.assertEqual(program.query("grandparent", "alice", "carol"), 1.0)
        self.assertEqual(program.query("ancestor", "alice", "dave", recursive=True), 1.0)
        self.assertEqual(program.query("ancestor", "dave", "alice", recursive=True), 0.0)

    def test_tl_file_query_and_proof(self):
        loaded = load_tl("examples/code_dependencies.tl")
        self.assertEqual(loaded.program.query("depends_on", "worker", "models", recursive=True), 1.0)
        proof = prove(loaded.program, "depends_on", "worker", "models", recursive=True)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.head, ("depends_on", "worker", "models"))
        self.assertEqual(len(proof.body), 2)

    def test_rule_aware_proof_with_witness(self):
        loaded = load_tl("examples/personal_memory.tl")
        proof = prove(loaded.program, "should_follow_up", "ryan", "tensor_demo")
        self.assertIsNotNone(proof)
        self.assertEqual(proof.head, ("should_follow_up", "ryan", "tensor_demo"))
        self.assertEqual(len(proof.body), 2)

    def test_false_query_returns_no_proof(self):
        loaded = load_tl("examples/code_dependencies.tl")
        proof = prove(loaded.program, "depends_on", "models", "worker")
        self.assertIsNone(proof)

    def test_prove_negative_not_a_fact(self):
        loaded = load_tl("examples/code_dependencies.tl")
        neg_proof = prove_negative(loaded.program, "depends_on", "models", "worker")
        self.assertIsNotNone(neg_proof)
        self.assertEqual(neg_proof.head, ("depends_on", "models", "worker"))
        self.assertIsNotNone(neg_proof.reason)

    def test_prove_negative_true_query_returns_none(self):
        program = Program()
        program.domain("Person", ["alice", "bob"])
        program.relation("parent", "Person", "Person")
        program.fact("parent", "alice", "bob")

        neg_proof = prove_negative(program, "parent", "alice", "bob")
        self.assertIsNone(neg_proof)

    def test_prove_negative_no_rules(self):
        program = Program()
        program.domain("Person", ["alice", "bob"])
        program.relation("parent", "Person", "Person")
        program.relation("uncle", "Person", "Person")
        program.fact("parent", "alice", "bob")
    
        neg_proof = prove_negative(program, "uncle", "alice", "bob")
        self.assertIsNotNone(neg_proof)
        self.assertEqual(neg_proof.reason, "no_rules")


if __name__ == "__main__":
    unittest.main()
