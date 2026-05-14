import unittest

import torch

from tensor_logic import (
    Atom,
    Domain,
    Program,
    Relation,
    Rule,
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
    fmt_negative_proof_tree,
    fmt_proof_tree,
)
from tensor_logic.semirings import gf2_matmul


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
        self.assertIsInstance(rule, Rule)
        self.assertIsInstance(rule.head, Atom)
        self.assertEqual(rule.head.relation, "uncle")
        self.assertEqual(rule.head.args, ("X", "Y"))
        self.assertEqual(rule.head.rel, "uncle")
        self.assertEqual((rule.head.left, rule.head.right), ("X", "Y"))
        self.assertEqual(rule.body[0].relation, "sibling")
        self.assertEqual(rule.body[0].args, ("X", "P"))
        result, err = evaluate_rule(GRAPH, rule)
        self.assertIsNone(err)
        self.assertTrue(query_relation(result, "carol", "dave"))
        self.assertFalse(query_relation(result, "alice", "dave"))

    def test_stratified_negation(self):
        rule = parse_rule(
            '<tl_rule head="sibling_not_parent(X, Y)" body="sibling(X, Y), !parent(X, Y)"></tl_rule>'
        )
        self.assertTrue(rule.body[1].negated)
        self.assertEqual(rule.body[1], Atom("parent", ("X", "Y"), negated=True))
        result, err = evaluate_rule(GRAPH, rule)
        self.assertIsNone(err)
        self.assertTrue(query_relation(result, "bob", "carol"))
        self.assertFalse(query_relation(result, "alice", "bob"))

    def test_tl_rule_parse_rejects_invalid_syntax(self):
        self.assertIsNone(parse_rule("no tag here"))
        self.assertIsNone(parse_rule('<tl_rule head="!bad(X, Y)" body="parent(X, Y)"></tl_rule>'))
        self.assertIsNone(parse_rule('<tl_rule head="bad(X, Y)" body=""></tl_rule>'))
        self.assertIsNone(parse_rule('<tl_rule head="bad(X, Y)" body="parent(X, Y) garbage"></tl_rule>'))
        self.assertIsNone(parse_rule('<tl_rule head="bad(X, Y)" body="parent(X, Y),"></tl_rule>'))

    def test_tl_rule_evaluate_reports_unknown_relation(self):
        rule = parse_rule('<tl_rule head="derived(X, Y)" body="missing(X, Y)"></tl_rule>')
        result, err = evaluate_rule(GRAPH, rule)
        self.assertIsNone(result)
        self.assertEqual(err, "unknown body relation: missing")

    def test_tl_rule_evaluate_reports_unknown_negated_relation(self):
        rule = parse_rule('<tl_rule head="derived(X, Y)" body="parent(X, Y), !missing(X, Y)"></tl_rule>')
        result, err = evaluate_rule(GRAPH, rule)
        self.assertIsNone(result)
        self.assertEqual(err, "unknown negated relation: missing")

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

    def test_program_query_rejects_extra_symbols(self):
        program = Program()
        program.domain("Node", ["a", "b", "c"])
        program.relation("edge", "Node", "Node")
        program.fact("edge", "a", "b")

        with self.assertRaisesRegex(ValueError, "expects 2 args, got 3"):
            program.query("edge", "a", "b", "c")

    def test_tl_file_query_and_proof(self):
        loaded = load_tl("examples/code_dependencies.tl")
        self.assertEqual(loaded.program.query("depends_on", "worker", "models", recursive=True), 1.0)
        proof = prove(loaded.program, "depends_on", "worker", "models", recursive=True)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.head, ("depends_on", "worker", "models"))
        self.assertGreater(len(proof.body), 0)

    def test_tl_file_rejects_unknown_command_flag(self):
        import tempfile
        from pathlib import Path

        source = "\n".join(
            [
                "domain Node { a b c }",
                "relation edge(Node, Node)",
                "relation path(Node, Node)",
                "fact edge(a, b)",
                "fact edge(b, c)",
                "rule path(x,y) := edge(x,y).step()",
                "rule path(x,y) := edge(x,z) * path(z,y).step()",
                "query path(a, c) recursvie",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_flag.tl"
            path.write_text(source, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown command flag\\(s\\): recursvie"):
                load_tl(str(path))

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

    def test_format_negative_proof_tree(self):
        loaded = load_tl("examples/code_dependencies.tl")
        neg_proof = prove_negative(loaded.program, "depends_on", "models", "worker")
        self.assertIsNotNone(neg_proof)

        formatted = fmt_negative_proof_tree(neg_proof)
        self.assertIn("depends_on(models, worker)", formatted)
        self.assertIsInstance(formatted, str)


    def test_validation_errors(self):
        program = Program()
        program.domain("X", ["a", "b"])
        program.relation("edge", "X", "X")
        program.fact("edge", "a", "b")

        with self.assertRaises(ValueError, msg="unknown relation"):
            program.fact("nonexistent", "a", "b")
        with self.assertRaises(ValueError, msg="unknown symbol"):
            program.fact("edge", "a", "z")
        with self.assertRaises(ValueError, msg="unknown domain"):
            program.relation("foo", "Y", "X")
        with self.assertRaises(ValueError, msg="prove unknown relation"):
            prove(program, "nonexistent", "a", "b")
        with self.assertRaises(ValueError, msg="prove unknown symbol"):
            prove(program, "edge", "a", "z")

    def test_query_rejects_wrong_arity_and_unknown_symbols(self):
        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")
        program.fact("edge", "a", "b")

        with self.assertRaisesRegex(ValueError, "edge expects 2 symbols, got 1"):
            program.query("edge", "a")
        with self.assertRaisesRegex(ValueError, "edge expects 2 symbols, got 3"):
            program.query("edge", "a", "b", "extra")
        with self.assertRaisesRegex(ValueError, "symbol 'z' not in domain"):
            program.query("edge", "a", "z")

    def test_recursive_query_rejects_wrong_arity(self):
        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")
        program.fact("edge", "a", "b")

        with self.assertRaisesRegex(ValueError, "edge expects 2 symbols, got 3"):
            program.query("edge", "a", "b", "extra", recursive=True)

    def test_confidence_propagates(self):
        program = Program()
        program.domain("X", ["a", "b", "c"])
        program.relation("edge", "X", "X")
        program.relation("path", "X", "X")
        program.fact("edge", "a", "b", value=0.9)
        program.fact("edge", "b", "c", value=0.8)
        program.rule("path(x,z) := (edge(x,y) * edge(y,z)).step()")

        proof = prove(program, "path", "a", "c")
        self.assertIsNotNone(proof)
        self.assertAlmostEqual(proof.confidence, 0.72, places=5)
        self.assertAlmostEqual(proof.body[0].confidence, 0.9, places=5)
        self.assertAlmostEqual(proof.body[1].confidence, 0.8, places=5)
        # Confidence tag appears in formatted output
        formatted = fmt_proof_tree(proof)
        self.assertIn("(0.72)", formatted)

    def test_source_backed_facts(self):
        loaded = load_tl("examples/personal_memory.tl")
        proof = prove(loaded.program, "should_follow_up", "ryan", "tensor_demo")
        self.assertIsNotNone(proof)
        # Leaf nodes should have source locations from the .tl file
        leaf = proof.body[0]
        self.assertIsNotNone(leaf.source)
        self.assertIn("personal_memory.tl", leaf.source.file)
        self.assertGreater(leaf.source.lineno, 0)
        # Formatted tree should include file:line annotations
        formatted = fmt_proof_tree(proof)
        self.assertIn("personal_memory.tl:", formatted)

    def test_prove_with_multiple_rules(self):
        # connected(x,y) if there is a direct link OR a via_hub path
        program = Program()
        program.domain("Node", ["a", "b", "c", "d"])
        program.relation("direct", "Node", "Node")
        program.relation("via_hub", "Node", "Node")
        program.relation("connected", "Node", "Node")
        program.fact("direct", "a", "b")
        program.fact("via_hub", "c", "d")
        program.rule("connected(x,y) := direct(x,y).step()")
        program.rule("connected(x,y) := via_hub(x,y).step()")

        self.assertEqual(len(program.rules["connected"]), 2)

        # Provable via rule 1
        proof1 = prove(program, "connected", "a", "b")
        self.assertIsNotNone(proof1)

        # Provable via rule 2
        proof2 = prove(program, "connected", "c", "d")
        self.assertIsNotNone(proof2)

    def test_prove_negative_all_rules_failed(self):
        program = Program()
        program.domain("Node", ["a", "b", "c", "d"])
        program.relation("direct", "Node", "Node")
        program.relation("via_hub", "Node", "Node")
        program.relation("connected", "Node", "Node")
        program.fact("direct", "a", "b")
        program.fact("via_hub", "c", "d")
        program.rule("connected(x,y) := direct(x,y).step()")
        program.rule("connected(x,y) := via_hub(x,y).step()")

        # a→d: neither rule can prove this
        neg = prove_negative(program, "connected", "a", "d")
        self.assertIsNotNone(neg)
        self.assertEqual(neg.head, ("connected", "a", "d"))
        self.assertEqual(neg.reason, "all_rules_failed")
        self.assertEqual(len(neg.body), 2)


    def _make_ancestor_program(self):
        from tensor_logic.program import Program
        program = Program()
        program.domain("Person", ["alice", "bob", "carol", "dave"])
        program.relation("parent", "Person", "Person")
        program.relation("ancestor", "Person", "Person")
        program.fact("parent", "alice", "bob")
        program.fact("parent", "bob", "carol")
        program.fact("parent", "carol", "dave")
        program.rule("ancestor(x,y) := parent(x,y).step()")
        program.rule("ancestor(x,y) := parent(x,z) * ancestor(z,y).step()")
        return program

    def test_tabled_recursive_proof_direct(self):
        from tensor_logic.proofs import prove
        program = self._make_ancestor_program()
        proof = prove(program, "ancestor", "alice", "bob")
        self.assertIsNotNone(proof)
        self.assertEqual(proof.head, ("ancestor", "alice", "bob"))

    def test_tabled_recursive_proof_deep(self):
        from tensor_logic.proofs import prove
        program = self._make_ancestor_program()
        proof = prove(program, "ancestor", "alice", "dave")
        self.assertIsNotNone(proof)
        self.assertEqual(proof.head, ("ancestor", "alice", "dave"))

    def test_tabled_recursive_proof_negative(self):
        from tensor_logic.proofs import prove
        program = self._make_ancestor_program()
        proof = prove(program, "ancestor", "dave", "alice")
        self.assertIsNone(proof)

    def test_tabled_proof_shows_rule_structure(self):
        from tensor_logic.proofs import prove, fmt_proof_tree
        from tensor_logic.program import Program
        program = Program()
        program.domain("Person", ["alice", "bob", "carol"])
        program.relation("parent", "Person", "Person")
        program.relation("ancestor", "Person", "Person")
        program.fact("parent", "alice", "bob")
        program.fact("parent", "bob", "carol")
        program.rule("ancestor(x,y) := parent(x,y).step()")
        program.rule("ancestor(x,y) := parent(x,z) * ancestor(z,y).step()")
        proof = prove(program, "ancestor", "alice", "carol")
        self.assertIsNotNone(proof)
        tree = fmt_proof_tree(proof)
        self.assertIn("parent(alice", tree)
        self.assertIn("ancestor", tree)

    def test_tabled_proof_handles_cycle_in_data(self):
        # Without tabling, cyclic data (a→b→a) causes infinite recursion.
        from tensor_logic.proofs import prove
        from tensor_logic.program import Program
        program = Program()
        program.domain("Node", ["a", "b", "c"])
        program.relation("edge", "Node", "Node")
        program.relation("reachable", "Node", "Node")
        program.fact("edge", "a", "b")
        program.fact("edge", "b", "a")  # cycle: a→b→a
        program.fact("edge", "b", "c")
        program.rule("reachable(x,y) := edge(x,y).step()")
        program.rule("reachable(x,y) := edge(x,z) * reachable(z,y).step()")
        # Should find proof and not hang
        proof = prove(program, "reachable", "a", "c")
        self.assertIsNotNone(proof)
        self.assertEqual(proof.head, ("reachable", "a", "c"))

    def test_tabled_proof_false_through_cycle_terminates(self):
        # Without tabling: prove("reachable", "a", "c") where c is unreachable
        # but there's a cycle a→b→a would loop forever.
        from tensor_logic.proofs import prove
        from tensor_logic.program import Program
        program = Program()
        program.domain("Node", ["a", "b", "c"])
        program.relation("edge", "Node", "Node")
        program.relation("reachable", "Node", "Node")
        program.fact("edge", "a", "b")
        program.fact("edge", "b", "a")  # cycle only — c is unreachable
        program.rule("reachable(x,y) := edge(x,y).step()")
        program.rule("reachable(x,y) := edge(x,z) * reachable(z,y).step()")
        # c is unreachable — should return None without infinite loop
        proof = prove(program, "reachable", "a", "c")
        self.assertIsNone(proof)

    def test_repl_parse_and_execute(self):
        from tensor_logic.__main__ import _repl_eval
        from tensor_logic.program import Program
        import io
        program = Program()
        out = io.StringIO()
        _repl_eval(program, "domain Node { a, b, c }", out)
        _repl_eval(program, "relation edge(Node, Node)", out)
        _repl_eval(program, "fact edge(a, b)", out)
        _repl_eval(program, "query edge(a, b)", out)
        result = out.getvalue()
        self.assertIn("True", result)

    def test_repl_prove_command(self):
        from tensor_logic.__main__ import _repl_eval
        from tensor_logic.program import Program
        import io
        program = Program()
        out = io.StringIO()
        _repl_eval(program, "domain P { alice, bob }", out)
        _repl_eval(program, "relation knows(P, P)", out)
        _repl_eval(program, "fact knows(alice, bob)", out)
        _repl_eval(program, "prove knows(alice, bob)", out)
        result = out.getvalue()
        self.assertIn("knows(alice, bob)", result)

    def test_include_directive(self):
        import tempfile, os
        from tensor_logic.file_format import load_tl
        with tempfile.TemporaryDirectory() as tmpdir:
            included_path = os.path.join(tmpdir, "nodes.tl")
            base_path = os.path.join(tmpdir, "base.tl")
            with open(included_path, "w") as f:
                f.write('domain Node { alice, bob }\n')
                f.write('relation knows(Node, Node)\n')
                f.write('fact knows(alice, bob)\n')
            with open(base_path, "w") as f:
                f.write('include "nodes.tl"\n')
                f.write('query knows(alice, bob)\n')
            loaded = load_tl(base_path)
            self.assertIn("knows", loaded.program.relations)
            self.assertEqual(len(loaded.commands), 1)
            self.assertEqual(loaded.commands[0].kind, "query")

    def test_include_cycle_raises(self):
        import tempfile, os
        from tensor_logic.file_format import load_tl
        with tempfile.TemporaryDirectory() as tmpdir:
            a_path = os.path.join(tmpdir, "a.tl")
            b_path = os.path.join(tmpdir, "b.tl")
            with open(a_path, "w") as f:
                f.write('include "b.tl"\n')
            with open(b_path, "w") as f:
                f.write('include "a.tl"\n')
            with self.assertRaises(ValueError):
                load_tl(a_path)

    def test_tl_file_rejects_empty_comma_items(self):
        import tempfile, os
        from tensor_logic.file_format import load_tl
        with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as f:
            f.write("domain Node { a,, b }\n")
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, r"empty item in list"):
                load_tl(path)
        finally:
            os.unlink(path)

    def test_proof_json_roundtrip(self):
        from tensor_logic import format_proof_result
        from tensor_logic.proofs import Proof
        original = Proof(
            head=("path", "a", "c"),
            body=(
                Proof(head=("edge", "a", "b"), confidence=0.9),
                Proof(head=("edge", "b", "c"), confidence=0.8),
            ),
            confidence=0.72,
        )
        d = format_proof_result(proof=original, format_type="json")["proof"]
        restored = Proof.from_json(d)
        self.assertEqual(restored.head, original.head)
        self.assertAlmostEqual(restored.confidence, original.confidence, places=6)
        self.assertEqual(len(restored.body), 2)
        self.assertEqual(restored.body[0].head, ("edge", "a", "b"))
        self.assertAlmostEqual(restored.body[0].confidence, 0.9, places=6)

    def test_negative_proof_json_roundtrip(self):
        from tensor_logic import format_proof_result
        from tensor_logic.proofs import NegativeProof
        original = NegativeProof(
            head=("edge", "a", "z"),
            reason="no_rules",
        )
        d = format_proof_result(negative_proof=original, format_type="json")
        restored = NegativeProof.from_json(d)
        self.assertEqual(restored.head, original.head)
        self.assertEqual(restored.reason, original.reason)


    def test_negative_proof_json_uses_nested_explanation_nodes(self):
        from tensor_logic import format_proof_result
        from tensor_logic.proofs import NegativeProof

        original = NegativeProof(
            head=("path", "a", "c"),
            reason="rule_body_failed",
            body=(
                NegativeProof(
                    head=("edge", "b", "c"),
                    reason="no_fact",
                ),
            ),
        )

        result = format_proof_result(negative_proof=original, format_type="json")

        self.assertFalse(result["answer"])
        self.assertEqual(result["explanation"]["body"][0]["head"], ["edge", "b", "c"])
        self.assertEqual(result["explanation"]["body"][0]["reason"], "no_fact")
        self.assertNotIn("explanation", result["explanation"]["body"][0])


    def test_disjunctive_rule_splits_into_two_rules(self):
        # rule with '+' body should register as 2 separate Rules, not 1 merged Rule
        program = Program()
        program.domain("Node", ["a", "b", "c"])
        program.relation("edge", "Node", "Node")
        program.relation("reach", "Node", "Node")
        program.fact("edge", "a", "b")
        program.rule("reach(x,y) := (edge(x,y) + reach(x,z) * edge(z,y)).step()")
        self.assertEqual(len(program.rules["reach"]), 2)

    def test_disjunctive_rule_prover_uses_first_disjunct(self):
        # prove(reach, a, b) should succeed via rule 1 (edge only), not fall back to BFS
        program = Program()
        program.domain("Node", ["a", "b", "c"])
        program.relation("edge", "Node", "Node")
        program.relation("reach", "Node", "Node")
        program.fact("edge", "a", "b")
        program.fact("edge", "b", "c")
        program.rule("reach(x,y) := (edge(x,y) + reach(x,z) * edge(z,y)).step()")
        proof = prove(program, "reach", "a", "b")
        self.assertIsNotNone(proof)
        # body must reference 'edge', not the BFS chain format
        self.assertEqual(len(proof.body), 1)
        self.assertEqual(proof.body[0].head[0], "edge")

    def test_disjunctive_rule_in_tl_file_uses_rule_structure(self):
        # code_dependencies.tl uses (imports + depends_on*imports).step() — 2 rules
        loaded = load_tl("examples/code_dependencies.tl")
        self.assertEqual(len(loaded.program.rules["depends_on"]), 2)
        # direct import — should prove via rule 1 (just imports), not BFS
        proof = prove(loaded.program, "depends_on", "worker", "api")
        self.assertIsNotNone(proof)
        self.assertEqual(proof.body[0].head[0], "imports")

    def test_ingest_python_imports_generates_dependency_program(self):
        import os
        import tempfile
        from tensor_logic.ingest import ingest_python, render_python_imports_tl

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = os.path.join(tmpdir, "pkg")
            os.mkdir(pkg)
            with open(os.path.join(pkg, "__init__.py"), "w") as f:
                f.write("")
            with open(os.path.join(pkg, "api.py"), "w") as f:
                f.write("from . import db\n")
            with open(os.path.join(pkg, "db.py"), "w") as f:
                f.write("from . import models\n")
            with open(os.path.join(pkg, "models.py"), "w") as f:
                f.write("")
            graph = ingest_python(tmpdir)
            text = render_python_imports_tl(graph)
            tl_path = os.path.join(tmpdir, "repo.tl")
            with open(tl_path, "w") as f:
                f.write(text)

            loaded = load_tl(tl_path)
            self.assertEqual(loaded.program.query("depends_on", "pkg_api", "pkg_models", recursive=True), 1.0)
            proof = prove(loaded.program, "depends_on", "pkg_api", "pkg_models")
            self.assertIsNotNone(proof)

    def test_repo_graph_helpers_and_report(self):
        from tensor_logic.repo_graph_view import RepoGraphView, dependency_report, load_repo_graph

        graph = load_repo_graph("examples/code_dependencies.tl")
        self.assertIn("worker", graph.modules)
        self.assertIn(("worker", "api"), graph.imports)

        view = RepoGraphView.load("examples/code_dependencies.tl")
        self.assertEqual(view.direct_imports("worker"), ("api",))
        self.assertEqual(view.search("mo"), ["models"])
        self.assertEqual(view.imports_path("worker", "models"), ["worker", "api", "db", "models"])
        report = dependency_report("examples/code_dependencies.tl", module="worker", src="worker", dst="models")
        self.assertIn("direct imports(worker): api", report)
        self.assertIn("depends_on(worker, models) = True", report)
        self.assertIn("path: worker -> api -> db -> models", report)

    def test_proof_tree_viewer_renders_positive_and_negative(self):
        from tensor_logic.proof_tree_viewer import build_proof_tree_view, render_proof_tree

        positive = {
            "answer": True,
            "proof": {
                "head": ["depends_on", "worker", "models"],
                "confidence": 0.81,
                "source": {"file": "examples/code_dependencies.tl", "lineno": 14},
                "body": [{"head": ["imports", "worker", "api"], "confidence": 0.9, "body": []}],
            },
        }
        rendered = render_proof_tree(build_proof_tree_view(positive))
        self.assertIn("depends_on(worker, models)", rendered)
        self.assertIn("(0.81)", rendered)
        self.assertIn("[examples/code_dependencies.tl:14]", rendered)

        negative = {
            "answer": False,
            "explanation": {
                "head": ["depends_on", "models", "worker"],
                "reason": "all_rules_failed",
                "body": [{"head": ["imports", "models", "worker"], "reason": "no_rules", "body": []}],
            },
        }
        rendered = render_proof_tree(build_proof_tree_view(negative), collapsed={"0"})
        self.assertIn("[+] depends_on(models, worker) = False", rendered)
        self.assertNotIn("imports(models, worker)", rendered)

    def test_http_api_helpers_and_repo_ingest(self):
        import os
        import tempfile
        from tensor_logic.http_api import ingest_python_source, prove_source, query_source, run_source

        source = open("examples/code_dependencies.tl", encoding="utf-8").read()
        self.assertEqual(len(run_source(source)["outputs"]), 2)
        query = query_source(source, "depends_on", ["worker", "models"], recursive=True)
        self.assertTrue(query["answer"])
        proof = prove_source(source, "depends_on", ["worker", "models"], recursive=True, format_type="json")
        self.assertTrue(proof["answer"])
        self.assertIn("proof", proof)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = os.path.join(tmpdir, "pkg")
            os.mkdir(pkg)
            with open(os.path.join(pkg, "__init__.py"), "w") as f:
                f.write("")
            with open(os.path.join(pkg, "api.py"), "w") as f:
                f.write("from . import db\n")
            with open(os.path.join(pkg, "db.py"), "w") as f:
                f.write("")
            tl_source = ingest_python_source(tmpdir)
        self.assertIn("fact imports(pkg_api, pkg_db)", tl_source)

    def test_http_api_prove_validates_request_before_source_parse(self):
        from http import HTTPStatus
        from tensor_logic.http_api import ApiError, prove_source

        malformed_source = "this is not valid TL"

        with self.assertRaises(ApiError) as caught:
            prove_source(malformed_source, "edge", ["a"])
        self.assertEqual(caught.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(caught.exception.message, "prove requires exactly 2 args")

        with self.assertRaises(ApiError) as caught:
            prove_source(malformed_source, "edge", ["a", "b"], format_type="xml")
        self.assertEqual(caught.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(caught.exception.message, "format must be 'tree' or 'json'")

    def test_cli_and_http_share_positive_proof_json_semantics(self):
        import json
        import subprocess
        import sys
        from tensor_logic.http_api import prove_source

        def normalize_source_files(node):
            if isinstance(node, dict):
                return {
                    key: (
                        {"lineno": value["lineno"]}
                        if key == "source" and isinstance(value, dict) and "lineno" in value
                        else normalize_source_files(value)
                    )
                    for key, value in node.items()
                }
            if isinstance(node, list):
                return [normalize_source_files(item) for item in node]
            return node

        source = open("examples/code_dependencies.tl", encoding="utf-8").read()
        http_result = prove_source(source, "depends_on", ["worker", "models"], recursive=True, format_type="json")
        cli_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensor_logic",
                "prove",
                "examples/code_dependencies.tl",
                "depends_on",
                "worker",
                "models",
                "--recursive",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(normalize_source_files(json.loads(cli_result.stdout)), normalize_source_files(http_result))

    def test_cli_and_http_share_negative_proof_json_semantics(self):
        import json
        import subprocess
        import sys
        from tensor_logic.http_api import prove_source

        source = open("examples/code_dependencies.tl", encoding="utf-8").read()
        http_result = prove_source(
            source,
            "depends_on",
            ["models", "worker"],
            recursive=True,
            why_not=True,
            format_type="json",
        )
        cli_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensor_logic",
                "prove",
                "examples/code_dependencies.tl",
                "depends_on",
                "models",
                "worker",
                "--recursive",
                "--why-not",
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(json.loads(cli_result.stdout), http_result)

    def test_cli_run_query_and_prove_execute_tl_commands(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile

        source = "\n".join(
            [
                "domain Node { a, b, c }",
                "relation edge(Node, Node)",
                "relation path(Node, Node)",
                "fact edge(a, b)",
                "fact edge(b, c)",
                "rule path(x,y) := edge(x,y).step()",
                "rule path(x,y) := edge(x,z) * path(z,y).step()",
                "query path(a, c) recursive",
                "prove path(a, c) recursive",
            ]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as f:
            f.write(source)
            path = f.name
        try:
            run_result = subprocess.run(
                [sys.executable, "-m", "tensor_logic", "run", path],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("path(a, c) = True", run_result.stdout)
            self.assertIn("edge(a, b)", run_result.stdout)

            query_result = subprocess.run(
                [sys.executable, "-m", "tensor_logic", "query", path, "path", "a", "c", "--recursive"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(query_result.stdout.strip(), "path(a, c) = True")

            prove_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tensor_logic",
                    "prove",
                    path,
                    "path",
                    "a",
                    "c",
                    "--recursive",
                    "--format",
                    "json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(json.loads(prove_result.stdout)["answer"])
        finally:
            os.unlink(path)

    def test_cli_invalid_arity_and_parse_errors_are_external_failures(self):
        import os
        import subprocess
        import sys
        import tempfile

        source = "\n".join(
            [
                "domain Node { a, b }",
                "relation edge(Node, Node)",
                "fact edge(a, b)",
            ]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as f:
            f.write(source)
            valid_path = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as f:
            f.write("not valid tl\n")
            invalid_path = f.name
        try:
            arity = subprocess.run(
                [sys.executable, "-m", "tensor_logic", "query", valid_path, "edge", "a"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(arity.returncode, 0)
            self.assertIn("query currently supports binary relations", arity.stderr)

            parse = subprocess.run(
                [sys.executable, "-m", "tensor_logic", "run", invalid_path],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(parse.returncode, 0)
            self.assertIn("unrecognized statement", parse.stderr)
        finally:
            os.unlink(valid_path)
            os.unlink(invalid_path)

    def test_http_source_execution_preserves_arity_and_load_errors(self):
        from http import HTTPStatus
        from tensor_logic.http_api import ApiError, prove_source, query_source, run_source

        source = "\n".join(
            [
                "domain Node { a, b }",
                "relation edge(Node, Node)",
                "fact edge(a, b)",
            ]
        )
        with self.assertRaises(ApiError) as caught:
            query_source(source, "edge", ["a"])
        self.assertEqual(caught.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(caught.exception.message, "query requires exactly 2 args")

        with self.assertRaises(ApiError) as caught:
            prove_source(source, "edge", ["a"])
        self.assertEqual(caught.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(caught.exception.message, "prove requires exactly 2 args")

        with self.assertRaises(ValueError) as parse:
            run_source("not valid tl\n")
        self.assertIn("unrecognized statement", str(parse.exception))

    def test_repl_uses_recursive_query_and_prove_semantics(self):
        from tensor_logic.__main__ import _repl_eval
        from tensor_logic.program import Program
        import io

        program = Program()
        out = io.StringIO()
        _repl_eval(program, "domain Node { a, b, c }", out)
        _repl_eval(program, "relation edge(Node, Node)", out)
        _repl_eval(program, "relation path(Node, Node)", out)
        _repl_eval(program, "fact edge(a, b)", out)
        _repl_eval(program, "fact edge(b, c)", out)
        _repl_eval(program, "rule path(x,y) := edge(x,y).step()", out)
        _repl_eval(program, "rule path(x,y) := edge(x,z) * path(z,y).step()", out)
        _repl_eval(program, "query path(a, c) recursive", out)
        _repl_eval(program, "prove path(a, c) recursive", out)
        result = out.getvalue()
        self.assertIn("path(a, c) = True", result)
        self.assertIn("edge(a, b)", result)



    def test_execute_command_returns_query_output(self):
        from tensor_logic import execute_command
        from tensor_logic.file_format import Command
        from tensor_logic.program import Program

        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")
        program.fact("edge", "a", "b")

        result = execute_command(program, Command("query", "edge", ("a", "b")))
        self.assertEqual(result.text, "edge(a, b) = True")

    def test_execute_command_returns_proof_output(self):
        from tensor_logic import execute_command
        from tensor_logic.file_format import Command
        from tensor_logic.program import Program

        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")
        program.fact("edge", "a", "b")

        result = execute_command(program, Command("prove", "edge", ("a", "b")))
        self.assertEqual(result.text, "edge(a, b)")

    def test_execute_command_returns_json_proof_output(self):
        import json
        from tensor_logic import execute_command
        from tensor_logic.file_format import Command
        from tensor_logic.program import Program

        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")
        program.fact("edge", "a", "b")

        result = execute_command(program, Command("prove", "edge", ("a", "b")), format_type="json")
        self.assertEqual(json.loads(result.text)["proof"]["head"], ["edge", "a", "b"])

    def test_execute_command_returns_negative_proof_output_when_requested(self):
        from tensor_logic import execute_command
        from tensor_logic.file_format import Command
        from tensor_logic.program import Program

        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")

        result = execute_command(program, Command("prove", "edge", ("a", "b")), why_not=True)
        self.assertIn("edge(a, b) = False", result.text)
        self.assertIn("reason: no_rules", result.text)

    def test_execute_source_helpers_share_loading_and_result_semantics(self):
        from tensor_logic.execution import execute_source_prove, execute_source_query, execute_source_run

        source = "\n".join(
            [
                "domain Node { a, b }",
                "relation edge(Node, Node)",
                "fact edge(a, b)",
                "query edge(a, b)",
                "prove edge(a, b)",
            ]
        )

        outputs = execute_source_run(source)["outputs"]
        self.assertEqual(outputs[0], "edge(a, b) = True")
        self.assertIn("edge(a, b)", outputs[1])
        self.assertTrue(execute_source_query(source, "edge", ("a", "b"))["answer"])
        proof = execute_source_prove(source, "edge", ("a", "b"), format_type="json")
        self.assertEqual(proof["proof"]["head"], ["edge", "a", "b"])

    def test_execute_prove_owns_proof_format_validation(self):
        from tensor_logic.execution import execute_prove
        from tensor_logic.program import Program

        program = Program()
        program.domain("Node", ["a", "b"])
        program.relation("edge", "Node", "Node")

        with self.assertRaises(ValueError) as caught:
            execute_prove(program, "edge", ("a", "b"), format_type="xml")
        self.assertEqual(str(caught.exception), "format must be 'tree' or 'json'")

    def test_web_workbench_sample_is_valid_tl(self):
        import re
        from tensor_logic.file_format import load_tl

        app_js = open("web_workbench/static/app.js", encoding="utf-8").read()
        match = re.search(r"sourceEl\.value = `(?P<source>.*?)`;", app_js, re.S)
        self.assertIsNotNone(match)
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".tl", delete=False, encoding="utf-8") as f:
            f.write(match.group("source"))
            path = f.name
        loaded = load_tl(path)
        self.assertEqual(loaded.program.query("ancestor", "alice", "cara", recursive=True), 1.0)


if __name__ == "__main__":
    unittest.main()
