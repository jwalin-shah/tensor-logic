import unittest

import json
import tempfile
import threading
from pathlib import Path
from urllib.request import Request, urlopen

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
    fmt_negative_proof_tree,
    fmt_proof_tree,
)
from tensor_logic.semirings import gf2_matmul
from tensor_logic.http_api import TensorLogicHandler
from http.server import ThreadingHTTPServer


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
        self.assertGreater(len(proof.body), 0)

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

    def test_proof_json_roundtrip(self):
        from tensor_logic.proofs import Proof
        from tensor_logic.__main__ import _proof_to_json
        original = Proof(
            head=("path", "a", "c"),
            body=(
                Proof(head=("edge", "a", "b"), confidence=0.9),
                Proof(head=("edge", "b", "c"), confidence=0.8),
            ),
            confidence=0.72,
        )
        d = _proof_to_json(original)
        restored = Proof.from_json(d)
        self.assertEqual(restored.head, original.head)
        self.assertAlmostEqual(restored.confidence, original.confidence, places=6)
        self.assertEqual(len(restored.body), 2)
        self.assertEqual(restored.body[0].head, ("edge", "a", "b"))
        self.assertAlmostEqual(restored.body[0].confidence, 0.9, places=6)

    def test_negative_proof_json_roundtrip(self):
        from tensor_logic.proofs import NegativeProof
        from tensor_logic.__main__ import _negative_proof_to_json
        original = NegativeProof(
            head=("edge", "a", "z"),
            reason="no_rules",
        )
        d = _negative_proof_to_json(original)
        restored = NegativeProof.from_json(d)
        self.assertEqual(restored.head, original.head)
        self.assertEqual(restored.reason, original.reason)


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


class TensorLogicHttpApiTest(unittest.TestCase):
    def _post_json(self, base_url: str, path: str, payload: dict):
        req = Request(
            url=f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_source_api_helpers_on_code_dependencies_example(self):
        from tensor_logic.http_api import run_source, query_source, prove_source

        source = Path("examples/code_dependencies.tl").read_text(encoding="utf-8")

        run_result = run_source(source)
        self.assertEqual(len(run_result["outputs"]), 2)
        self.assertIn("depends_on(worker, models) = True", run_result["outputs"][0])

        query_result = query_source(source, relation="depends_on", args=["worker", "models"], recursive=True)
        self.assertTrue(query_result["answer"])
        self.assertEqual(query_result["value"], 1.0)

        prove_result = prove_source(
            source,
            relation="depends_on",
            args=["models", "worker"],
            recursive=True,
            why_not=True,
            format_type="json",
        )
        self.assertFalse(prove_result["answer"])
        self.assertIn("explanation", prove_result)

    def test_ingest_python_builds_tl(self):
        from tensor_logic.http_api import ingest_python

        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = Path(tmpdir) / "worker.py"
            py_path.write_text("import api\nfrom db import models\n", encoding="utf-8")
            tl_source = ingest_python(str(py_path))

        self.assertIn("domain Module", tl_source)
        self.assertIn("fact imports(worker, api)", tl_source)
        self.assertIn("fact imports(worker, db)", tl_source)
        self.assertIn("rule depends_on", tl_source)

    def test_http_endpoints(self):
        source = Path("examples/code_dependencies.tl").read_text(encoding="utf-8")

        server = ThreadingHTTPServer(("127.0.0.1", 0), TensorLogicHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            host, port = server.server_address
            base_url = f"http://{host}:{port}"

            status, run_payload = self._post_json(base_url, "/run", {"source": source})
            self.assertEqual(status, 200)
            self.assertEqual(len(run_payload["outputs"]), 2)

            status, query_payload = self._post_json(
                base_url,
                "/query",
                {
                    "source": source,
                    "relation": "depends_on",
                    "args": ["worker", "models"],
                    "recursive": True,
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(query_payload["answer"])

            status, prove_payload = self._post_json(
                base_url,
                "/prove",
                {
                    "source": source,
                    "relation": "depends_on",
                    "args": ["worker", "models"],
                    "recursive": True,
                    "format": "json",
                },
            )
            self.assertEqual(status, 200)
            self.assertTrue(prove_payload["answer"])
            self.assertIn("proof", prove_payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
