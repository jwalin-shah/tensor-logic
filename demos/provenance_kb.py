import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensor_logic import evaluate_with_provenance, fmt_proof, parse_rule


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

RULES = [
    parse_rule('<tl_rule head="grandparent(X, Y)" body="parent(X, Z), parent(Z, Y)"></tl_rule>'),
    parse_rule('<tl_rule head="uncle(X, Y)" body="sibling(X, P), parent(P, Y)"></tl_rule>'),
    parse_rule('<tl_rule head="cousin(X, Y)" body="parent(P, X), sibling(P, Q), parent(Q, Y)"></tl_rule>'),
]


def ask(rel: str, subj: str, obj: str):
    proofs = evaluate_with_provenance(GRAPH, RULES, rel, subj, obj, sort=True, top_k=3)
    print(f"Q: {rel}({subj}, {obj})?")
    if not proofs:
        print("A: no")
        print()
        return
    print(f"A: yes ({len(proofs)} proof{'s' if len(proofs) != 1 else ''} shown)")
    for i, proof in enumerate(proofs, start=1):
        print(f"Proof {i}:")
        print(fmt_proof(proof, indent=1))
    print()


def main():
    print("TL provenance KB demo")
    print("=" * 72)
    ask("grandparent", "alice", "dave")
    ask("uncle", "bob", "frank")
    ask("cousin", "dave", "frank")
    ask("cousin", "dave", "eve")


if __name__ == "__main__":
    main()
