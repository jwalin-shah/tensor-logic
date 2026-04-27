import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensor_logic import Domain, Relation


def main():
    people = Domain(["alice", "bob", "carol", "dave", "eve"])

    parent = Relation("parent", people, people)
    sibling = Relation("sibling", people, people)
    grandparent = Relation("grandparent", people, people)
    uncle = Relation("uncle", people, people)
    ancestor = Relation("ancestor", people, people)

    parent["alice", "bob"] = 1
    parent["alice", "carol"] = 1
    parent["bob", "dave"] = 1
    parent["carol", "eve"] = 1
    sibling["bob", "carol"] = 1
    sibling["carol", "bob"] = 1

    grandparent["x", "z"] = (parent["x", "y"] * parent["y", "z"]).step()
    uncle["x", "z"] = (sibling["x", "y"] * parent["y", "z"]).step()
    ancestor["x", "z"] = (parent["x", "z"] + ancestor["x", "y"] * parent["y", "z"]).step()

    print("Tensor Logic language demo")
    print("=" * 72)
    print('grandparent["x","z"] = step(parent["x","y"] * parent["y","z"])')
    print('uncle["x","z"]       = step(sibling["x","y"] * parent["y","z"])')
    print('ancestor["x","z"]    = fixpoint(step(parent["x","z"] + ancestor["x","y"] * parent["y","z"]))')
    print()
    for rel, subj, obj in [
        (grandparent, "alice", "dave"),
        (grandparent, "alice", "eve"),
        (uncle, "bob", "eve"),
        (uncle, "carol", "dave"),
        (uncle, "alice", "dave"),
    ]:
        print(f"{rel.name}({subj}, {obj}) = {int(rel.value(subj, obj))}")
    print(f"ancestor(alice, dave) = {int(ancestor.reachable('alice', 'dave'))}")
    print(f"ancestor(alice, eve) = {int(ancestor.reachable('alice', 'eve'))}")
    print(f"ancestor(bob, eve) = {int(ancestor.reachable('bob', 'eve'))}")


if __name__ == "__main__":
    main()
