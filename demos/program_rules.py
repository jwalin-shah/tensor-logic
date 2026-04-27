import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensor_logic import Program


def main():
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

    print("Tensor Logic Program demo")
    print("=" * 72)
    print("grandparent(x,z) := (parent(x,y) * parent(y,z)).step()")
    print("ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()")
    print()
    print("grandparent(alice, carol) =", int(program.query("grandparent", "alice", "carol")))
    print("ancestor(alice, dave) =", int(program.query("ancestor", "alice", "dave", recursive=True)))
    print("ancestor(dave, alice) =", int(program.query("ancestor", "dave", "alice", recursive=True)))


if __name__ == "__main__":
    main()
