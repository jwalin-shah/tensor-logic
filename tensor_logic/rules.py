import re

import torch

from .program import Atom, Rule

RULE_RE = re.compile(
    r'<tl_rule\s+head="(?P<head>[^"]+)"\s+body="(?P<body>[^"]+)"\s*></tl_rule>'
)
ATOM_RE = re.compile(r'(?P<neg>!?)\s*(?P<rel>\w+)\s*\(\s*(?P<a>\w+)\s*,\s*(?P<b>\w+)\s*\)')


def parse_rule(text: str) -> Rule | None:
    """Parse `<tl_rule head="r(X,Y)" body="a(X,Z), !b(Z,Y)"></tl_rule>`."""
    match = RULE_RE.search(text)
    if not match:
        return None
    head_match = ATOM_RE.fullmatch(match.group("head").strip())
    body_atoms = _parse_body_atoms(match.group("body"))
    if not head_match or head_match.group("neg") or not body_atoms:
        return None
    head = Atom(
        head_match.group("rel"),
        (head_match.group("a"), head_match.group("b")),
    )
    return Rule(head, tuple(body_atoms))


def _parse_body_atoms(body: str) -> list[Atom] | None:
    atoms: list[Atom] = []
    pos = 0
    expecting_atom = True
    while pos < len(body):
        if body[pos].isspace():
            pos += 1
            continue
        if not expecting_atom:
            if body[pos] != ",":
                return None
            pos += 1
            expecting_atom = True
            continue
        atom = ATOM_RE.match(body, pos)
        if atom is None:
            return None
        atoms.append(
            Atom(
                atom.group("rel"),
                (atom.group("a"), atom.group("b")),
                atom.group("neg") == "!",
            )
        )
        pos = atom.end()
        expecting_atom = False
    if expecting_atom:
        return None
    return atoms


def _binary_args(atom: Atom) -> tuple[str, str] | None:
    if len(atom.args) != 2:
        return None
    return atom.args[0], atom.args[1]


def all_entities(graph: dict[str, list[list[str]] | list[tuple[str, str]]]) -> list[str]:
    entities = set()
    for pairs in graph.values():
        for src, dst in pairs:
            entities.add(src)
            entities.add(dst)
    return sorted(entities)


def build_relation(name_to_idx: dict[str, int], pairs) -> torch.Tensor:
    R = torch.zeros(len(name_to_idx), len(name_to_idx))
    for src, dst in pairs:
        if src in name_to_idx and dst in name_to_idx:
            R[name_to_idx[src], name_to_idx[dst]] = 1.0
    return R


def relation_tensors(graph: dict, name_to_idx: dict[str, int] | None = None):
    if name_to_idx is None:
        names = all_entities(graph)
        name_to_idx = {name: i for i, name in enumerate(names)}
    else:
        names = sorted(name_to_idx, key=name_to_idx.get)
    tensors = {rel: build_relation(name_to_idx, pairs) for rel, pairs in graph.items()}
    return tensors, names, name_to_idx


def evaluate_rule(graph: dict, rule: Rule, name_to_idx: dict[str, int] | None = None):
    """Evaluate a binary Datalog-style rule as a boolean relation tensor.

    Positive body atoms are joined with einsum. Negated atoms are restricted
    to the head variables and applied as element-wise set difference.
    """
    rel_tensors, names, name_to_idx = relation_tensors(graph, name_to_idx)
    pos_atoms = [atom for atom in rule.body if not atom.negated]
    neg_atoms = [atom for atom in rule.body if atom.negated]
    n = len(names)

    if pos_atoms:
        var_to_axis = {}
        operands = []
        operand_strs = []
        for atom in pos_atoms:
            args = _binary_args(atom)
            if args is None:
                return None, f"body relation {atom.relation} expects 2 args, got {len(atom.args)}"
            left, right = args
            if atom.relation not in rel_tensors:
                return None, f"unknown body relation: {atom.relation}"
            for var in (left, right):
                if var not in var_to_axis:
                    var_to_axis[var] = chr(ord("a") + len(var_to_axis))
            operands.append(rel_tensors[atom.relation])
            operand_strs.append(var_to_axis[left] + var_to_axis[right])
        head_args = _binary_args(rule.head)
        if head_args is None:
            return None, f"head relation {rule.head.relation} expects 2 args, got {len(rule.head.args)}"
        head_left, head_right = head_args
        if head_left not in var_to_axis or head_right not in var_to_axis:
            return None, f"head variables {head_left}, {head_right} not in positive body"
        equation = ",".join(operand_strs) + "->" + var_to_axis[head_left] + var_to_axis[head_right]
        result = (torch.einsum(equation, *operands) > 0).float()
    else:
        head_args = _binary_args(rule.head)
        if head_args is None:
            return None, f"head relation {rule.head.relation} expects 2 args, got {len(rule.head.args)}"
        head_left, head_right = head_args
        result = torch.ones(n, n)

    head_vars = {head_left, head_right}
    for atom in neg_atoms:
        args = _binary_args(atom)
        if args is None:
            return None, f"negated relation {atom.relation} expects 2 args, got {len(atom.args)}"
        left, right = args
        if atom.relation not in rel_tensors:
            return None, f"unknown negated relation: {atom.relation}"
        if {left, right} != head_vars:
            return None, (
                f"negated atom {atom.relation}({left}, {right}) must match head "
                f"variables {head_left}, {head_right}"
            )
        neg_tensor = rel_tensors[atom.relation]
        if (left, right) == (head_left, head_right):
            neg_aligned = neg_tensor
        else:
            neg_aligned = neg_tensor.t()
        result = result * (1.0 - neg_aligned)

    return ((result > 0).float(), names, name_to_idx, rule.head.relation), None


def query_relation(head_result, subj: str, obj: str) -> bool:
    head_tensor, _names, name_to_idx, _head_rel = head_result
    if subj not in name_to_idx or obj not in name_to_idx:
        return False
    return bool(head_tensor[name_to_idx[subj], name_to_idx[obj]] > 0)
