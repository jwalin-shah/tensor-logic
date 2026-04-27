from __future__ import annotations

from dataclasses import dataclass
from collections import deque

from .program import Program, Atom, Rule


@dataclass(frozen=True)
class Proof:
    head: tuple[str, str, str]
    body: tuple["Proof", ...] = ()

@dataclass(frozen=True)
class NegativeProof:
    head: tuple[str, str, str]
    body: tuple["NegativeProof", ...] = ()
    reason: str = ""


def prove(program: Program, relation_name: str, src: str, dst: str, recursive: bool = False) -> Proof | None:
    relation = program.relations[relation_name]
    if relation.data[relation.domains[0].id(src), relation.domains[1].id(dst)].item() > 0:
        return Proof((relation_name, src, dst))
    if relation_name in program.rules:
        proof = _prove_from_rule(program, relation_name, src, dst)
        if proof is not None:
            return proof
    if recursive:
        return _prove_recursive_chain(program, relation_name, src, dst)
    return None

def prove_negative(program: Program, relation_name: str, src: str, dst: str, recursive: bool = False) -> NegativeProof | None:
    relation = program.relations[relation_name]
    is_fact = relation.data[relation.domains[0].id(src), relation.domains[1].id(dst)].item() > 0
    
    if is_fact:
        return None
    
    if relation_name not in program.rules:
        return NegativeProof((relation_name, src, dst), reason="no_rules")
    
    rule = program.rules[relation_name]
    rule_failure = _prove_negative_from_rule(program, relation_name, rule, src, dst)
    
    if rule_failure is not None:
        return rule_failure
    
    if recursive:
        neg_recursive = _prove_negative_recursive_chain(program, relation_name, src, dst)
        if neg_recursive is not None:
            return neg_recursive
    
    return None


def _prove_negative_from_rule(program: Program, relation_name: str, rule: Rule, src: str, dst: str) -> NegativeProof | None:
    bindings = _bind_variables(rule.head, src, dst)
    if bindings is None:
        return NegativeProof((relation_name, src, dst), reason="rule_head_mismatch")
    
    body_proof = _prove_negative_body_atoms(program, rule.body, bindings)
    if body_proof is not None:
        return NegativeProof((relation_name, src, dst), body=(body_proof,), reason="rule_body_failed")
    
    return None


def _prove_negative_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str]) -> NegativeProof | None:
    unbound_vars = _find_unbound_vars(atoms, bindings)
    
    if not unbound_vars:
        return _try_prove_negative_body_atoms(program, atoms, bindings)
    
    var = unbound_vars.pop()
    witnesses = _get_witness_domain(program, atoms, bindings, var)
    failed_witnesses = []
    
    for symbol in witnesses:
        extended_bindings = {**bindings, var: symbol}
        neg_proof = _try_prove_negative_body_atoms(program, atoms, extended_bindings)
        if neg_proof is None:
            return None
        failed_witnesses.append(neg_proof)
    
    if failed_witnesses:
        return NegativeProof(
            ("∃" + var, "", ""),
            body=tuple(failed_witnesses),
            reason="no_witness"
        )
    
    return None


def _try_prove_negative_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str]) -> NegativeProof | None:
    neg_proofs = []
    for atom in atoms:
        if len(atom.args) != 2:
            return NegativeProof(
                (atom.relation, "", ""),
                reason="invalid_arity"
            )
        bound_src = bindings.get(atom.args[0], atom.args[0])
        bound_dst = bindings.get(atom.args[1], atom.args[1])
        
        atom_proof = prove(program, atom.relation, bound_src, bound_dst, recursive=False)
        if atom_proof is None:
            atom_neg_proof = prove_negative(program, atom.relation, bound_src, bound_dst, recursive=False)
            if atom_neg_proof is not None:
                neg_proofs.append(atom_neg_proof)
            else:
                neg_proofs.append(NegativeProof(
                    (atom.relation, bound_src, bound_dst),
                    reason="not_provable"
                ))
            return NegativeProof(
                (atoms[0].relation, "", ""),
                body=tuple(neg_proofs),
                reason="atom_failed"
            )
    
    return None


def _prove_negative_recursive_chain(program: Program, relation_name: str, src: str, dst: str) -> NegativeProof | None:
    relation = program.relations[relation_name]
    if len(relation.domains) != 2:
        return None
    
    base = _find_base_relation(program, relation)
    if base is None:
        return NegativeProof((relation_name, src, dst), reason="no_base_relation")
    
    start = base.domains[0].id(src)
    goal = base.domains[1].id(dst)
    queue = deque([(start, [start])])
    seen = {start}
    
    while queue:
        current, path = queue.popleft()
        if current == goal and len(path) > 1:
            return None
        
        row = base.data[current]
        for nxt, value in enumerate(row):
            if value.item() <= 0 or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    
    return NegativeProof((relation_name, src, dst), reason="no_recursive_path")


def _prove_from_rule(program: Program, relation_name: str, src: str, dst: str) -> Proof | None:
    rule = program.rules[relation_name]
    bindings = _bind_variables(rule.head, src, dst)
    if bindings is None:
        return None
    body_proofs = _prove_body_atoms(program, rule.body, bindings)
    if body_proofs is None:
        return None
    return Proof((relation_name, src, dst), tuple(body_proofs))


def _bind_variables(head_atom: Atom, src: str, dst: str) -> dict[str, str] | None:
    if len(head_atom.args) != 2:
        return None
    bindings = {head_atom.args[0]: src, head_atom.args[1]: dst}
    return bindings


def _prove_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str]) -> list[Proof] | None:
    unbound_vars = _find_unbound_vars(atoms, bindings)
    if not unbound_vars:
        return _try_prove_body_atoms(program, atoms, bindings)
    var = unbound_vars.pop()
    for symbol in _get_witness_domain(program, atoms, bindings, var):
        extended_bindings = {**bindings, var: symbol}
        proofs = _try_prove_body_atoms(program, atoms, extended_bindings)
        if proofs is not None:
            return proofs
    return None


def _find_unbound_vars(atoms: tuple[Atom, ...], bindings: dict[str, str]) -> set[str]:
    unbound = set()
    for atom in atoms:
        for arg in atom.args:
            if arg not in bindings:
                unbound.add(arg)
    return unbound


def _get_witness_domain(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str], var: str) -> list[str]:
    for atom in atoms:
        for i, arg in enumerate(atom.args):
            if arg == var:
                rel = program.relations[atom.relation]
                domain = rel.domains[i]
                return list(domain.symbols)
    return []


def _try_prove_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str]) -> list[Proof] | None:
    proofs = []
    for atom in atoms:
        if len(atom.args) != 2:
            return None
        bound_src = bindings.get(atom.args[0], atom.args[0])
        bound_dst = bindings.get(atom.args[1], atom.args[1])
        atom_proof = prove(program, atom.relation, bound_src, bound_dst, recursive=False)
        if atom_proof is None:
            return None
        proofs.append(atom_proof)
    return proofs


def fmt_proof_tree(proof: Proof, indent: int = 0) -> str:
    pad = "  " * indent
    rel, src, dst = proof.head
    lines = [f"{pad}{rel}({src}, {dst})"]
    for child in proof.body:
        lines.append(fmt_proof_tree(child, indent + 1))
    return "\n".join(lines)

def fmt_negative_proof_tree(neg_proof: NegativeProof, indent: int = 0) -> str:
    pad = "  " * indent
    rel, src, dst = neg_proof.head
    
    if rel.startswith("∃"):
        lines = [f"{pad}[Witness enumeration failed]"]
    else:
        lines = [f"{pad}{rel}({src}, {dst}) = False"]
    
    if neg_proof.reason:
        lines.append(f"{pad}  reason: {neg_proof.reason}")
    
    for child in neg_proof.body:
        lines.append(fmt_negative_proof_tree(child, indent + 1))
    
    return "\n".join(lines)


def _prove_recursive_chain(program: Program, relation_name: str, src: str, dst: str) -> Proof | None:
    relation = program.relations[relation_name]
    if len(relation.domains) != 2:
        return None
    base = _find_base_relation(program, relation)
    if base is None:
        return None
    start = base.domains[0].id(src)
    goal = base.domains[1].id(dst)
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        if current == goal and len(path) > 1:
            return _path_to_proof(relation_name, base.name, base.domains[0].symbols, path)
        row = base.data[current]
        for nxt, value in enumerate(row):
            if value.item() <= 0 or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def _find_base_relation(program: Program, recursive_relation) -> object | None:
    for relation in program.relations.values():
        if relation is recursive_relation:
            continue
        if len(relation.domains) == 2 and relation.domains == recursive_relation.domains:
            return relation
    return None


def _path_to_proof(recursive_name: str, base_name: str, symbols: tuple[str, ...], path: list[int]) -> Proof:
    facts = tuple(
        Proof((base_name, symbols[path[i]], symbols[path[i + 1]]))
        for i in range(len(path) - 1)
    )
    return Proof((recursive_name, symbols[path[0]], symbols[path[-1]]), facts)
