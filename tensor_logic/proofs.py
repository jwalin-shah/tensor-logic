from __future__ import annotations

import math

from dataclasses import dataclass
from collections import deque

from .program import Program, Atom, Rule

_PENDING = object()  # sentinel: goal is currently being proved (cycle guard)


@dataclass(frozen=True)
class Proof:
    head: tuple[str, str, str]
    body: tuple["Proof", ...] = ()
    source: object = None
    confidence: float = 1.0

    @classmethod
    def from_json(cls, d: dict) -> "Proof":
        head = tuple(d["head"])
        body = tuple(cls.from_json(child) for child in d.get("body", []))
        return cls(head=head, body=body, confidence=d.get("confidence", 1.0))

@dataclass(frozen=True)
class NegativeProof:
    head: tuple[str, str, str]
    body: tuple["NegativeProof", ...] = ()
    reason: str = ""

    @classmethod
    def from_json(cls, d: dict) -> "NegativeProof":
        explanation = d.get("explanation", d)
        head = tuple(explanation["head"])
        reason = explanation.get("reason", "")
        body = tuple(
            cls.from_json({"explanation": child["explanation"]} if "explanation" in child else child)
            for child in explanation.get("body", [])
        )
        return cls(head=head, body=body, reason=reason)


def prove(program: Program, relation_name: str, src: str, dst: str,
          recursive: bool = False, _table: dict | None = None,
          _do: dict | None = None) -> Proof | None:
    if relation_name not in program.relations:
        raise ValueError(f"relation '{relation_name}' not defined")
    relation = program.relations[relation_name]
    _check_symbols(relation, relation_name, src, dst)
    if _do is not None and (relation_name, src, dst) in _do:
        val = _do[(relation_name, src, dst)]
        return Proof((relation_name, src, dst), confidence=float(val)) if val > 0 else None
    val = relation.data[relation.domains[0].id(src), relation.domains[1].id(dst)].item()
    if val > 0:
        source = program.sources.get((relation_name, src, dst))
        return Proof((relation_name, src, dst), source=source, confidence=val)
    if relation_name not in program.rules:
        return None
    if _table is None:
        _table = {}
    key = (relation_name, src, dst)
    if key in _table:
        entry = _table[key]
        return None if entry is _PENDING else entry
    _table[key] = _PENDING
    result = None
    for rule in program.rules[relation_name]:
        proof = _prove_from_rule(program, relation_name, rule, src, dst, _table=_table, _do=_do)
        if proof is not None:
            result = proof
            break
    _table[key] = result
    if result is None and recursive:
        result = _prove_recursive_chain(program, relation_name, src, dst)
    return result

def prove_negative(program: Program, relation_name: str, src: str, dst: str,
                   recursive: bool = False,
                   _table: dict | None = None,
                   _neg_table: dict | None = None) -> NegativeProof | None:
    if relation_name not in program.relations:
        raise ValueError(f"relation '{relation_name}' not defined")
    relation = program.relations[relation_name]
    _check_symbols(relation, relation_name, src, dst)
    is_fact = relation.data[relation.domains[0].id(src), relation.domains[1].id(dst)].item() > 0
    if is_fact:
        return None
    if relation_name not in program.rules:
        return NegativeProof((relation_name, src, dst), reason="no_rules")
    if _table is None:
        _table = {}
    if _neg_table is None:
        _neg_table = {}
    key = (relation_name, src, dst)
    if key in _neg_table:
        entry = _neg_table[key]
        return None if entry is _PENDING else entry
    _neg_table[key] = _PENDING
    rule_failures = []
    for rule in program.rules[relation_name]:
        failure = _prove_negative_from_rule(program, relation_name, rule, src, dst,
                                            _table=_table, _neg_table=_neg_table)
        if failure is None:
            _neg_table[key] = None
            return None
        rule_failures.append(failure)
    if len(rule_failures) == 1:
        result = rule_failures[0]
    else:
        result = NegativeProof(
            (relation_name, src, dst),
            body=tuple(rule_failures),
            reason="all_rules_failed"
        )
    _neg_table[key] = result
    return result


def _prove_negative_from_rule(program: Program, relation_name: str, rule: Rule,
                               src: str, dst: str,
                               _table: dict | None = None,
                               _neg_table: dict | None = None) -> NegativeProof | None:
    bindings = _bind_variables(rule.head, src, dst)
    if bindings is None:
        return NegativeProof((relation_name, src, dst), reason="rule_head_mismatch")
    body_proof = _prove_negative_body_atoms(program, rule.body, bindings,
                                            _table=_table, _neg_table=_neg_table)
    if body_proof is not None:
        return NegativeProof((relation_name, src, dst), body=(body_proof,), reason="rule_body_failed")
    return None


def _prove_negative_body_atoms(program: Program, atoms: tuple[Atom, ...],
                                bindings: dict[str, str],
                                _table: dict | None = None,
                                _neg_table: dict | None = None) -> NegativeProof | None:
    unbound_vars = _find_unbound_vars(atoms, bindings)
    if not unbound_vars:
        return _try_prove_negative_body_atoms(program, atoms, bindings,
                                              _table=_table, _neg_table=_neg_table)
    var = unbound_vars.pop()
    witnesses = _get_witness_domain(program, atoms, bindings, var)
    failed_witnesses = []
    for symbol in witnesses:
        extended_bindings = {**bindings, var: symbol}
        neg_proof = _try_prove_negative_body_atoms(program, atoms, extended_bindings,
                                                   _table=_table, _neg_table=_neg_table)
        if neg_proof is None:
            return None
        failed_witnesses.append(neg_proof)
    if failed_witnesses:
        return NegativeProof(
            ("\u2203" + var, "", ""),
            body=tuple(failed_witnesses),
            reason="no_witness"
        )
    return None


def _try_prove_negative_body_atoms(program: Program, atoms: tuple[Atom, ...],
                                    bindings: dict[str, str],
                                    _table: dict | None = None,
                                    _neg_table: dict | None = None) -> NegativeProof | None:
    neg_proofs = []
    for atom in atoms:
        if len(atom.args) != 2:
            return NegativeProof((atom.relation, "", ""), reason="invalid_arity")
        bound_src = bindings.get(atom.args[0], atom.args[0])
        bound_dst = bindings.get(atom.args[1], atom.args[1])
        atom_proof = prove(program, atom.relation, bound_src, bound_dst,
                           recursive=False, _table=_table)
        if atom_proof is None:
            atom_neg_proof = prove_negative(program, atom.relation, bound_src, bound_dst,
                                            recursive=False, _table=_table, _neg_table=_neg_table)
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


def _prove_from_rule(program: Program, relation_name: str, rule: Rule, src: str, dst: str,
                     _table: dict | None = None, _do: dict | None = None) -> Proof | None:
    bindings = _bind_variables(rule.head, src, dst)
    if bindings is None:
        return None
    body_proofs = _prove_body_atoms(program, rule.body, bindings, _table=_table, _do=_do)
    if body_proofs is None:
        return None
    confidence = math.prod(p.confidence for p in body_proofs)
    return Proof((relation_name, src, dst), tuple(body_proofs), confidence=confidence)

def _check_symbols(relation, relation_name: str, src: str, dst: str) -> None:
    if len(relation.domains) < 2:
        return
    for i, (domain, symbol) in enumerate(zip(relation.domains, (src, dst))):
        if symbol not in domain.index:
            raise ValueError(f"symbol '{symbol}' not in domain for arg {i} of '{relation_name}' (known: {', '.join(domain.symbols)})")


def _bind_variables(head_atom: Atom, src: str, dst: str) -> dict[str, str] | None:
    if len(head_atom.args) != 2:
        return None
    bindings = {head_atom.args[0]: src, head_atom.args[1]: dst}
    return bindings


def _prove_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str],
                      _table: dict | None = None, _do: dict | None = None) -> list[Proof] | None:
    unbound_vars = _find_unbound_vars(atoms, bindings)
    if not unbound_vars:
        return _try_prove_body_atoms(program, atoms, bindings, _table=_table, _do=_do)
    var = unbound_vars.pop()
    for symbol in _get_witness_domain(program, atoms, bindings, var):
        extended_bindings = {**bindings, var: symbol}
        proofs = _try_prove_body_atoms(program, atoms, extended_bindings, _table=_table, _do=_do)
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


def _try_prove_body_atoms(program: Program, atoms: tuple[Atom, ...], bindings: dict[str, str],
                          _table: dict | None = None, _do: dict | None = None) -> list[Proof] | None:
    proofs = []
    for atom in atoms:
        if len(atom.args) != 2:
            return None
        bound_src = bindings.get(atom.args[0], atom.args[0])
        bound_dst = bindings.get(atom.args[1], atom.args[1])
        atom_proof = prove(program, atom.relation, bound_src, bound_dst, _table=_table, _do=_do)
        if atom_proof is None:
            return None
        proofs.append(atom_proof)
    return proofs


def prove_with_do(program: Program, relation_name: str, src: str, dst: str,
                  do: dict[tuple[str, str, str], float],
                  recursive: bool = False) -> Proof | None:
    """prove() under Pearl do()-interventions.

    do: {(relation_name, src, dst): value} — each entry asserts the fact at
    the given value and severs all rule derivations that would re-derive it.
    do(fact, 0) makes the fact unprovable regardless of rules in the program.
    """
    return prove(program, relation_name, src, dst, recursive=recursive, _do=do)


def fmt_proof_tree(proof: Proof, indent: int = 0) -> str:
    pad = "  " * indent
    rel, src, dst = proof.head
    conf_tag = f" ({proof.confidence:.2f})" if proof.confidence != 1.0 else ""
    source_tag = f"  [{proof.source.file}:{proof.source.lineno}]" if proof.source else ""
    lines = [f"{pad}{rel}({src}, {dst}){conf_tag}{source_tag}"]
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
