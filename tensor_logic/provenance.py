from itertools import product

from .rules import Rule, all_entities


def primitive_proofs(graph: dict, rel: str, subj: str, obj: str):
    if rel not in graph:
        return []
    for src, dst in graph[rel]:
        if src == subj and dst == obj:
            return [{"primitive": (rel, subj, obj)}]
    return []


def evaluate_with_provenance(
    graph: dict,
    rules: list[Rule],
    rel: str,
    subj: str,
    obj: str,
    max_depth: int = 6,
    top_k: int | None = None,
    sort: bool = False,
    _depth: int = 0,
):
    """Return proof trees for a query, optionally ranked and truncated."""
    if _depth > max_depth:
        return []

    proofs = primitive_proofs(graph, rel, subj, obj)
    entities = all_entities(graph)

    for rule in rules:
        if rule.head.rel != rel:
            continue
        bindings = {rule.head.left: subj, rule.head.right: obj}
        body_vars = {var for atom in rule.body for var in (atom.left, atom.right)}
        free_vars = [var for var in sorted(body_vars) if var not in bindings]
        for assignment in _enumerate_assignments(free_vars, entities):
            full_bindings = {**bindings, **assignment}
            body_proofs = _try_body(graph, rules, rule, full_bindings, max_depth, _depth)
            for body in body_proofs:
                proofs.append(
                    {
                        "head": (rel, subj, obj),
                        "rule": rule,
                        "body": body,
                    }
                )
                if top_k is not None and not sort and len(proofs) >= top_k:
                    return proofs

    if sort:
        proofs = sorted(proofs, key=proof_score)
    if top_k is not None:
        proofs = proofs[:top_k]
    return proofs


def _enumerate_assignments(vars_: list[str], entities: list[str]):
    if not vars_:
        yield {}
        return
    for values in product(entities, repeat=len(vars_)):
        yield dict(zip(vars_, values))


def _try_body(graph, rules, rule: Rule, bindings, max_depth, depth):
    per_atom = []
    for atom in rule.body:
        if atom.negated:
            if atom.left not in bindings or atom.right not in bindings:
                return []
            has_fact = bool(primitive_proofs(graph, atom.rel, bindings[atom.left], bindings[atom.right]))
            if has_fact:
                return []
            continue
        if atom.left not in bindings or atom.right not in bindings:
            return []
        atom_proofs = evaluate_with_provenance(
            graph,
            rules,
            atom.rel,
            bindings[atom.left],
            bindings[atom.right],
            max_depth=max_depth,
            _depth=depth + 1,
        )
        if not atom_proofs:
            return []
        per_atom.append(atom_proofs)
    if not per_atom:
        return [[]]
    return [list(items) for items in product(*per_atom)]


def proof_score(proof) -> tuple[int, int]:
    """Rank by rule firings, then total tree nodes."""
    return (_rule_firings(proof), _tree_nodes(proof))


def _rule_firings(proof) -> int:
    if "primitive" in proof:
        return 0
    return 1 + sum(_rule_firings(child) for child in proof["body"])


def _tree_nodes(proof) -> int:
    if "primitive" in proof:
        return 1
    return 1 + sum(_tree_nodes(child) for child in proof["body"])


def fmt_proof(proof, indent: int = 0) -> str:
    pad = "  " * indent
    if "primitive" in proof:
        rel, subj, obj = proof["primitive"]
        return f"{pad}- {rel}({subj}, {obj})  [primitive]"
    head_rel, subj, obj = proof["head"]
    rule = proof["rule"]
    body = ", ".join(
        f"{'!' if atom.negated else ''}{atom.rel}({atom.left},{atom.right})"
        for atom in rule.body
    )
    lines = [f"{pad}- {head_rel}({subj}, {obj})  via  {rule.head.rel}({rule.head.left}, {rule.head.right}) :- {body}"]
    for child in proof["body"]:
        lines.append(fmt_proof(child, indent + 1))
    return "\n".join(lines)
