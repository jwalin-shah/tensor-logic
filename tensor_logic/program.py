from __future__ import annotations

from dataclasses import dataclass
import re

from .language import Domain, Relation, as_expr


TOKEN_RE = re.compile(r"\s*(?:(?P<ID>[A-Za-z_]\w*)|(?P<ASSIGN>:=)|(?P<LP>\()|(?P<RP>\))|(?P<COMMA>,)|(?P<PLUS>\+)|(?P<STAR>\*)|(?P<DOT>\.)|(?P<NUMBER>\d+(?:\.\d+)?))")


@dataclass(frozen=True)
class Atom:
    relation: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom, ...]


@dataclass(frozen=True)
class FactSource:
    file: str
    lineno: int


@dataclass(frozen=True)
class Program:
    domains: dict[str, Domain]
    relations: dict[str, Relation]
    rules: dict[str, list[Rule]]
    sources: dict[tuple, FactSource]

    def __init__(self):
        object.__setattr__(self, "domains", {})
        object.__setattr__(self, "relations", {})
        object.__setattr__(self, "rules", {})
        object.__setattr__(self, "sources", {})

    def domain(self, name: str, symbols) -> Domain:
        domain = Domain(symbols)
        self.domains[name] = domain
        return domain

    def relation(self, name: str, *domain_names: str) -> Relation:
        if name in self.relations:
            return self.relations[name]
        for d in domain_names:
            if d not in self.domains:
                raise ValueError(f"domain '{d}' not defined (known: {', '.join(self.domains) or 'none'})")
        domains = tuple(self.domains[domain_name] for domain_name in domain_names)
        relation = Relation(name, *domains)
        self.relations[name] = relation
        return relation

    def fact(self, relation_name: str, *symbols: str, value: float = 1.0, source: FactSource | None = None) -> None:
        if relation_name not in self.relations:
            raise ValueError(f"relation '{relation_name}' not defined (known: {', '.join(self.relations) or 'none'})")
        rel = self.relations[relation_name]
        if len(symbols) != len(rel.domains):
            raise ValueError(f"relation '{relation_name}' expects {len(rel.domains)} args, got {len(symbols)}")
        for i, (domain, symbol) in enumerate(zip(rel.domains, symbols)):
            if symbol not in domain.index:
                raise ValueError(f"symbol '{symbol}' not in domain for arg {i} of '{relation_name}' (known: {', '.join(domain.symbols)})")
        rel[symbols] = value
        if source is not None:
            self.sources[(relation_name, *symbols)] = source

    def rule(self, text: str) -> None:
        head, expr = RuleParser(self).parse_rule(text)
        relation_name, indices = head
        self.relations[relation_name][indices] = expr
        for rule_ast in RuleParser(self).parse_rule_ast_list(text):
            self.rules.setdefault(rule_ast.head.relation, []).append(rule_ast)

    def eval(self, relation_name: str, *indices: str, semiring: str = "boolean"):
        return self.relations[relation_name].eval(*indices, semiring=semiring)

    def fixpoint(self, relation_name: str, *indices: str, semiring: str = "boolean", max_iters: int | None = None):
        return self.relations[relation_name].fixpoint(*indices, semiring=semiring, max_iters=max_iters)

    def query(self, relation_name: str, *symbols: str, recursive: bool = False, semiring: str = "boolean") -> float:
        relation = self.relations[relation_name]
        if recursive:
            return relation.reachable(*symbols, semiring=semiring)
        return relation.value(*symbols, semiring=semiring)


class RuleParser:
    def __init__(self, program: Program):
        self.program = program
        self.tokens: list[tuple[str, str]] = []
        self.pos = 0

    def parse_rule(self, text: str):
        self.tokens = _tokenize(text)
        self.pos = 0
        head = self._parse_head()
        self._expect("ASSIGN")
        expr = self._parse_sum()
        self._expect("EOF")
        return head, expr


    def parse_rule_ast(self, text: str) -> Rule:
        self.tokens = _tokenize(text)
        self.pos = 0
        head_name, head_indices = self._parse_head()
        self._expect("ASSIGN")
        body_atoms = self._extract_atoms_from_expr()
        self._expect("EOF")
        head_atom = Atom(head_name, head_indices)
        return Rule(head_atom, tuple(body_atoms))

    def parse_rule_ast_list(self, text: str) -> list[Rule]:
        """Return one Rule per OR-disjunct in the rule body."""
        self.tokens = _tokenize(text)
        self.pos = 0
        head_name, head_indices = self._parse_head()
        self._expect("ASSIGN")
        head_atom = Atom(head_name, head_indices)
        groups = self._split_disjuncts()
        rules = []
        for group_toks in groups:
            saved_tokens, saved_pos = self.tokens, self.pos
            self.tokens = group_toks + [("EOF", "")]
            self.pos = 0
            atoms: list[Atom] = []
            self._extract_atoms_recursive(atoms)
            self.tokens, self.pos = saved_tokens, saved_pos
            if atoms:
                rules.append(Rule(head_atom, tuple(atoms)))
        return rules or [Rule(head_atom, ())]

    def _split_disjuncts(self) -> list[list[tuple[str, str]]]:
        if self._peek()[0] == "LP" and self._is_wrapping_paren():
            self.pos += 1
            groups = self._collect_split_tokens(stop_at_rp=True)
            self._expect("RP")
            while self._accept("DOT") is not None:
                self._expect("ID"); self._expect("LP"); self._expect("RP")
        else:
            groups = self._collect_split_tokens(stop_at_rp=False)
        return groups

    def _is_wrapping_paren(self) -> bool:
        i, depth = self.pos + 1, 1
        while i < len(self.tokens):
            t = self.tokens[i][0]
            if t == "LP": depth += 1
            elif t == "RP":
                depth -= 1
                if depth == 0:
                    j = i + 1
                    while j + 3 < len(self.tokens) and self.tokens[j][0] == "DOT":
                        j += 4
                    return self.tokens[j][0] == "EOF"
            elif t == "EOF":
                break
            i += 1
        return False

    def _collect_split_tokens(self, stop_at_rp: bool) -> list[list[tuple[str, str]]]:
        groups: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        depth = 0
        while True:
            t, v = self._peek()
            if t == "EOF": break
            if t == "RP" and depth == 0 and stop_at_rp: break
            self.pos += 1
            if t == "LP": depth += 1; current.append((t, v))
            elif t == "RP": depth -= 1; current.append((t, v))
            elif t == "PLUS" and depth == 0: groups.append(current); current = []
            else: current.append((t, v))
        groups.append(current)
        return groups

    def _extract_atoms_from_expr(self) -> list[Atom]:
        atoms = []
        self._extract_atoms_recursive(atoms)
        return atoms

    def _extract_atoms_recursive(self, atoms: list[Atom]) -> None:
        self._parse_product_atoms(atoms)
        while self._accept("PLUS") is not None:
            self._parse_product_atoms(atoms)

    def _parse_product_atoms(self, atoms: list[Atom]) -> None:
        self._parse_factor_atoms(atoms)
        while self._accept("STAR") is not None:
            self._parse_factor_atoms(atoms)

    def _parse_factor_atoms(self, atoms: list[Atom]) -> None:
        if self._accept("LP") is not None:
            self._extract_atoms_recursive(atoms)
            self._expect("RP")
        elif self._peek()[0] == "NUMBER":
            self._expect("NUMBER")
        else:
            name = self._expect("ID")
            if name in self.program.relations:
                indices = self._parse_arg_list()
                atoms.append(Atom(name, indices))
            else:
                self._parse_arg_list()
        while self._accept("DOT") is not None:
            method = self._expect("ID")
            self._expect("LP")
            self._expect("RP")

    def _parse_head(self):
        name = self._expect("ID")
        indices = self._parse_arg_list()
        if name not in self.program.relations:
            raise ValueError(f"unknown relation in rule head: {name}")
        if len(indices) != len(self.program.relations[name].domains):
            raise ValueError(f"{name} expects {len(self.program.relations[name].domains)} indices, got {len(indices)}")
        return name, tuple(indices)

    def _parse_sum(self):
        expr = self._parse_product()
        while self._accept("PLUS") is not None:
            expr = expr + self._parse_product()
        return expr

    def _parse_product(self):
        expr = self._parse_factor()
        while self._accept("STAR") is not None:
            expr = expr * self._parse_factor()
        return expr

    def _parse_factor(self):
        if self._accept("LP") is not None:
            expr = self._parse_sum()
            self._expect("RP")
        elif self._peek()[0] == "NUMBER":
            expr = as_expr(float(self._expect("NUMBER")))
        else:
            expr = self._parse_relation_ref()
        while self._accept("DOT") is not None:
            method = self._expect("ID")
            self._expect("LP")
            self._expect("RP")
            if method == "step":
                expr = expr.step()
            elif method == "sigmoid":
                expr = expr.sigmoid()
            else:
                raise ValueError(f"unknown expression method: {method}")
        return expr

    def _parse_relation_ref(self):
        name = self._expect("ID")
        if name not in self.program.relations:
            raise ValueError(f"unknown relation: {name}")
        indices = self._parse_arg_list()
        relation = self.program.relations[name]
        if len(indices) != len(relation.domains):
            raise ValueError(f"{name} expects {len(relation.domains)} indices, got {len(indices)}")
        return relation[tuple(indices)]

    def _parse_arg_list(self):
        self._expect("LP")
        args = [self._expect("ID")]
        while self._accept("COMMA") is not None:
            args.append(self._expect("ID"))
        self._expect("RP")
        return tuple(args)

    def _peek(self):
        return self.tokens[self.pos]

    def _accept(self, token_type: str):
        if self._peek()[0] == token_type:
            value = self._peek()[1]
            self.pos += 1
            return value
        return None

    def _expect(self, token_type: str) -> str:
        actual_type, value = self._peek()
        if actual_type != token_type:
            raise ValueError(f"expected {token_type}, got {actual_type} ({value!r})")
        self.pos += 1
        return value


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens = []
    pos = 0
    while pos < len(text):
        match = TOKEN_RE.match(text, pos)
        if not match:
            raise ValueError(f"unexpected character at {pos}: {text[pos:pos + 20]!r}")
        pos = match.end()
        for token_type, value in match.groupdict().items():
            if value is not None:
                tokens.append((token_type, value))
                break
    tokens.append(("EOF", ""))
    return tokens
