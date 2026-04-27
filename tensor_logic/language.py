from __future__ import annotations

from dataclasses import dataclass
import torch


LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class Domain:
    symbols: tuple[str, ...]

    def __init__(self, symbols):
        object.__setattr__(self, "symbols", tuple(symbols))
        object.__setattr__(self, "index", {symbol: i for i, symbol in enumerate(symbols)})

    def __len__(self) -> int:
        return len(self.symbols)

    def id(self, symbol: str) -> int:
        return self.index[symbol]


@dataclass(frozen=True)
class TensorRef:
    tensor: "Relation"
    indices: tuple[str, ...]

    def step(self) -> "Step":
        return Step(self)

    def sigmoid(self) -> "Sigmoid":
        return Sigmoid(self)

    def __mul__(self, other) -> "Product":
        return Product(as_expr(self), as_expr(other))

    def __add__(self, other) -> "Sum":
        return Sum(as_expr(self), as_expr(other))

    def __sub__(self, other) -> "Difference":
        return Difference(as_expr(self), as_expr(other))

    def __rsub__(self, other) -> "Difference":
        return Difference(as_expr(other), as_expr(self))


@dataclass(frozen=True)
class Scalar:
    value: float


@dataclass(frozen=True)
class Product:
    left: Expr
    right: Expr

    def step(self) -> "Step":
        return Step(self)

    def sigmoid(self) -> "Sigmoid":
        return Sigmoid(self)

    def __mul__(self, other) -> "Product":
        return Product(as_expr(self), as_expr(other))

    def __add__(self, other) -> "Sum":
        return Sum(as_expr(self), as_expr(other))


@dataclass(frozen=True)
class Sum:
    left: Expr
    right: Expr

    def step(self) -> "Step":
        return Step(self)

    def sigmoid(self) -> "Sigmoid":
        return Sigmoid(self)

    def __add__(self, other) -> "Sum":
        return Sum(as_expr(self), as_expr(other))

    def __mul__(self, other) -> "Product":
        return Product(as_expr(self), as_expr(other))


@dataclass(frozen=True)
class Difference:
    left: Expr
    right: Expr

    def step(self) -> "Step":
        return Step(self)

    def sigmoid(self) -> "Sigmoid":
        return Sigmoid(self)


@dataclass(frozen=True)
class Step:
    expr: Expr


@dataclass(frozen=True)
class Sigmoid:
    expr: Expr


Expr = TensorRef | Scalar | Product | Sum | Difference | Step | Sigmoid


def as_expr(value) -> Expr:
    if isinstance(value, (TensorRef, Scalar, Product, Sum, Difference, Step, Sigmoid)):
        return value
    if isinstance(value, (int, float)):
        return Scalar(float(value))
    raise TypeError(f"cannot convert {type(value).__name__} to tensor-logic expression")


class Relation:
    """Named relation tensor over one or more finite domains."""

    def __init__(self, name: str, *domains: Domain):
        if not domains:
            raise ValueError("Relation requires at least one domain")
        self.name = name
        self.domains = tuple(domains)
        self.data = torch.zeros(*(len(domain) for domain in domains), dtype=torch.float32)
        self.equations: list[Equation] = []

    def __getitem__(self, indices) -> TensorRef:
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != len(self.domains):
            raise ValueError(f"{self.name} expects {len(self.domains)} indices, got {len(indices)}")
        return TensorRef(self, tuple(indices))

    def __setitem__(self, indices, value) -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != len(self.domains):
            raise ValueError(f"{self.name} expects {len(self.domains)} indices, got {len(indices)}")
        if all(_is_constant(domain, idx) for domain, idx in zip(self.domains, indices)):
            coords = tuple(domain.id(idx) for domain, idx in zip(self.domains, indices))
            self.data[coords] = float(value)
            return
        self.equations.append(Equation(self, tuple(indices), as_expr(value)))

    def eval(self, *indices: str, semiring: str = "real") -> torch.Tensor:
        if not indices:
            indices = self.default_indices()
        result = self.data.clone()
        for equation in self.equations:
            value = evaluate_expr(equation.expr, equation.indices, semiring=semiring)
            result = result + value
        if semiring == "boolean":
            result = (result > 0).float()
        return _reorder(result, self.default_indices(), tuple(indices))

    def value(self, *symbols: str, semiring: str = "boolean") -> float:
        tensor = self.eval(semiring=semiring)
        coords = tuple(domain.id(symbol) for domain, symbol in zip(self.domains, symbols))
        return float(tensor[coords].item())

    def fixpoint(
        self,
        *indices: str,
        semiring: str = "boolean",
        max_iters: int | None = None,
        include_facts: bool = True,
    ) -> torch.Tensor:
        if not indices:
            indices = self.default_indices()
        if max_iters is None:
            max_iters = max(len(domain) for domain in self.domains) + 1
        seed = self.data.clone() if include_facts else torch.zeros_like(self.data)
        result = seed
        for _ in range(max_iters):
            env = {self: result}
            next_result = seed.clone()
            for equation in self.equations:
                value = evaluate_expr(equation.expr, equation.indices, semiring=semiring, env=env)
                next_result = next_result + value
            if semiring == "boolean":
                next_result = (next_result > 0).float()
            if torch.equal(next_result, result):
                break
            result = next_result
        return _reorder(result, self.default_indices(), tuple(indices))

    def reachable(self, *symbols: str, semiring: str = "boolean", max_iters: int | None = None) -> float:
        tensor = self.fixpoint(semiring=semiring, max_iters=max_iters)
        coords = tuple(domain.id(symbol) for domain, symbol in zip(self.domains, symbols))
        return float(tensor[coords].item())

    def default_indices(self) -> tuple[str, ...]:
        return tuple(chr(ord("a") + i) for i in range(len(self.domains)))


@dataclass(frozen=True)
class Equation:
    target: Relation
    indices: tuple[str, ...]
    expr: Expr


def facts(relation: Relation, pairs) -> Relation:
    for fact in pairs:
        relation[tuple(fact)] = 1.0
    return relation


def evaluate_expr(
    expr: Expr,
    out_indices: tuple[str, ...],
    semiring: str = "real",
    env: dict["Relation", torch.Tensor] | None = None,
) -> torch.Tensor:
    expr = as_expr(expr)
    if isinstance(expr, Scalar):
        return torch.tensor(expr.value, dtype=torch.float32)
    if isinstance(expr, TensorRef):
        if env is not None and expr.tensor in env:
            data = env[expr.tensor]
        else:
            data = expr.tensor.eval(semiring=semiring)
        return _reorder(data, expr.tensor.default_indices(), out_indices, aliases=expr.indices)
    if isinstance(expr, Product):
        return _einsum_product(expr.left, expr.right, out_indices, semiring, env)
    if isinstance(expr, Sum):
        left = evaluate_expr(expr.left, out_indices, semiring, env)
        right = evaluate_expr(expr.right, out_indices, semiring, env)
        if semiring == "boolean":
            return ((left + right) > 0).float()
        return left + right
    if isinstance(expr, Difference):
        return evaluate_expr(expr.left, out_indices, semiring, env) - evaluate_expr(expr.right, out_indices, semiring, env)
    if isinstance(expr, Step):
        return (evaluate_expr(expr.expr, out_indices, semiring, env) > 0).float()
    if isinstance(expr, Sigmoid):
        return torch.sigmoid(evaluate_expr(expr.expr, out_indices, semiring, env))
    raise TypeError(f"unknown expression type: {type(expr).__name__}")


def _einsum_product(
    left: Expr,
    right: Expr,
    out_indices: tuple[str, ...],
    semiring: str,
    env: dict["Relation", torch.Tensor] | None,
) -> torch.Tensor:
    left_native = tuple(_collect_indices(left))
    right_native = tuple(_collect_indices(right))
    left_value = evaluate_expr(left, left_native, semiring, env)
    right_value = evaluate_expr(right, right_native, semiring, env)
    all_indices = tuple(dict.fromkeys(left_native + right_native + out_indices))
    if len(all_indices) > len(LETTERS):
        raise ValueError("too many distinct indices for einsum")
    idx_to_letter = {idx: LETTERS[i] for i, idx in enumerate(all_indices)}
    equation = (
        "".join(idx_to_letter[idx] for idx in left_native)
        + ","
        + "".join(idx_to_letter[idx] for idx in right_native)
        + "->"
        + "".join(idx_to_letter[idx] for idx in out_indices)
    )
    out = torch.einsum(equation, left_value, right_value)
    if semiring == "boolean":
        out = (out > 0).float()
    return out


def _collect_indices(expr: Expr) -> list[str]:
    expr = as_expr(expr)
    if isinstance(expr, TensorRef):
        return list(expr.indices)
    if isinstance(expr, Scalar):
        return []
    if isinstance(expr, (Product, Sum, Difference)):
        return list(dict.fromkeys(_collect_indices(expr.left) + _collect_indices(expr.right)))
    if isinstance(expr, (Step, Sigmoid)):
        return _collect_indices(expr.expr)
    raise TypeError(f"unknown expression type: {type(expr).__name__}")


def _reorder(
    tensor: torch.Tensor,
    current: tuple[str, ...],
    desired: tuple[str, ...],
    aliases: tuple[str, ...] | None = None,
) -> torch.Tensor:
    if aliases is not None:
        if len(aliases) != len(current):
            raise ValueError(f"alias arity {aliases} does not match indices {current}")
        current = aliases
    if current == desired:
        return tensor
    if set(current) != set(desired):
        raise ValueError(f"cannot reorder indices {current} to {desired}")
    perm = [current.index(idx) for idx in desired]
    return tensor.permute(*perm)


def _is_constant(domain: Domain, idx: str) -> bool:
    return isinstance(idx, str) and idx in domain.index
