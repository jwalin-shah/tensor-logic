"""Restricted fixed-point kernel vs fuel-bounded script execution.

This module supports experiment #99. The restricted kernel operates over a
finite relation and therefore reaches a fixed point in at most |V| iterations
for simple transitive closure. The script-like runner accepts arbitrary state
transition functions but requires an external fuel/watchdog to prevent
non-termination.

The latter is not itself a proof of Turing completeness; it is the operational
baseline for comparing a trusted terminating fragment with unrestricted
host-language computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class FixedPointResult:
    relation: frozenset[tuple[str, str]]
    iterations: int
    additions: int
    converged: bool = True


@dataclass(frozen=True)
class ScriptResult(Generic[T]):
    state: T
    steps: int
    halted: bool
    fuel_exhausted: bool


def transitive_closure(
    edges: Iterable[tuple[str, str]],
) -> FixedPointResult:
    relation = set(edges)
    nodes = {node for edge in relation for node in edge}
    if not relation:
        return FixedPointResult(
            relation=frozenset(),
            iterations=0,
            additions=0,
        )

    additions = 0
    iterations = 0
    while True:
        iterations += 1
        by_left: dict[str, set[str]] = {}
        by_right: dict[str, set[str]] = {}
        for left, right in relation:
            by_left.setdefault(left, set()).add(right)
            by_right.setdefault(right, set()).add(left)

        new_pairs: set[tuple[str, str]] = set()
        for middle in nodes:
            for left in by_right.get(middle, ()):
                for right in by_left.get(middle, ()):
                    pair = (left, right)
                    if pair not in relation:
                        new_pairs.add(pair)

        if not new_pairs:
            return FixedPointResult(
                relation=frozenset(relation),
                iterations=iterations,
                additions=additions,
            )
        relation.update(new_pairs)
        additions += len(new_pairs)

        # Defensive invariant: a binary relation over a finite domain cannot
        # contain more than |V|^2 distinct tuples.
        if len(relation) > len(nodes) ** 2:
            raise AssertionError("finite relation exceeded domain bound")


def run_with_fuel(
    initial_state: T,
    step_fn: Callable[[T], tuple[T, bool]],
    *,
    fuel: int,
) -> ScriptResult[T]:
    if fuel < 0:
        raise ValueError("fuel cannot be negative")
    state = initial_state
    for step in range(1, fuel + 1):
        state, halted = step_fn(state)
        if halted:
            return ScriptResult(
                state=state,
                steps=step,
                halted=True,
                fuel_exhausted=False,
            )
    return ScriptResult(
        state=state,
        steps=fuel,
        halted=False,
        fuel_exhausted=True,
    )


def finite_closure_iteration_bound(node_count: int) -> int:
    """Loose finite upper bound for monotonically adding binary tuples."""
    if node_count < 0:
        raise ValueError("node_count cannot be negative")
    return node_count * node_count
