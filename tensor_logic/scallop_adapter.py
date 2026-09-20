"""Optional adapter contract for the upstream Scallop runtime.

Pinned source inspected:
  scallop-lang/scallop@668bfb6d45ce302fd4ffa7f29916baf3c7ce36ef

This module does not reimplement Scallop. It normalizes our experiment inputs to
the public scallopy API and fails clearly when scallopy is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Iterable, Sequence


SCALLOP_UPSTREAM_COMMIT = (
    "668bfb6d45ce302fd4ffa7f29916baf3c7ce36ef"
)


@dataclass(frozen=True)
class DebugFactSpec:
    fact_id: int
    probability: float
    fact: tuple[str, str]

    def __post_init__(self) -> None:
        if self.fact_id <= 0:
            raise ValueError("Scallop debug fact IDs must start at 1")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0,1]")


@dataclass(frozen=True)
class ScallopClosureResult:
    relation: tuple[tuple[str, str], ...]
    provenance: str
    upstream_commit: str = SCALLOP_UPSTREAM_COMMIT


def scallopy_available() -> bool:
    try:
        importlib.import_module("scallopy")
        return True
    except ImportError:
        return False


def edge_path_program() -> str:
    return (
        "rel path(a, c) = edge(a, c) or "
        "path(a, b) and edge(b, c)"
    )


def run_unit_transitive_closure(
    edges: Iterable[tuple[str, str]],
) -> ScallopClosureResult:
    """Run the canonical recursive edge/path program in upstream Scallop."""
    try:
        scallopy = importlib.import_module("scallopy")
    except ImportError as exc:
        raise RuntimeError(
            "scallopy is not installed; build the pinned upstream Scallop "
            "runtime before executing this adapter"
        ) from exc

    ctx = scallopy.ScallopContext(provenance="unit")
    ctx.add_relation("edge", (str, str))
    ctx.add_facts("edge", list(edges))
    ctx.add_rule(edge_path_program())
    ctx.run()

    relation = tuple(
        sorted(
            tuple(row)
            for row in ctx.relation("path")
        )
    )
    return ScallopClosureResult(
        relation=relation,
        provenance="unit",
    )


def debug_fact_specs(
    edges: Sequence[tuple[str, str]],
    probabilities: Sequence[float],
) -> tuple[DebugFactSpec, ...]:
    """Assign the contiguous fact IDs required by difftopkproofsdebug."""
    if len(edges) != len(probabilities):
        raise ValueError("edge/probability counts differ")
    return tuple(
        DebugFactSpec(
            fact_id=index + 1,
            probability=float(probability),
            fact=tuple(edge),
        )
        for index, (edge, probability) in enumerate(
            zip(edges, probabilities)
        )
    )


def to_difftopk_debug_facts(
    specs: Sequence[DebugFactSpec],
):
    """Convert neutral specs to the exact tagged-fact shape Scallop documents.

    Result element shape:
      ((torch.tensor(probability), fact_id), (source, target))
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to construct differentiable Scallop tags"
        ) from exc

    expected_ids = tuple(range(1, len(specs) + 1))
    actual_ids = tuple(spec.fact_id for spec in specs)
    if actual_ids != expected_ids:
        raise ValueError(
            "Scallop debug fact IDs must be distinct, contiguous, and start at 1"
        )

    return [
        (
            (torch.tensor(spec.probability), spec.fact_id),
            spec.fact,
        )
        for spec in specs
    ]
