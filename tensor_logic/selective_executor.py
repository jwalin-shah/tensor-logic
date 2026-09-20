"""Selective execution of compiled query plans.

The planner determines what is required. This executor enforces that decision by
loading only named primitive tensors and executing only the ordered derived
views in the compiled query.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

from .query_execution import QueryExecutionAssessment
from .query_runtime import CompiledQuery, QueryRegistry


OperatorFn = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]
PrimitiveLoader = Callable[[str], Any]


@dataclass(frozen=True)
class RegisteredOperator:
    operator_id: str
    operator_version: str
    fn: OperatorFn


@dataclass(frozen=True)
class SelectiveExecutionTrace:
    query_digest: str
    primitive_reads: tuple[str, ...]
    executed_views: tuple[str, ...]
    operator_keys: tuple[str, ...]

    @property
    def digest(self) -> str:
        raw = json.dumps(
            {
                "query_digest": self.query_digest,
                "primitive_reads": list(self.primitive_reads),
                "executed_views": list(self.executed_views),
                "operator_keys": list(self.operator_keys),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SelectiveExecutionResult:
    output: Any
    trace: SelectiveExecutionTrace


class QueryOperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[
            tuple[str, str],
            RegisteredOperator,
        ] = {}

    def register(
        self,
        operator_id: str,
        operator_version: str,
        fn: OperatorFn,
    ) -> None:
        if not operator_id or not operator_version:
            raise ValueError("operator id/version are required")
        key = (operator_id, operator_version)
        existing = self._operators.get(key)
        item = RegisteredOperator(
            operator_id=operator_id,
            operator_version=operator_version,
            fn=fn,
        )
        if existing is not None and existing.fn is not fn:
            raise ValueError(
                f"operator {operator_id}@{operator_version} already registered"
            )
        self._operators[key] = item

    def get(
        self,
        operator_id: str,
        operator_version: str,
    ) -> RegisteredOperator:
        key = (operator_id, operator_version)
        try:
            return self._operators[key]
        except KeyError as exc:
            raise ValueError(
                f"operator {operator_id}@{operator_version} is not registered"
            ) from exc


def execute_compiled_query(
    registry: QueryRegistry,
    query: CompiledQuery,
    primitive_loader: PrimitiveLoader,
    operators: QueryOperatorRegistry,
    *,
    assessment: QueryExecutionAssessment | None = None,
) -> SelectiveExecutionResult:
    if assessment is not None and not assessment.runnable:
        raise ValueError(
            "query execution blocked by preflight: "
            + ";".join(assessment.reasons)
        )

    values: dict[str, Any] = {}
    primitive_reads: list[str] = []

    for name in query.primitive_tensors:
        values[name] = primitive_loader(name)
        primitive_reads.append(name)

    executed_views: list[str] = []
    operator_keys: list[str] = []

    for view_name in query.view_order:
        view = registry.views.get(view_name)
        if view is None:
            raise ValueError(
                f"compiled query references unknown view {view_name!r}"
            )
        missing = [
            name
            for name in view.inputs
            if name not in values
        ]
        if missing:
            raise ValueError(
                f"view {view_name!r} missing inputs: {missing}"
            )

        operator = operators.get(
            view.operator_id,
            view.operator_version,
        )
        inputs = tuple(values[name] for name in view.inputs)
        values[view_name] = operator.fn(inputs, view.parameters)
        executed_views.append(view_name)
        operator_keys.append(
            f"{view.operator_id}@{view.operator_version}"
        )

    if query.target not in values:
        # Primitive-only query.
        if query.target in query.primitive_tensors:
            output = values[query.target]
        else:
            raise ValueError("query target was not produced")
    else:
        output = values[query.target]

    return SelectiveExecutionResult(
        output=output,
        trace=SelectiveExecutionTrace(
            query_digest=query.digest,
            primitive_reads=tuple(primitive_reads),
            executed_views=tuple(executed_views),
            operator_keys=tuple(operator_keys),
        ),
    )
