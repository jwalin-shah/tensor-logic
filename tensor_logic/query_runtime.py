"""Selective query compilation and transitive dependency invalidation.

The trusted planner is deliberately typed and deterministic. Natural-language
systems may select a target query outside this module, but once a target is
chosen the dependency closure, source gates, and recomputation order require no
LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SourceRequirement:
    source: str
    require_readable: bool = True
    max_age_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source is required")
        if self.max_age_seconds is not None and self.max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative")


@dataclass(frozen=True)
class SourceState:
    source: str
    readable: bool
    age_seconds: float | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source is required")
        if self.age_seconds is not None and self.age_seconds < 0:
            raise ValueError("age_seconds cannot be negative")


@dataclass(frozen=True)
class PrimitiveTensorSpec:
    name: str
    source_requirements: tuple[SourceRequirement, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("primitive tensor name is required")


@dataclass(frozen=True)
class DerivedViewSpec:
    output: str
    inputs: tuple[str, ...]
    operator_id: str
    operator_version: str
    source_requirements: tuple[SourceRequirement, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output:
            raise ValueError("output is required")
        if not self.inputs:
            raise ValueError("derived view requires at least one input")
        if not self.operator_id or not self.operator_version:
            raise ValueError("operator id/version are required")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "output": self.output,
                "inputs": list(self.inputs),
                "operator_id": self.operator_id,
                "operator_version": self.operator_version,
                "source_requirements": [
                    _source_requirement_payload(req)
                    for req in self.source_requirements
                ],
                "parameters": dict(self.parameters),
            }
        )


@dataclass(frozen=True)
class CompiledQuery:
    target: str
    primitive_tensors: tuple[str, ...]
    view_order: tuple[str, ...]
    required_sources: tuple[SourceRequirement, ...]
    definition_digests: tuple[str, ...]

    @property
    def digest(self) -> str:
        return _digest(
            {
                "target": self.target,
                "primitive_tensors": list(self.primitive_tensors),
                "view_order": list(self.view_order),
                "required_sources": [
                    _source_requirement_payload(req)
                    for req in self.required_sources
                ],
                "definition_digests": list(self.definition_digests),
            }
        )


@dataclass(frozen=True)
class QueryReadiness:
    ready: bool
    blocked_sources: tuple[str, ...]
    stale_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]


@dataclass(frozen=True)
class InvalidationResult:
    changed_inputs: tuple[str, ...]
    dirty_views: tuple[str, ...]
    recompute_order: tuple[str, ...]


class QueryRegistry:
    def __init__(self) -> None:
        self._primitives: dict[str, PrimitiveTensorSpec] = {}
        self._views: dict[str, DerivedViewSpec] = {}

    def add_primitive(self, spec: PrimitiveTensorSpec) -> None:
        if spec.name in self._views:
            raise ValueError(
                f"{spec.name!r} already exists as a derived view"
            )
        existing = self._primitives.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(
                f"primitive {spec.name!r} already has another definition"
            )
        self._primitives[spec.name] = spec

    def add_view(self, spec: DerivedViewSpec) -> None:
        if spec.output in self._primitives:
            raise ValueError(
                f"{spec.output!r} already exists as a primitive tensor"
            )
        existing = self._views.get(spec.output)
        if existing is not None and existing != spec:
            raise ValueError(
                f"view {spec.output!r} already has another definition"
            )
        self._views[spec.output] = spec
        self._validate_acyclic()

    @property
    def primitives(self) -> Mapping[str, PrimitiveTensorSpec]:
        return dict(self._primitives)

    @property
    def views(self) -> Mapping[str, DerivedViewSpec]:
        return dict(self._views)

    def compile(self, target: str) -> CompiledQuery:
        if target not in self._primitives and target not in self._views:
            raise ValueError(f"unknown query target: {target}")

        primitive_names: set[str] = set()
        required_sources: dict[str, SourceRequirement] = {}
        view_order: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def add_requirement(req: SourceRequirement) -> None:
            previous = required_sources.get(req.source)
            if previous is None:
                required_sources[req.source] = req
                return
            max_age = _stricter_age(
                previous.max_age_seconds,
                req.max_age_seconds,
            )
            required_sources[req.source] = SourceRequirement(
                source=req.source,
                require_readable=(
                    previous.require_readable or req.require_readable
                ),
                max_age_seconds=max_age,
            )

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(
                    f"query dependency cycle detected at {name!r}"
                )

            primitive = self._primitives.get(name)
            if primitive is not None:
                primitive_names.add(name)
                for req in primitive.source_requirements:
                    add_requirement(req)
                visited.add(name)
                return

            view = self._views.get(name)
            if view is None:
                raise ValueError(
                    f"view dependency {name!r} has no primitive/view definition"
                )

            visiting.add(name)
            for input_name in view.inputs:
                visit(input_name)
            visiting.remove(name)

            for req in view.source_requirements:
                add_requirement(req)
            view_order.append(name)
            visited.add(name)

        visit(target)

        ordered_requirements = tuple(
            required_sources[name]
            for name in sorted(required_sources)
        )
        return CompiledQuery(
            target=target,
            primitive_tensors=tuple(sorted(primitive_names)),
            view_order=tuple(view_order),
            required_sources=ordered_requirements,
            definition_digests=tuple(
                self._views[name].digest
                for name in view_order
            ),
        )

    def readiness(
        self,
        query: CompiledQuery,
        states: Mapping[str, SourceState],
    ) -> QueryReadiness:
        blocked: list[str] = []
        stale: list[str] = []
        missing: list[str] = []

        for req in query.required_sources:
            state = states.get(req.source)
            if state is None:
                missing.append(req.source)
                continue
            if req.require_readable and not state.readable:
                blocked.append(req.source)
                continue
            if (
                req.max_age_seconds is not None
                and (
                    state.age_seconds is None
                    or state.age_seconds > req.max_age_seconds
                )
            ):
                stale.append(req.source)

        return QueryReadiness(
            ready=not (blocked or stale or missing),
            blocked_sources=tuple(sorted(blocked)),
            stale_sources=tuple(sorted(stale)),
            missing_sources=tuple(sorted(missing)),
        )

    def invalidate(
        self,
        changed_inputs: Iterable[str],
    ) -> InvalidationResult:
        changed = tuple(sorted(set(changed_inputs)))
        reverse: dict[str, set[str]] = {}
        for view_name, view in self._views.items():
            for input_name in view.inputs:
                reverse.setdefault(input_name, set()).add(view_name)

        dirty: set[str] = set()
        frontier = list(changed)
        while frontier:
            item = frontier.pop()
            for dependent in sorted(reverse.get(item, ())):
                if dependent in dirty:
                    continue
                dirty.add(dependent)
                frontier.append(dependent)

        if not dirty:
            return InvalidationResult(
                changed_inputs=changed,
                dirty_views=(),
                recompute_order=(),
            )

        # Compile each dirty target and preserve only dirty derived nodes. Sorting
        # by dependency depth then name creates a deterministic valid order.
        depth_cache: dict[str, int] = {}

        def depth(name: str) -> int:
            if name in depth_cache:
                return depth_cache[name]
            view = self._views[name]
            child_depths = [
                depth(inp)
                for inp in view.inputs
                if inp in dirty
            ]
            value = 1 + max(child_depths, default=0)
            depth_cache[name] = value
            return value

        order = tuple(sorted(dirty, key=lambda name: (depth(name), name)))
        return InvalidationResult(
            changed_inputs=changed,
            dirty_views=tuple(sorted(dirty)),
            recompute_order=order,
        )

    def _validate_acyclic(self) -> None:
        for output in self._views:
            visiting: set[str] = set()
            visited: set[str] = set()

            def walk(name: str) -> None:
                if name in visited:
                    return
                if name in visiting:
                    raise ValueError(
                        f"query dependency cycle detected at {name!r}"
                    )
                view = self._views.get(name)
                if view is None:
                    visited.add(name)
                    return
                visiting.add(name)
                for input_name in view.inputs:
                    if input_name in self._views:
                        walk(input_name)
                visiting.remove(name)
                visited.add(name)

            walk(output)


def _stricter_age(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _source_requirement_payload(
    req: SourceRequirement,
) -> dict[str, Any]:
    return {
        "source": req.source,
        "require_readable": req.require_readable,
        "max_age_seconds": req.max_age_seconds,
    }


def _digest(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
