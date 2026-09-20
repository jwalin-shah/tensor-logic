"""Generic loss-preserving tensor projection for arbitrary context records.

This is the coverage layer underneath semantic adapters. It ensures that fields
we have not modeled semantically are still available as typed tensor state with
provenance rather than disappearing into an opaque JSON blob.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from .world_tensor import (
    CoordinateProvenance,
    TensorWorld,
)


@dataclass(frozen=True)
class GenericRecord:
    section: str
    record_id: str
    payload: Any
    provenance: CoordinateProvenance


def tensorize_generic_context(
    context: dict[str, Any],
) -> TensorWorld:
    records = list(_records_from_context(context))

    declared_sections = set(
        context.get("sections", {}).keys()
        if isinstance(context.get("sections", {}), dict)
        else ()
    )
    sections = sorted(
        {record.section for record in records}.union(declared_sections)
    )
    record_ids = sorted({record.record_id for record in records})

    flattened: dict[
        str,
        list[tuple[str, str, Any]],
    ] = {}
    properties: set[str] = set()
    symbols: set[str] = set()
    scalar_types: set[str] = set()
    container_kinds = {"dict", "list"}

    for record in records:
        entries = list(_flatten(record.payload))
        flattened[record.record_id] = entries
        for path, kind, value in entries:
            properties.add(path)
            if kind == "string":
                symbols.add(str(value))
                scalar_types.add("string")
            elif kind == "bool":
                scalar_types.add("bool")
            elif kind == "int":
                scalar_types.add("int")
            elif kind == "float":
                scalar_types.add("float")
            elif kind == "null":
                scalar_types.add("null")

    world = TensorWorld()
    world.add_axis("Section", "Section", sections)
    world.add_axis("Record", "Record", record_ids)
    world.add_axis("Property", "Property", properties)
    world.add_axis("Symbol", "Symbol", symbols)
    world.add_axis(
        "ScalarType",
        "ScalarType",
        scalar_types,
    )
    world.add_axis(
        "ContainerKind",
        "ContainerKind",
        container_kinds,
    )

    world.add_tensor(
        "section_present",
        ("Section",),
    )
    world.add_tensor(
        "section_record",
        ("Section", "Record"),
    )
    world.add_tensor(
        "scalar_type",
        ("Record", "Property", "ScalarType"),
    )
    world.add_tensor(
        "string_value",
        ("Record", "Property", "Symbol"),
    )
    world.add_tensor(
        "numeric_value",
        ("Record", "Property"),
        value_kind="real",
    )
    world.add_tensor(
        "bool_value",
        ("Record", "Property"),
    )
    world.add_tensor(
        "is_null",
        ("Record", "Property"),
    )
    world.add_tensor(
        "container_kind",
        ("Record", "Property", "ContainerKind"),
    )
    world.add_tensor(
        "container_size",
        ("Record", "Property"),
        value_kind="real",
    )

    by_id = {record.record_id: record for record in records}
    for section_name in sections:
        world.tensors["section_present"].set(
            (section_name,),
            1.0,
            provenance=CoordinateProvenance(
                source_refs=("lifeops.context.v1",),
                metadata={"section": section_name},
            ),
        )

    for record in records:
        world.tensors["section_record"].set(
            (record.section, record.record_id),
            1.0,
            provenance=record.provenance,
        )

    for record_id, entries in flattened.items():
        provenance = by_id[record_id].provenance
        for path, kind, value in entries:
            if kind in {"dict", "list"}:
                world.tensors["container_kind"].set(
                    (record_id, path, kind),
                    1.0,
                    provenance=provenance,
                )
                world.tensors["container_size"].set(
                    (record_id, path),
                    float(value),
                    provenance=provenance,
                )
                continue

            if kind == "null":
                world.tensors["scalar_type"].set(
                    (record_id, path, "null"),
                    1.0,
                    provenance=provenance,
                )
                world.tensors["is_null"].set(
                    (record_id, path),
                    1.0,
                    provenance=provenance,
                )
                continue

            world.tensors["scalar_type"].set(
                (record_id, path, kind),
                1.0,
                provenance=provenance,
            )
            if kind == "string":
                world.tensors["string_value"].set(
                    (record_id, path, str(value)),
                    1.0,
                    provenance=provenance,
                )
            elif kind == "bool":
                world.tensors["bool_value"].set(
                    (record_id, path),
                    1.0 if value else 0.0,
                    provenance=provenance,
                )
            elif kind in {"int", "float"}:
                world.tensors["numeric_value"].set(
                    (record_id, path),
                    float(value),
                    provenance=provenance,
                )

    return world


def reconstruct_generic_context(
    world: TensorWorld,
) -> dict[str, Any]:
    """Reconstruct a LifeOps-style context snapshot from generic tensors."""
    result = dict(
        reconstruct_generic_record(
            world,
            "context:root",
        )
    )

    section_records: dict[str, list[str]] = {}
    for section, record_id in world.tensors[
        "section_record"
    ].coordinates():
        section_records.setdefault(section, []).append(record_id)

    sections: dict[str, list[Any]] = {}
    for section in world.axes["Section"].symbols:
        if section.startswith("__"):
            continue
        records = sorted(section_records.get(section, []))
        sections[section] = [
            reconstruct_generic_record(world, record_id)
            for record_id in records
        ]
    result["sections"] = sections

    record_axis = world.axes["Record"].index
    if "source_health:root" in record_axis:
        result["source_health"] = reconstruct_generic_record(
            world,
            "source_health:root",
        )
    if "provenance:root" in record_axis:
        result["provenance"] = reconstruct_generic_record(
            world,
            "provenance:root",
        )
    return result


def reconstruct_generic_record(
    world: TensorWorld,
    record_id: str,
) -> Any:
    """Reconstruct one record payload from the generic tensor projection."""
    if record_id not in world.axes["Record"].index:
        raise ValueError(f"unknown record: {record_id}")

    entries: dict[str, tuple[str, Any]] = {}
    for coordinate in world.tensors["container_kind"].coordinates():
        rec, path, kind = coordinate
        if rec != record_id:
            continue
        size = int(
            world.tensors["container_size"].get((rec, path))
        )
        entries[path] = (kind, size)

    for coordinate in world.tensors["scalar_type"].coordinates():
        rec, path, scalar_type = coordinate
        if rec != record_id:
            continue
        if scalar_type == "null":
            entries[path] = ("null", None)
        elif scalar_type == "string":
            symbol = _string_value_for(world, rec, path)
            entries[path] = ("string", symbol)
        elif scalar_type == "bool":
            entries[path] = (
                "bool",
                bool(world.tensors["bool_value"].get((rec, path))),
            )
        elif scalar_type == "int":
            entries[path] = (
                "int",
                int(world.tensors["numeric_value"].get((rec, path))),
            )
        elif scalar_type == "float":
            entries[path] = (
                "float",
                float(world.tensors["numeric_value"].get((rec, path))),
            )

    return _unflatten(entries)


def _records_from_context(
    context: dict[str, Any],
) -> Iterable[GenericRecord]:
    root_payload = {
        key: value
        for key, value in context.items()
        if key not in {"sections", "source_health", "provenance"}
    }
    yield GenericRecord(
        section="__context__",
        record_id="context:root",
        payload=root_payload,
        provenance=CoordinateProvenance(
            source_refs=("lifeops.context.v1",),
            metadata={"scope": "root"},
        ),
    )

    sections = context.get("sections", {})
    for section_name, values in sorted(sections.items()):
        if not isinstance(values, list):
            values = [values]
        for index, value in enumerate(values):
            record_id = _record_id(section_name, index, value)
            yield GenericRecord(
                section=section_name,
                record_id=record_id,
                payload=value,
                provenance=_item_provenance(
                    value,
                    section_name,
                ),
            )

    for key in ("source_health", "provenance"):
        if key in context:
            yield GenericRecord(
                section=f"__{key}__",
                record_id=f"{key}:root",
                payload=context[key],
                provenance=CoordinateProvenance(
                    source_refs=(f"lifeops.{key}",),
                    metadata={"scope": key},
                ),
            )


def _record_id(section: str, index: int, value: Any) -> str:
    if isinstance(value, dict):
        for key in (
            "item_id",
            "event_id",
            "person_id",
            "contact_id",
            "project_id",
            "goal_id",
            "decision_id",
            "commitment_id",
            "document_id",
            "note_id",
            "id",
        ):
            if value.get(key) is not None:
                return f"{section}:{value[key]}"
    return f"{section}:{index}"


def _item_provenance(
    value: Any,
    section: str,
) -> CoordinateProvenance:
    if not isinstance(value, dict):
        return CoordinateProvenance(
            source_refs=(f"lifeops.section.{section}",),
        )
    attribution = value.get("attribution", {})
    source = value.get("source")
    source_ref = value.get("source_ref")
    refs = []
    if source_ref is not None:
        refs.append(_stable_json(source_ref))
    return CoordinateProvenance(
        evidence_refs=tuple(refs),
        source_refs=tuple(
            item
            for item in (
                attribution.get("authority"),
                source,
            )
            if item
        ),
        confidence=value.get("confidence"),
        metadata={
            "derived": bool(attribution.get("derived")),
            "method": attribution.get("method"),
            "section": section,
        },
    )


def _flatten(
    value: Any,
    path: str = "$",
) -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        yield (path, "dict", len(value))
        for key in sorted(value):
            child = f"{path}.{_escape_key(str(key))}"
            yield from _flatten(value[key], child)
        return
    if isinstance(value, list):
        yield (path, "list", len(value))
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            yield from _flatten(item, child)
        return
    if value is None:
        yield (path, "null", None)
    elif isinstance(value, bool):
        yield (path, "bool", value)
    elif isinstance(value, int):
        yield (path, "int", value)
    elif isinstance(value, float):
        yield (path, "float", value)
    else:
        yield (path, "string", str(value))


def _unflatten(entries: dict[str, tuple[str, Any]]) -> Any:
    if "$" not in entries:
        raise ValueError("generic tensor record has no root container")
    return _build_value("$", entries)


def _build_value(
    path: str,
    entries: dict[str, tuple[str, Any]],
) -> Any:
    kind, value = entries[path]
    if kind == "dict":
        result = {}
        prefix = path + "."
        child_keys = set()
        for candidate in entries:
            if not candidate.startswith(prefix):
                continue
            remainder = candidate[len(prefix):]
            first = _first_path_segment(remainder)
            if first is not None:
                child_keys.add(first)
        for escaped in sorted(child_keys):
            child_path = prefix + escaped
            result[_unescape_key(escaped)] = _build_value(
                child_path,
                entries,
            )
        return result
    if kind == "list":
        return [
            _build_value(f"{path}[{index}]", entries)
            for index in range(value)
        ]
    return value


def _first_path_segment(remainder: str) -> str | None:
    if not remainder or remainder.startswith("["):
        return None
    depth = 0
    for index, char in enumerate(remainder):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "." and depth == 0:
            return remainder[:index]
    bracket = remainder.find("[")
    if bracket >= 0:
        return remainder[:bracket]
    return remainder


def _string_value_for(
    world: TensorWorld,
    record_id: str,
    path: str,
) -> str:
    matches = [
        symbol
        for rec, prop, symbol in world.tensors[
            "string_value"
        ].coordinates()
        if rec == record_id and prop == path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one string value for {record_id} {path}"
        )
    return matches[0]


def _escape_key(key: str) -> str:
    return (
        key.replace("~", "~0")
        .replace(".", "~1")
        .replace("[", "~2")
        .replace("]", "~3")
    )


def _unescape_key(key: str) -> str:
    return (
        key.replace("~3", "]")
        .replace("~2", "[")
        .replace("~1", ".")
        .replace("~0", "~")
    )


def _stable_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
