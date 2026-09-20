"""Adapter from LifeOps context.v1 into transport-level world tensors.

The adapter preserves source authority and derived/candidate status. LifeOps
attention projections are not promoted into Personal Physics admitted facts.
They become candidate tensors whose provenance records the underlying source
reference and derivation method.
"""

from __future__ import annotations

import json
from typing import Any

from .world_tensor import (
    CoordinateProvenance,
    TensorWorld,
)


def tensorize_life_context(
    context: dict[str, Any],
) -> TensorWorld:
    if context.get("schema_version") != "lifeops.context.v1":
        raise ValueError(
            "expected LifeOps schema_version lifeops.context.v1"
        )

    sections = context.get("sections", {})
    attention = list(sections.get("attention", []))
    people = list(sections.get("people", []))
    source_health = context.get("source_health", {})
    providers = list(
        source_health.get("providers", {}).get(
            "providers",
            [],
        )
    )

    person_symbols = {
        item["item_id"]
        for item in people
        if item.get("item_id")
    }
    for item in attention:
        for participant in (
            item.get("details", {}).get("participants", [])
        ):
            person_symbols.add(
                f"participant:{participant}"
            )

    item_symbols = {
        item["item_id"]
        for item in attention
        if item.get("item_id")
    }
    source_symbols = {
        item.get("source", "unknown")
        for item in attention
    }
    source_symbols.update(
        provider.get("provider", "unknown")
        for provider in providers
    )
    class_symbols = {
        item.get("attention_class", "unknown")
        for item in attention
    }
    category_symbols = {
        item.get("category", "unknown")
        for item in attention
    }
    time_symbols = {
        item.get("details", {}).get("last_message_at")
        for item in attention
        if item.get("details", {}).get("last_message_at")
    }
    provider_symbols = {
        provider.get("provider")
        for provider in providers
        if provider.get("provider")
    }

    world = TensorWorld()
    axes = {
        "Person": person_symbols,
        "AttentionItem": item_symbols,
        "Source": source_symbols,
        "AttentionClass": class_symbols,
        "Category": category_symbols,
        "TimeBucket": time_symbols,
        "Provider": provider_symbols,
    }
    for name, symbols in axes.items():
        world.add_axis(name, name, symbols)

    definitions = {
        "attention_source": (
            ("AttentionItem", "Source"),
            "boolean",
        ),
        "attention_class": (
            ("AttentionItem", "AttentionClass"),
            "boolean",
        ),
        "attention_category": (
            ("AttentionItem", "Category"),
            "boolean",
        ),
        "attention_participant": (
            ("Person", "AttentionItem"),
            "boolean",
        ),
        "attention_last_message": (
            ("AttentionItem", "TimeBucket"),
            "boolean",
        ),
        "candidate_needs_reply": (
            ("AttentionItem",),
            "boolean",
        ),
        "candidate_rank": (
            ("AttentionItem",),
            "real",
        ),
        "transport_projection_derived": (
            ("AttentionItem",),
            "boolean",
        ),
        "provider_readable": (
            ("Provider",),
            "boolean",
        ),
        "provider_syncable": (
            ("Provider",),
            "boolean",
        ),
        "provider_writable": (
            ("Provider",),
            "boolean",
        ),
    }
    for name, (axis_names, value_kind) in definitions.items():
        world.add_tensor(
            name,
            axis_names,
            value_kind=value_kind,
        )

    for item in attention:
        item_id = item.get("item_id")
        if not item_id:
            continue
        source = item.get("source", "unknown")
        attention_class = item.get(
            "attention_class",
            "unknown",
        )
        category = item.get("category", "unknown")
        attribution = item.get("attribution", {})
        details = item.get("details", {})
        provenance = CoordinateProvenance(
            evidence_refs=(
                _stable_ref(item.get("source_ref")),
            ),
            source_refs=(
                attribution.get("authority", source),
            ),
            confidence=None,
            metadata={
                "epistemic_status": "candidate"
                if attribution.get("derived")
                else "source_observation",
                "derived": bool(attribution.get("derived")),
                "method": attribution.get("method"),
                "state": item.get("state"),
                "reason": item.get("reason"),
                "read_only": item.get("read_only", True),
            },
        )
        world.tensors["attention_source"].set(
            (item_id, source),
            1.0,
            provenance=provenance,
        )
        world.tensors["attention_class"].set(
            (item_id, attention_class),
            1.0,
            provenance=provenance,
        )
        world.tensors["attention_category"].set(
            (item_id, category),
            1.0,
            provenance=provenance,
        )

        for participant in details.get("participants", []):
            world.tensors["attention_participant"].set(
                (f"participant:{participant}", item_id),
                1.0,
                provenance=provenance,
            )

        last_message_at = details.get("last_message_at")
        if last_message_at:
            world.tensors["attention_last_message"].set(
                (item_id, last_message_at),
                1.0,
                provenance=provenance,
            )

        if bool(details.get("needs_reply")):
            world.tensors["candidate_needs_reply"].set(
                (item_id,),
                1.0,
                provenance=provenance,
            )

        rank = details.get("rank")
        if rank is not None:
            world.tensors["candidate_rank"].set(
                (item_id,),
                float(rank),
                provenance=provenance,
            )

        if bool(attribution.get("derived")):
            world.tensors[
                "transport_projection_derived"
            ].set(
                (item_id,),
                1.0,
                provenance=provenance,
            )

    for provider in providers:
        provider_name = provider.get("provider")
        if not provider_name:
            continue
        provider_provenance = CoordinateProvenance(
            source_refs=("lifeops.source_health",),
            metadata={
                "blockers": list(provider.get("blockers", [])),
                "notes": provider.get("notes"),
            },
        )
        for tensor_name, field_name in (
            ("provider_readable", "readable"),
            ("provider_syncable", "syncable"),
            ("provider_writable", "writable"),
        ):
            world.tensors[tensor_name].set(
                (provider_name,),
                1.0 if provider.get(field_name) else 0.0,
                provenance=provider_provenance,
            )

    return world


def _stable_ref(value: Any) -> str:
    if value is None:
        return "source_ref:none"
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
