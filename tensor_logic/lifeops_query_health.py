"""Translate LifeOps source-health into deterministic query planner gates.

Provider capability and captured evidence freshness are distinct:
- provider:<name> describes configured/readable API capability;
- capture:<key> describes one concrete capture/account probe;
- capture-source:<source_id> aggregates the freshest readable capture for a
  source family.

Queries that need every account should depend on exact capture:<key> entries
rather than the aggregate capture-source:<source_id> convenience state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .query_runtime import SourceState


def lifeops_source_states(
    source_health: Mapping[str, Any],
) -> dict[str, SourceState]:
    states: dict[str, SourceState] = {}

    providers_block = source_health.get("providers", {})
    provider_checked_at = _parse_time(
        providers_block.get("checked_at")
    )
    for provider in providers_block.get("providers", ()):
        name = provider.get("provider")
        if not name:
            continue
        states[f"provider:{name}"] = SourceState(
            source=f"provider:{name}",
            readable=bool(provider.get("readable")),
            age_seconds=0.0 if provider_checked_at is not None else None,
            revision=_provider_revision(provider),
        )

    capture_block = source_health.get("capture", {})
    source_groups: dict[str, list[SourceState]] = {}
    for capture in capture_block.get("sources", ()):
        key = capture.get("key")
        source_id = capture.get("source_id")
        if not key or not source_id:
            continue

        checked_at = _parse_time(capture.get("checked_at"))
        success_at = _parse_time(capture.get("last_success_at"))
        age_seconds = _age_seconds(checked_at, success_at)

        state = SourceState(
            source=f"capture:{key}",
            readable=bool(capture.get("readable")),
            age_seconds=age_seconds,
            revision=_capture_revision(capture),
        )
        states[state.source] = state
        source_groups.setdefault(source_id, []).append(state)

    for source_id, members in source_groups.items():
        readable = [member for member in members if member.readable]
        if readable:
            known_ages = [
                member.age_seconds
                for member in readable
                if member.age_seconds is not None
            ]
            freshest_age = min(known_ages) if known_ages else None
            revision = "|".join(
                sorted(
                    member.revision
                    for member in readable
                    if member.revision is not None
                )
            ) or None
            aggregate = SourceState(
                source=f"capture-source:{source_id}",
                readable=True,
                age_seconds=freshest_age,
                revision=revision,
            )
        else:
            aggregate = SourceState(
                source=f"capture-source:{source_id}",
                readable=False,
                age_seconds=None,
                revision=None,
            )
        states[aggregate.source] = aggregate

    return states


def _age_seconds(
    checked_at: datetime | None,
    success_at: datetime | None,
) -> float | None:
    if checked_at is None or success_at is None:
        return None
    return max((checked_at - success_at).total_seconds(), 0.0)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _provider_revision(provider: Mapping[str, Any]) -> str:
    fields = (
        str(provider.get("configured", False)),
        str(provider.get("authenticated", False)),
        str(provider.get("readable", False)),
        str(provider.get("syncable", False)),
        str(provider.get("writable", False)),
        ",".join(sorted(str(x) for x in provider.get("blockers", ()))),
    )
    return ":".join(fields)


def _capture_revision(capture: Mapping[str, Any]) -> str:
    return ":".join(
        (
            str(capture.get("status", "")),
            str(capture.get("last_success_at", "")),
            str(capture.get("newest_seen_id", "")),
            str(capture.get("item_count", 0)),
        )
    )
