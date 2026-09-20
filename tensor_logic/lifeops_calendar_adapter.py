"""Conservative normalization of LifeOps calendar records for Personal Physics.

Input shape matches the read-only LifeOps calendar_events projection. The
adapter only admits timed, located events with explicit IDs. It does not infer
locations from titles/descriptions and does not call the route service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CalendarPrimitiveBundle:
    attends: dict[str, tuple[str, ...]]
    located_at: dict[str, str]
    starts_at: dict[str, float]
    ends_at: dict[str, float]
    source_refs: dict[str, dict[str, Any]]
    excluded_event_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    revision: str


def calendar_events_to_primitives(
    events: Sequence[Mapping[str, Any]],
    *,
    person_id: str = "me",
) -> CalendarPrimitiveBundle:
    if not person_id:
        raise ValueError("person_id is required")

    admitted: list[tuple[float, str]] = []
    located_at: dict[str, str] = {}
    starts_at: dict[str, float] = {}
    ends_at: dict[str, float] = {}
    source_refs: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    limitations: list[str] = []

    for index, event in enumerate(events):
        event_id = str(event.get("event_id") or "")
        if not event_id:
            limitations.append(f"event[{index}]:missing_event_id")
            continue

        if bool(event.get("all_day")):
            excluded.append(event_id)
            limitations.append(f"{event_id}:all_day_excluded")
            continue

        location = str(event.get("location") or "").strip()
        start_raw = event.get("start")
        end_raw = event.get("end")

        if not location:
            excluded.append(event_id)
            limitations.append(f"{event_id}:missing_location")
            continue
        if not isinstance(start_raw, str) or not isinstance(end_raw, str):
            excluded.append(event_id)
            limitations.append(f"{event_id}:missing_time")
            continue

        try:
            start = _parse_iso(start_raw)
            end = _parse_iso(end_raw)
        except ValueError:
            excluded.append(event_id)
            limitations.append(f"{event_id}:invalid_time")
            continue

        start_minutes = start.timestamp() / 60.0
        end_minutes = end.timestamp() / 60.0
        if end_minutes < start_minutes:
            excluded.append(event_id)
            limitations.append(f"{event_id}:end_before_start")
            continue

        admitted.append((start_minutes, event_id))
        located_at[event_id] = location
        starts_at[event_id] = start_minutes
        ends_at[event_id] = end_minutes
        source_refs[event_id] = {
            "kind": "calendar_event",
            "source": "google_calendar",
            "event_id": event_id,
            "calendar_id": str(event.get("calendar_id") or ""),
            "account": str(event.get("account") or ""),
        }

    admitted.sort()
    attends = {
        person_id: tuple(event_id for _, event_id in admitted)
    }

    payload = {
        "person_id": person_id,
        "events": [
            {
                "event_id": event_id,
                "location": located_at[event_id],
                "start": starts_at[event_id],
                "end": ends_at[event_id],
                "source_ref": source_refs[event_id],
            }
            for _, event_id in admitted
        ],
        "excluded": sorted(excluded),
        "limitations": sorted(limitations),
    }
    revision = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return CalendarPrimitiveBundle(
        attends=attends,
        located_at=located_at,
        starts_at=starts_at,
        ends_at=ends_at,
        source_refs=source_refs,
        excluded_event_ids=tuple(sorted(excluded)),
        limitations=tuple(sorted(limitations)),
        revision=revision,
    )


def _parse_iso(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("calendar timestamps must be timezone-aware")
    return parsed
