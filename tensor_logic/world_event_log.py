"""Append-only evidence/event history for hybrid world models."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable


@dataclass(frozen=True)
class WorldEvent:
    event_id: str
    event_type: str
    occurred_at: str
    entity_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "occurred_at": self.occurred_at,
                "entity_refs": list(self.entity_refs),
                "source_refs": list(self.source_refs),
                "evidence_refs": list(self.evidence_refs),
                "payload": self.payload,
            }
        )


@dataclass(frozen=True)
class SequencedEvent:
    sequence: int
    event: WorldEvent


@dataclass(frozen=True)
class EventCheckpoint:
    sequence: int
    log_digest: str


class AppendOnlyWorldLog:
    def __init__(self) -> None:
        self._events: list[SequencedEvent] = []
        self._event_ids: set[str] = set()

    def append(self, event: WorldEvent) -> SequencedEvent:
        if not event.event_id:
            raise ValueError("event_id is required")
        if event.event_id in self._event_ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        item = SequencedEvent(
            sequence=len(self._events) + 1,
            event=event,
        )
        self._events.append(item)
        self._event_ids.add(event.event_id)
        return item

    def extend(self, events: Iterable[WorldEvent]) -> tuple[SequencedEvent, ...]:
        return tuple(self.append(event) for event in events)

    @property
    def events(self) -> tuple[SequencedEvent, ...]:
        return tuple(self._events)

    @property
    def sequence(self) -> int:
        return len(self._events)

    @property
    def digest(self) -> str:
        return _digest(
            [
                {
                    "sequence": item.sequence,
                    "event_digest": item.event.digest,
                }
                for item in self._events
            ]
        )

    def checkpoint(self) -> EventCheckpoint:
        return EventCheckpoint(
            sequence=self.sequence,
            log_digest=self.digest,
        )

    def since(
        self,
        checkpoint: EventCheckpoint | int,
    ) -> tuple[SequencedEvent, ...]:
        sequence = (
            checkpoint.sequence
            if isinstance(checkpoint, EventCheckpoint)
            else int(checkpoint)
        )
        if sequence < 0 or sequence > self.sequence:
            raise ValueError("checkpoint sequence outside log")
        return tuple(
            item
            for item in self._events
            if item.sequence > sequence
        )


def _digest(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
