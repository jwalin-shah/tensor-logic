from tensor_logic.world_event_log import (
    AppendOnlyWorldLog,
    WorldEvent,
)


def _event(event_id: str, value: int):
    return WorldEvent(
        event_id=event_id,
        event_type="observation",
        occurred_at=f"2026-09-20T16:{value:02d}:00Z",
        entity_refs=(f"entity:{value}",),
        source_refs=("source:test",),
        evidence_refs=(f"evidence:{value}",),
        payload={"value": value},
    )


def test_event_log_is_append_only_and_sequenced():
    log = AppendOnlyWorldLog()
    first = log.append(_event("e1", 1))
    second = log.append(_event("e2", 2))

    assert first.sequence == 1
    assert second.sequence == 2
    assert log.sequence == 2
    assert [item.event.event_id for item in log.events] == ["e1", "e2"]


def test_checkpoint_and_delta_read():
    log = AppendOnlyWorldLog()
    log.append(_event("e1", 1))
    checkpoint = log.checkpoint()
    log.append(_event("e2", 2))
    log.append(_event("e3", 3))

    delta = log.since(checkpoint)

    assert [item.event.event_id for item in delta] == ["e2", "e3"]
    assert checkpoint.sequence == 1
    assert checkpoint.log_digest != log.digest


def test_log_digest_is_deterministic_for_same_history():
    left = AppendOnlyWorldLog()
    right = AppendOnlyWorldLog()
    for event in (_event("e1", 1), _event("e2", 2)):
        left.append(event)
        right.append(event)

    assert left.digest == right.digest
    assert left.events[0].event.digest == right.events[0].event.digest


def test_duplicate_event_id_fails_closed():
    log = AppendOnlyWorldLog()
    log.append(_event("same", 1))

    try:
        log.append(_event("same", 2))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate event ids should fail")
