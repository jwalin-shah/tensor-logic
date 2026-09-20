from tensor_logic.lifeops_calendar_adapter import (
    calendar_events_to_primitives,
)


def _events():
    return [
        {
            "summary": "A",
            "start": "2026-09-23T13:00:00-07:00",
            "end": "2026-09-23T14:00:00-07:00",
            "location": "Place A",
            "account": "acct@example.com",
            "all_day": False,
            "event_id": "e1",
            "calendar_id": "primary",
        },
        {
            "summary": "B",
            "start": "2026-09-23T15:00:00-07:00",
            "end": "2026-09-23T16:00:00-07:00",
            "location": "Place B",
            "account": "acct@example.com",
            "all_day": False,
            "event_id": "e2",
            "calendar_id": "primary",
        },
    ]


def test_exact_calendar_events_normalize_to_query_primitives():
    bundle = calendar_events_to_primitives(_events())

    assert bundle.attends == {"me": ("e1", "e2")}
    assert bundle.located_at == {
        "e1": "Place A",
        "e2": "Place B",
    }
    assert bundle.starts_at["e1"] < bundle.starts_at["e2"]
    assert bundle.ends_at["e1"] < bundle.starts_at["e2"]
    assert bundle.source_refs["e1"] == {
        "kind": "calendar_event",
        "source": "google_calendar",
        "event_id": "e1",
        "calendar_id": "primary",
        "account": "acct@example.com",
    }
    assert bundle.excluded_event_ids == ()
    assert bundle.limitations == ()
    assert len(bundle.revision) == 64


def test_input_order_does_not_change_revision_or_attends_order():
    left = calendar_events_to_primitives(_events())
    right = calendar_events_to_primitives(list(reversed(_events())))

    assert left.revision == right.revision
    assert left.attends == right.attends


def test_all_day_or_missing_location_are_excluded_not_inferred():
    events = _events()
    events.extend(
        [
            {
                "summary": "All day",
                "start": "2026-09-24",
                "end": "2026-09-25",
                "location": "Somewhere",
                "all_day": True,
                "event_id": "e3",
            },
            {
                "summary": "No location",
                "start": "2026-09-24T10:00:00-07:00",
                "end": "2026-09-24T11:00:00-07:00",
                "location": "",
                "all_day": False,
                "event_id": "e4",
            },
        ]
    )

    bundle = calendar_events_to_primitives(events)

    assert bundle.attends == {"me": ("e1", "e2")}
    assert bundle.excluded_event_ids == ("e3", "e4")
    assert "e3:all_day_excluded" in bundle.limitations
    assert "e4:missing_location" in bundle.limitations


def test_timezone_naive_calendar_event_is_rejected():
    events = [
        {
            "start": "2026-09-23T13:00:00",
            "end": "2026-09-23T14:00:00",
            "location": "Place A",
            "all_day": False,
            "event_id": "e1",
        }
    ]

    bundle = calendar_events_to_primitives(events)

    assert bundle.attends == {"me": ()}
    assert bundle.excluded_event_ids == ("e1",)
    assert bundle.limitations == ("e1:invalid_time",)


def test_missing_event_id_is_limitation_not_synthetic_identity():
    events = [
        {
            "start": "2026-09-23T13:00:00-07:00",
            "end": "2026-09-23T14:00:00-07:00",
            "location": "Place A",
            "all_day": False,
        }
    ]

    bundle = calendar_events_to_primitives(events)

    assert bundle.attends == {"me": ()}
    assert bundle.source_refs == {}
    assert bundle.limitations == ("event[0]:missing_event_id",)
