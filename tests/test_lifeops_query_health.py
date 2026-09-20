from tensor_logic.lifeops_query_health import lifeops_source_states
from tensor_logic.query_runtime import (
    PrimitiveTensorSpec,
    QueryRegistry,
    SourceRequirement,
)


def _health():
    return {
        "providers": {
            "checked_at": "2026-09-20T17:20:00+00:00",
            "providers": [
                {
                    "provider": "google_gmail",
                    "configured": True,
                    "authenticated": True,
                    "readable": True,
                    "syncable": True,
                    "writable": True,
                    "blockers": [],
                },
                {
                    "provider": "whatsapp",
                    "configured": False,
                    "authenticated": False,
                    "readable": False,
                    "syncable": False,
                    "writable": False,
                    "blockers": ["backing_store_missing"],
                },
            ],
        },
        "capture": {
            "sources": [
                {
                    "key": "gmail:acct-a",
                    "source_id": "gmail",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:19:50+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "m1",
                    "item_count": 1,
                },
                {
                    "key": "gmail:acct-b",
                    "source_id": "gmail",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:10:00+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "m2",
                    "item_count": 1,
                },
                {
                    "key": "whatsapp:",
                    "source_id": "whatsapp",
                    "readable": False,
                    "status": "not_configured",
                    "last_success_at": "",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "",
                    "item_count": 0,
                },
            ],
        },
    }


def test_provider_and_capture_health_are_separate_namespaces():
    states = lifeops_source_states(_health())

    assert states["provider:google_gmail"].readable is True
    assert states["capture:gmail:acct-a"].readable is True
    assert states["capture:gmail:acct-a"].age_seconds == 10.0


def test_aggregate_capture_source_uses_freshest_readable_account():
    states = lifeops_source_states(_health())

    gmail = states["capture-source:gmail"]

    assert gmail.readable is True
    assert gmail.age_seconds == 10.0
    assert gmail.revision is not None


def test_unconfigured_capture_source_stays_unreadable():
    states = lifeops_source_states(_health())

    assert states["capture-source:whatsapp"].readable is False
    assert states["capture-source:whatsapp"].age_seconds is None
    assert states["provider:whatsapp"].readable is False


def test_query_freshness_uses_capture_not_provider_authentication():
    registry = QueryRegistry()
    registry.add_primitive(
        PrimitiveTensorSpec(
            "message_sender",
            (
                SourceRequirement(
                    "capture-source:gmail",
                    max_age_seconds=30,
                ),
            ),
        )
    )
    query = registry.compile("message_sender")
    states = lifeops_source_states(_health())

    assert registry.readiness(query, states).ready is True

    stale_health = _health()
    for capture in stale_health["capture"]["sources"]:
        if capture["source_id"] == "gmail":
            capture["last_success_at"] = "2026-09-20T16:00:00+00:00"

    stale_states = lifeops_source_states(stale_health)
    readiness = registry.readiness(query, stale_states)

    assert readiness.ready is False
    assert readiness.stale_sources == ("capture-source:gmail",)


def test_exact_capture_requirement_can_require_specific_account():
    registry = QueryRegistry()
    registry.add_primitive(
        PrimitiveTensorSpec(
            "account_specific_mail",
            (
                SourceRequirement(
                    "capture:gmail:acct-b",
                    max_age_seconds=300,
                ),
            ),
        )
    )
    query = registry.compile("account_specific_mail")
    states = lifeops_source_states(_health())
    readiness = registry.readiness(query, states)

    assert readiness.ready is False
    assert readiness.stale_sources == ("capture:gmail:acct-b",)
