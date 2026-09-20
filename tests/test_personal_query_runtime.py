from tensor_logic.personal_query_catalog import (
    RELATIONSHIP_TARGET,
    TRAVEL_TARGET,
)
from tensor_logic.personal_query_runtime import preflight_personal_query
from tensor_logic.query_execution import QueryExecutionStatus
from tensor_logic.query_runtime import SourceState


def _health():
    return {
        "providers": {
            "checked_at": "2026-09-20T17:20:00+00:00",
            "providers": [],
        },
        "capture": {
            "sources": [
                {
                    "key": "google_calendar:acct",
                    "source_id": "google_calendar",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:19:50+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "cal1",
                    "item_count": 1,
                },
                {
                    "key": "gmail:acct",
                    "source_id": "gmail",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:19:40+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "m1",
                    "item_count": 1,
                },
                {
                    "key": "imessage:",
                    "source_id": "imessage",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:19:55+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "i1",
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
                {
                    "key": "linkedin:",
                    "source_id": "linkedin",
                    "readable": False,
                    "status": "not_configured",
                    "last_success_at": "",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "",
                    "item_count": 0,
                },
                {
                    "key": "google_tasks:acct",
                    "source_id": "google_tasks",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:19:30+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "t1",
                    "item_count": 1,
                },
                {
                    "key": "apple_reminders:",
                    "source_id": "apple_reminders",
                    "readable": True,
                    "status": "ok",
                    "last_success_at": "2026-09-20T17:19:20+00:00",
                    "checked_at": "2026-09-20T17:20:00+00:00",
                    "newest_seen_id": "r1",
                    "item_count": 1,
                },
            ],
        },
    }


def test_travel_preflight_blocks_without_route_engine():
    result = preflight_personal_query(TRAVEL_TARGET, _health())

    assert result.assessment.status == QueryExecutionStatus.BLOCKED
    assert result.assessment.reasons == ("missing:route_engine",)


def test_travel_preflight_is_complete_with_fresh_route_state():
    result = preflight_personal_query(
        TRAVEL_TARGET,
        _health(),
        extra_states={
            "route_engine": SourceState(
                "route_engine",
                readable=True,
                age_seconds=5,
                revision="route:1",
            ),
        },
    )

    assert result.assessment.status == QueryExecutionStatus.READY_COMPLETE
    assert result.assessment.runnable is True
    assert result.query.target == TRAVEL_TARGET
    assert result.query.digest


def test_relationship_preflight_is_ready_partial_with_missing_optional_channels():
    result = preflight_personal_query(
        RELATIONSHIP_TARGET,
        _health(),
        extra_states={
            "human_policy": SourceState(
                "human_policy",
                readable=True,
                age_seconds=0,
                revision="policy:1",
            ),
        },
    )

    assert result.assessment.status == QueryExecutionStatus.READY_PARTIAL
    assert result.assessment.runnable is True
    assert "coverage_partial:relationship_messages" in (
        result.assessment.reasons
    )
    assert result.assessment.complete_coverage is False


def test_relationship_preflight_blocks_without_human_policy():
    result = preflight_personal_query(
        RELATIONSHIP_TARGET,
        _health(),
    )

    assert result.assessment.status == QueryExecutionStatus.BLOCKED
    assert "missing:human_policy" in result.assessment.reasons


def test_preflight_records_only_relevant_source_revisions():
    result = preflight_personal_query(
        RELATIONSHIP_TARGET,
        _health(),
        extra_states={
            "human_policy": SourceState(
                "human_policy",
                readable=True,
                age_seconds=0,
                revision="policy:1",
            ),
            "route_engine": SourceState(
                "route_engine",
                readable=True,
                age_seconds=0,
                revision="route:ignored",
            ),
        },
    )

    sources = dict(result.state_revisions)

    assert "human_policy" in sources
    assert "capture-source:gmail" in sources
    assert "capture-source:google_tasks" in sources
    assert "route_engine" not in sources
