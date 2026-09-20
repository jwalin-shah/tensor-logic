from tensor_logic.query_coverage import (
    CoverageRequirement,
    CoverageStatus,
    assess_coverage,
)
from tensor_logic.query_runtime import SourceState


def _states():
    return {
        "gmail": SourceState("gmail", readable=True, age_seconds=5),
        "imessage": SourceState("imessage", readable=True, age_seconds=2),
        "whatsapp": SourceState("whatsapp", readable=False),
        "linkedin": SourceState("linkedin", readable=False),
    }


def test_partial_coverage_is_usable_without_claiming_complete():
    result = assess_coverage(
        CoverageRequirement(
            "relationship_messages",
            ("gmail", "imessage", "whatsapp", "linkedin"),
            minimum_readable=2,
            max_age_seconds=60,
        ),
        _states(),
    )

    assert result.status == CoverageStatus.PARTIAL
    assert result.usable is True
    assert result.complete is False
    assert result.coverage_fraction == 0.5
    assert result.readable_sources == ("gmail", "imessage")
    assert result.unavailable_sources == ("linkedin", "whatsapp")


def test_complete_coverage_requires_every_intended_source():
    states = _states()
    states["whatsapp"] = SourceState(
        "whatsapp", readable=True, age_seconds=4
    )
    states["linkedin"] = SourceState(
        "linkedin", readable=True, age_seconds=3
    )

    result = assess_coverage(
        CoverageRequirement(
            "relationship_messages",
            ("gmail", "imessage", "whatsapp", "linkedin"),
            minimum_readable=2,
            max_age_seconds=60,
        ),
        states,
    )

    assert result.status == CoverageStatus.COMPLETE
    assert result.usable is True
    assert result.complete is True
    assert result.coverage_fraction == 1.0


def test_stale_readable_source_does_not_count_as_fresh_coverage():
    states = _states()
    states["gmail"] = SourceState(
        "gmail", readable=True, age_seconds=600
    )

    result = assess_coverage(
        CoverageRequirement(
            "relationship_messages",
            ("gmail", "imessage"),
            minimum_readable=2,
            max_age_seconds=60,
        ),
        states,
    )

    assert result.status == CoverageStatus.BLOCKED
    assert result.usable is False
    assert result.stale_sources == ("gmail",)
    assert result.readable_sources == ("imessage",)


def test_missing_source_is_explicit_not_equivalent_to_unavailable():
    states = {"gmail": SourceState("gmail", readable=True, age_seconds=1)}

    result = assess_coverage(
        CoverageRequirement(
            "messages",
            ("gmail", "imessage"),
            minimum_readable=1,
        ),
        states,
    )

    assert result.status == CoverageStatus.PARTIAL
    assert result.missing_sources == ("imessage",)
    assert result.unavailable_sources == ()
