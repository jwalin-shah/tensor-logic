from tensor_logic.personal_attention import (
    AttentionPolicy,
    AttentionSignals,
    DEFAULT_ATTENTION_POLICY,
    score_attention_candidates,
)


def test_attention_score_equals_visible_contributions():
    signals = {
        "person_a": AttentionSignals(
            importance=0.8,
            staleness=0.4,
            unresolved_followup=1.0,
            shared_project=0.2,
            evidence_refs=("message:a", "calendar:a"),
        ),
    }

    result = score_attention_candidates(
        signals,
        DEFAULT_ATTENTION_POLICY,
    )[0]

    assert result.candidate_only is True
    assert abs(
        result.score - sum(result.contributions.values())
    ) < 1e-6
    assert result.evidence_refs == (
        "message:a",
        "calendar:a",
    )


def test_policy_version_can_reorder_people_without_changing_signals():
    signals = {
        "person_a": AttentionSignals(
            importance=0.3,
            staleness=0.1,
            unresolved_followup=1.0,
            shared_project=0.0,
        ),
        "person_b": AttentionSignals(
            importance=1.0,
            staleness=0.9,
            unresolved_followup=0.0,
            shared_project=0.9,
        ),
    }

    followup_policy = AttentionPolicy(
        version="followup-first",
        importance_weight=0.10,
        staleness_weight=0.05,
        unresolved_followup_weight=0.80,
        shared_project_weight=0.05,
    )
    relationship_policy = AttentionPolicy(
        version="relationship-first",
        importance_weight=0.45,
        staleness_weight=0.25,
        unresolved_followup_weight=0.05,
        shared_project_weight=0.25,
    )

    followup_order = [
        item.person_id
        for item in score_attention_candidates(
            signals,
            followup_policy,
        )
    ]
    relationship_order = [
        item.person_id
        for item in score_attention_candidates(
            signals,
            relationship_policy,
        )
    ]

    assert followup_order == ["person_a", "person_b"]
    assert relationship_order == ["person_b", "person_a"]
    assert signals["person_a"].unresolved_followup == 1.0


def test_policy_digest_is_reproducible_and_versioned():
    one = AttentionPolicy(
        version="v1",
        importance_weight=0.25,
        staleness_weight=0.25,
        unresolved_followup_weight=0.25,
        shared_project_weight=0.25,
    )
    two = AttentionPolicy(
        version="v1",
        importance_weight=0.25,
        staleness_weight=0.25,
        unresolved_followup_weight=0.25,
        shared_project_weight=0.25,
    )
    three = AttentionPolicy(
        version="v2",
        importance_weight=0.25,
        staleness_weight=0.25,
        unresolved_followup_weight=0.25,
        shared_project_weight=0.25,
    )

    assert one.digest == two.digest
    assert one.digest != three.digest


def test_attention_signals_reject_out_of_range_values():
    try:
        AttentionSignals(importance=1.2)
    except ValueError as exc:
        assert "importance" in str(exc)
    else:
        raise AssertionError("expected validation error")
