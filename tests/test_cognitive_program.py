from tensor_logic.cognitive_program import CognitiveProgramTrace


def _build_trace():
    trace = CognitiveProgramTrace("attention-flow-v1")

    trace.add_artifact(
        "a-observations",
        kind="observation_bundle",
        payload={
            "calendar": ["event_a", "event_b"],
            "route_call": "route:17",
            "messages": ["msg:1"],
        },
        source_refs=("calendar:read", "route:17", "message:1"),
    )
    trace.add_artifact(
        "a-candidates",
        kind="candidate_facts",
        payload=[
            ["requested_followup", "me", "person_a"],
            ["insufficient_travel_gap", "event_a", "event_b"],
        ],
    )
    trace.add_artifact(
        "a-admitted",
        kind="admitted_facts",
        payload=[
            ["requested_followup", "me", "person_a"],
            ["insufficient_travel_gap", "event_a", "event_b"],
        ],
    )
    trace.add_artifact(
        "a-derived",
        kind="derivation",
        payload={
            "unresolved_followup": True,
            "infeasible_transition": True,
            "proof_digest": "proof:123",
        },
    )
    trace.add_artifact(
        "a-scores",
        kind="attention_scores",
        payload={
            "person_a": 0.76,
            "person_b": 0.43,
        },
    )
    trace.add_artifact(
        "a-ranking",
        kind="ranked_candidates",
        payload=["person_a", "person_b"],
    )
    trace.add_artifact(
        "a-proposal",
        kind="proposed_action",
        payload={
            "action": "surface_followup",
            "person": "person_a",
        },
    )
    trace.add_artifact(
        "a-action",
        kind="executed_action",
        payload={
            "action": "surface_followup",
            "status": "shown",
        },
    )
    trace.add_artifact(
        "a-outcome",
        kind="verified_outcome",
        payload={
            "shown": True,
            "user_correction": False,
        },
    )
    trace.add_artifact(
        "a-revision",
        kind="revision",
        payload={
            "policy_changed": False,
            "fact_retracted": False,
        },
    )

    trace.add_step(
        "s-observe",
        operator="OBSERVE",
        operator_id="lifeops.read",
        operator_version="1",
        input_artifact_ids=(),
        output_artifact_ids=("a-observations",),
        evidence_refs=("calendar:read", "route:17", "message:1"),
    )
    trace.add_step(
        "s-normalize",
        operator="NORMALIZE",
        operator_id="personal_physics.normalize",
        operator_version="1",
        input_artifact_ids=("a-observations",),
        output_artifact_ids=("a-candidates",),
        parent_step_ids=("s-observe",),
    )
    trace.add_step(
        "s-admit",
        operator="ADMIT",
        operator_id="lifeops.admission",
        operator_version="1",
        input_artifact_ids=("a-candidates",),
        output_artifact_ids=("a-admitted",),
        parent_step_ids=("s-normalize",),
        evidence_refs=("calendar:read", "route:17", "message:1"),
    )
    trace.add_step(
        "s-derive",
        operator="DERIVE",
        operator_id="tensor_logic.personal_physics",
        operator_version="1",
        input_artifact_ids=("a-admitted",),
        output_artifact_ids=("a-derived",),
        parent_step_ids=("s-admit",),
    )
    trace.add_step(
        "s-score",
        operator="SCORE",
        operator_id="tensor_logic.personal_attention",
        operator_version="attention-v1",
        input_artifact_ids=("a-derived",),
        output_artifact_ids=("a-scores",),
        parent_step_ids=("s-derive",),
    )
    trace.add_step(
        "s-compare",
        operator="COMPARE",
        operator_id="stable_sort",
        operator_version="1",
        input_artifact_ids=("a-scores",),
        output_artifact_ids=("a-ranking",),
        parent_step_ids=("s-score",),
    )
    trace.add_step(
        "s-propose",
        operator="PROPOSE",
        operator_id="attention.surface",
        operator_version="1",
        input_artifact_ids=("a-ranking", "a-derived"),
        output_artifact_ids=("a-proposal",),
        parent_step_ids=("s-compare",),
    )
    trace.add_step(
        "s-act",
        operator="ACT",
        operator_id="ui.surface",
        operator_version="1",
        input_artifact_ids=("a-proposal",),
        output_artifact_ids=("a-action",),
        parent_step_ids=("s-propose",),
        receipt_refs=("receipt:surface-1",),
    )
    trace.add_step(
        "s-verify",
        operator="VERIFY",
        operator_id="outcome.check",
        operator_version="1",
        input_artifact_ids=("a-action",),
        output_artifact_ids=("a-outcome",),
        parent_step_ids=("s-act",),
        receipt_refs=("receipt:surface-1",),
    )
    trace.add_step(
        "s-revise",
        operator="REVISE",
        operator_id="policy.revision",
        operator_version="1",
        input_artifact_ids=("a-outcome",),
        output_artifact_ids=("a-revision",),
        parent_step_ids=("s-verify",),
    )
    return trace


def test_cognitive_program_digest_is_reproducible():
    left = _build_trace()
    right = _build_trace()

    assert left.process_digest == right.process_digest
    assert left.replay_manifest() == right.replay_manifest()


def test_act_requires_receipt():
    trace = CognitiveProgramTrace("receipt-boundary")
    trace.add_artifact(
        "proposal",
        kind="proposal",
        payload={"action": "send"},
    )
    trace.add_artifact(
        "action",
        kind="action",
        payload={"status": "unknown"},
    )

    try:
        trace.add_step(
            "act",
            operator="ACT",
            operator_id="tool.send",
            operator_version="1",
            input_artifact_ids=("proposal",),
            output_artifact_ids=("action",),
        )
    except ValueError as exc:
        assert "receipt" in str(exc).lower()
    else:
        raise AssertionError("ACT without receipt should fail")


def test_admit_requires_evidence():
    trace = CognitiveProgramTrace("admission-boundary")
    trace.add_artifact(
        "candidate",
        kind="candidate",
        payload={"fact": "x"},
    )
    trace.add_artifact(
        "admitted",
        kind="admitted",
        payload={"fact": "x"},
    )

    try:
        trace.add_step(
            "admit",
            operator="ADMIT",
            operator_id="admission",
            operator_version="1",
            input_artifact_ids=("candidate",),
            output_artifact_ids=("admitted",),
        )
    except ValueError as exc:
        assert "evidence" in str(exc).lower()
    else:
        raise AssertionError("ADMIT without evidence should fail")


def test_replay_manifest_contains_no_free_form_reasoning_requirement():
    trace = _build_trace()
    manifest = trace.replay_manifest()

    assert manifest["process_digest"]
    assert {
        step["operator"]
        for step in manifest["steps"]
    } == {
        "OBSERVE",
        "NORMALIZE",
        "ADMIT",
        "DERIVE",
        "SCORE",
        "COMPARE",
        "PROPOSE",
        "ACT",
        "VERIFY",
        "REVISE",
    }
    for step in manifest["steps"]:
        assert "reasoning_text" not in step
