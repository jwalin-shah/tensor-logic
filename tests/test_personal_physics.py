from tensor_logic.personal_physics import (
    FactRecord,
    admit_interval_overlap,
    admit_travel_gap,
    build_personal_physics_v0,
)


def admitted(
    fact_id,
    relation,
    subject,
    object_,
    *,
    evidence=("ev",),
    source_kind="fixture",
):
    return FactRecord(
        fact_id=fact_id,
        relation=relation,
        subject=subject,
        object=object_,
        status="admitted",
        evidence_refs=tuple(evidence),
        source_kind=source_kind,
        source_ref="fixture",
        admission_ref="admission:test",
    )


def register_calendar_entities(world):
    world.register_entity("me", "Person")
    world.register_entity("event_a", "Event")
    world.register_entity("event_b", "Event")


def test_calendar_conflict_has_replayable_proof_and_retracts():
    world = build_personal_physics_v0()
    register_calendar_entities(world)
    world.add_fact(admitted("f-attend-a", "attends", "me", "event_a"))
    world.add_fact(admitted("f-attend-b", "attends", "me", "event_b"))
    overlap = admit_interval_overlap(
        world,
        fact_id="f-overlap",
        event_a="event_a",
        start_a_minute=600,
        end_a_minute=660,
        event_b="event_b",
        start_b_minute=630,
        end_b_minute=690,
        evidence_refs=("calendar:event_a", "calendar:event_b"),
        admission_ref="admission:calendar",
    )
    assert overlap is not None

    result = world.derive(
        "schedule_conflict",
        "event_a",
        "event_b",
    )
    assert result.entailed is True
    assert result.proof["rule_id"] == "R-calendar-conflict"
    primitive_ids = {
        fact_id
        for premise in result.proof["premises"]
        for fact_id in premise["fact_ids"]
    }
    assert primitive_ids == {
        "f-attend-a",
        "f-attend-b",
        "f-overlap",
    }

    before = result.world_digest
    world.set_fact_status("f-overlap", "retracted")
    after = world.derive(
        "schedule_conflict",
        "event_a",
        "event_b",
    )
    assert after.entailed is False
    assert after.world_digest != before


def test_candidate_fact_cannot_be_used_as_proof_premise():
    world = build_personal_physics_v0()
    register_calendar_entities(world)
    world.add_fact(admitted("f-attend-a", "attends", "me", "event_a"))
    world.add_fact(admitted("f-attend-b", "attends", "me", "event_b"))
    world.add_fact(
        FactRecord(
            fact_id="f-overlap-candidate",
            relation="overlaps",
            subject="event_a",
            object="event_b",
            status="candidate",
            evidence_refs=("llm:guess",),
            source_kind="llm",
            source_ref="model:test",
            confidence=0.99,
        )
    )

    result = world.derive(
        "schedule_conflict",
        "event_a",
        "event_b",
    )
    assert result.entailed is False


def test_unknown_is_not_false_for_followup():
    world = build_personal_physics_v0()
    world.register_entity("me", "Person")
    world.register_entity("person_b", "Person")
    world.add_fact(
        admitted(
            "f-request",
            "requested_followup",
            "me",
            "person_b",
        )
    )

    unknown = world.derive(
        "unresolved_followup",
        "me",
        "person_b",
    )
    assert unknown.entailed is False

    world.add_fact(
        admitted(
            "f-not-completed",
            "not_completed_followup",
            "me",
            "person_b",
            evidence=("explicit:reconciliation",),
        )
    )
    explicit = world.derive(
        "unresolved_followup",
        "me",
        "person_b",
    )
    assert explicit.entailed is True
    assert explicit.proof["rule_id"] == "R-unresolved-followup"


def test_routing_adapter_becomes_traceable_primitive_for_tensor_logic():
    world = build_personal_physics_v0()
    register_calendar_entities(world)
    world.add_fact(admitted("f-attend-a", "attends", "me", "event_a"))
    world.add_fact(admitted("f-attend-b", "attends", "me", "event_b"))

    travel = admit_travel_gap(
        world,
        fact_id="f-route-gap",
        event_a="event_a",
        event_a_end_minute=990,
        event_b="event_b",
        event_b_start_minute=1020,
        travel_minutes=43,
        evidence_refs=("route:call-17", "calendar:event_b"),
        admission_ref="admission:route",
        route_call_ref="toolcall:route-17",
    )
    assert travel is not None
    assert travel.metadata["available_gap_minutes"] == 30

    result = world.derive(
        "infeasible_transition",
        "event_a",
        "event_b",
    )
    assert result.entailed is True
    assert result.proof["rule_id"] == "R-travel-infeasible"
    primitive_ids = {
        fact_id
        for premise in result.proof["premises"]
        for fact_id in premise["fact_ids"]
    }
    assert "f-route-gap" in primitive_ids


def test_tool_call_evidence_is_part_of_the_proof_graph():
    world = build_personal_physics_v0()
    world.register_entity("route_call", "ToolCall")
    world.register_entity("maps", "Tool")
    world.register_entity("route_evidence", "Evidence")
    world.register_entity("travel_claim", "Claim")

    world.add_fact(
        admitted(
            "f-invoked",
            "invoked",
            "route_call",
            "maps",
            evidence=("receipt:route_call",),
        )
    )
    world.add_fact(
        admitted(
            "f-produced",
            "produced_evidence",
            "route_call",
            "route_evidence",
            evidence=("receipt:route_call",),
        )
    )
    world.add_fact(
        admitted(
            "f-support",
            "supports_claim",
            "route_evidence",
            "travel_claim",
            evidence=("route:response-hash",),
        )
    )

    result = world.derive(
        "tool_backed_claim",
        "maps",
        "travel_claim",
    )
    assert result.entailed is True
    assert result.proof["rule_id"] == "R-tool-backed-claim"
    fact_ids = {
        fact_id
        for premise in result.proof["premises"]
        for fact_id in premise["fact_ids"]
    }
    assert fact_ids == {
        "f-invoked",
        "f-produced",
        "f-support",
    }


def _build_replay_world():
    world = build_personal_physics_v0()
    register_calendar_entities(world)
    world.add_fact(admitted("f-attend-a", "attends", "me", "event_a"))
    world.add_fact(admitted("f-attend-b", "attends", "me", "event_b"))
    admit_interval_overlap(
        world,
        fact_id="f-overlap",
        event_a="event_a",
        start_a_minute=600,
        end_a_minute=660,
        event_b="event_b",
        start_b_minute=630,
        end_b_minute=690,
        evidence_refs=("calendar:event_a", "calendar:event_b"),
        admission_ref="admission:calendar",
    )
    return world


def test_replay_is_deterministic():
    left = _build_replay_world().derive(
        "schedule_conflict",
        "event_a",
        "event_b",
    )
    right = _build_replay_world().derive(
        "schedule_conflict",
        "event_a",
        "event_b",
    )

    assert left.world_digest == right.world_digest
    assert left.derivation_digest == right.derivation_digest
    assert left.proof == right.proof
