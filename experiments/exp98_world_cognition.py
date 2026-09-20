"""exp98: end-to-end tensorized world plus replayable cognitive program."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path

from tensor_logic.cognitive_program import CognitiveProgramTrace
from tensor_logic.personal_attention import (
    AttentionSignals,
    DEFAULT_ATTENTION_POLICY,
    score_attention_candidates,
)
from tensor_logic.personal_physics import (
    FactRecord,
    admit_travel_gap,
    build_personal_physics_v0,
)
from tensor_logic.world_tensor import (
    CoordinateProvenance,
    build_world_tensor_schema,
    tensorize_personal_world,
)


def admitted(fact_id, relation, subject, object_, *evidence):
    return FactRecord(
        fact_id=fact_id,
        relation=relation,
        subject=subject,
        object=object_,
        status="admitted",
        evidence_refs=tuple(evidence),
        source_kind="synthetic_exp98",
        source_ref="exp98_fixture",
        admission_ref="admission:exp98",
    )


def build_world():
    hard = build_personal_physics_v0()
    for entity_id, entity_type in (
        ("me", "Person"),
        ("person_a", "Person"),
        ("person_b", "Person"),
        ("event_a", "Event"),
        ("event_b", "Event"),
        ("project_x", "Project"),
        ("goal_x", "Goal"),
        ("route_call", "ToolCall"),
        ("maps", "Tool"),
        ("route_evidence", "Evidence"),
        ("travel_claim", "Claim"),
    ):
        hard.register_entity(entity_id, entity_type)

    for fact in (
        admitted(
            "f-attend-a",
            "attends",
            "me",
            "event_a",
            "calendar:event_a",
        ),
        admitted(
            "f-attend-b",
            "attends",
            "me",
            "event_b",
            "calendar:event_b",
        ),
        admitted(
            "f-followup-request",
            "requested_followup",
            "me",
            "person_a",
            "message:followup",
        ),
        admitted(
            "f-followup-open",
            "not_completed_followup",
            "me",
            "person_a",
            "reconciliation:followup",
        ),
        admitted(
            "f-works-on",
            "works_on",
            "me",
            "project_x",
            "project:state",
        ),
        admitted(
            "f-supports-goal",
            "supports_goal",
            "project_x",
            "goal_x",
            "user:goal",
        ),
        admitted(
            "f-invoked",
            "invoked",
            "route_call",
            "maps",
            "receipt:route-call",
        ),
        admitted(
            "f-produced",
            "produced_evidence",
            "route_call",
            "route_evidence",
            "receipt:route-call",
        ),
        admitted(
            "f-supports-claim",
            "supports_claim",
            "route_evidence",
            "travel_claim",
            "route:response-hash",
        ),
    ):
        hard.add_fact(fact)

    admit_travel_gap(
        hard,
        fact_id="f-travel-gap",
        event_a="event_a",
        event_a_end_minute=660,
        event_b="event_b",
        event_b_start_minute=690,
        travel_minutes=43,
        evidence_refs=(
            "calendar:event_a",
            "calendar:event_b",
            "receipt:route-call",
        ),
        admission_ref="admission:route",
        route_call_ref="toolcall:route-call",
    )
    return hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="experiments/exp98_world_cognition_data/results.json",
    )
    args = parser.parse_args()

    hard = build_world()
    hard_tensor = tensorize_personal_world(hard)

    numeric_world = build_world_tensor_schema(
        people=("me", "person_a", "person_b"),
        events=("event_a", "event_b"),
        places=("place_a", "place_b"),
        time_buckets=("11:00",),
        modes=("drive",),
        projects=("project_x",),
        goals=("goal_x",),
        tools=("maps",),
        tool_calls=("route_call",),
        evidence=("route_evidence",),
        claims=("travel_claim",),
    )
    numeric_world.tensors["travel_minutes"].set(
        ("place_a", "place_b", "11:00", "drive"),
        43.0,
        provenance=CoordinateProvenance(
            evidence_refs=("receipt:route-call",),
            source_refs=("maps",),
            admission_ref="admission:route",
        ),
    )
    numeric_world.tensors["goal_priority"].set(
        ("goal_x",),
        0.9,
        provenance=CoordinateProvenance(
            evidence_refs=("user:goal",),
            source_refs=("human_authority",),
            admission_ref="admission:user",
        ),
    )

    followup = hard.derive(
        "unresolved_followup",
        "me",
        "person_a",
    )
    travel = hard.derive(
        "infeasible_transition",
        "event_a",
        "event_b",
    )

    signals = {
        "person_a": AttentionSignals(
            importance=0.6,
            staleness=0.3,
            unresolved_followup=1.0,
            shared_project=0.2,
            evidence_refs=("proof:followup",),
        ),
        "person_b": AttentionSignals(
            importance=0.9,
            staleness=0.8,
            unresolved_followup=0.0,
            shared_project=0.5,
            evidence_refs=("user:importance",),
        ),
    }
    ranked = score_attention_candidates(
        signals,
        DEFAULT_ATTENTION_POLICY,
    )

    trace = CognitiveProgramTrace(
        "personal-physics-world-cognition-v1"
    )
    trace.add_artifact(
        "observations",
        kind="observation_bundle",
        payload={
            "calendar": ["event_a", "event_b"],
            "messages": ["followup"],
            "routing": 43,
        },
        source_refs=(
            "calendar:event_a",
            "calendar:event_b",
            "message:followup",
            "receipt:route-call",
        ),
    )
    trace.add_artifact(
        "world-tensors",
        kind="world_tensor_snapshot",
        payload={
            "hard_tensor_digest": hard_tensor.digest,
            "numeric_tensor_digest": numeric_world.digest,
        },
    )
    trace.add_artifact(
        "derivations",
        kind="derived_state",
        payload={
            "unresolved_followup": asdict(followup),
            "infeasible_transition": asdict(travel),
        },
    )
    trace.add_artifact(
        "scores",
        kind="attention_scores",
        payload=[
            {
                "person_id": item.person_id,
                "score": item.score,
                "contributions": item.contributions,
                "policy_digest": item.policy_digest,
            }
            for item in ranked
        ],
    )
    trace.add_artifact(
        "ranking",
        kind="ranked_candidates",
        payload=[item.person_id for item in ranked],
    )
    trace.add_artifact(
        "proposal",
        kind="proposed_attention",
        payload={
            "surface_person": ranked[0].person_id,
            "reason_refs": [
                followup.derivation_digest,
                DEFAULT_ATTENTION_POLICY.digest,
            ],
        },
    )
    trace.add_artifact(
        "action",
        kind="surfaced_attention",
        payload={
            "person": ranked[0].person_id,
            "status": "shown",
        },
    )
    trace.add_artifact(
        "outcome",
        kind="verified_outcome",
        payload={
            "shown": True,
            "user_correction": False,
        },
    )

    trace.add_step(
        "observe",
        operator="OBSERVE",
        operator_id="lifeops.sources",
        operator_version="1",
        input_artifact_ids=(),
        output_artifact_ids=("observations",),
        evidence_refs=(
            "calendar:event_a",
            "calendar:event_b",
            "message:followup",
            "receipt:route-call",
        ),
    )
    trace.add_step(
        "admit",
        operator="ADMIT",
        operator_id="lifeops.admission",
        operator_version="1",
        input_artifact_ids=("observations",),
        output_artifact_ids=("world-tensors",),
        parent_step_ids=("observe",),
        evidence_refs=(
            "calendar:event_a",
            "calendar:event_b",
            "message:followup",
            "receipt:route-call",
            "user:goal",
        ),
    )
    trace.add_step(
        "derive",
        operator="DERIVE",
        operator_id="tensor_logic.personal_physics",
        operator_version="1",
        input_artifact_ids=("world-tensors",),
        output_artifact_ids=("derivations",),
        parent_step_ids=("admit",),
    )
    trace.add_step(
        "score",
        operator="SCORE",
        operator_id="tensor_logic.personal_attention",
        operator_version=DEFAULT_ATTENTION_POLICY.version,
        input_artifact_ids=("derivations",),
        output_artifact_ids=("scores",),
        parent_step_ids=("derive",),
    )
    trace.add_step(
        "compare",
        operator="COMPARE",
        operator_id="stable_sort",
        operator_version="1",
        input_artifact_ids=("scores",),
        output_artifact_ids=("ranking",),
        parent_step_ids=("score",),
    )
    trace.add_step(
        "propose",
        operator="PROPOSE",
        operator_id="attention.surface",
        operator_version="1",
        input_artifact_ids=("ranking", "derivations"),
        output_artifact_ids=("proposal",),
        parent_step_ids=("compare",),
    )
    trace.add_step(
        "act",
        operator="ACT",
        operator_id="ui.surface",
        operator_version="1",
        input_artifact_ids=("proposal",),
        output_artifact_ids=("action",),
        parent_step_ids=("propose",),
        receipt_refs=("receipt:exp98-surface",),
    )
    trace.add_step(
        "verify",
        operator="VERIFY",
        operator_id="outcome.check",
        operator_version="1",
        input_artifact_ids=("action",),
        output_artifact_ids=("outcome",),
        parent_step_ids=("act",),
        receipt_refs=("receipt:exp98-surface",),
    )

    result = {
        "experiment": "exp98_world_cognition",
        "hard_world_digest": hard.world_digest,
        "hard_tensor_digest": hard_tensor.digest,
        "numeric_tensor_digest": numeric_world.digest,
        "travel_tensor_coordinate": {
            "coordinate": [
                "place_a",
                "place_b",
                "11:00",
                "drive",
            ],
            "minutes": numeric_world.tensors[
                "travel_minutes"
            ].get(
                (
                    "place_a",
                    "place_b",
                    "11:00",
                    "drive",
                )
            ),
        },
        "hard_derivations": {
            "unresolved_followup": followup.entailed,
            "infeasible_transition": travel.entailed,
        },
        "attention_ranking": [
            item.person_id for item in ranked
        ],
        "cognitive_program": trace.replay_manifest(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("exp98: tensorized world + cognitive program")
    print(
        "travel tensor: place_a -> place_b at 11:00 drive = "
        f"{result['travel_tensor_coordinate']['minutes']} min"
    )
    print(
        "hard derivations: "
        f"followup={followup.entailed} "
        f"travel={travel.entailed}"
    )
    print(
        "attention ranking: "
        + " > ".join(result["attention_ranking"])
    )
    print(
        "process digest: "
        f"{result['cognitive_program']['process_digest']}"
    )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
