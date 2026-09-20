"""exp96: Personal Physics v0 replayable mental-load scenario."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
import subprocess
from pathlib import Path

from tensor_logic.personal_physics import (
    FactRecord,
    admit_interval_overlap,
    admit_travel_gap,
    build_personal_physics_v0,
)


def current_commit() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def admitted(
    fact_id: str,
    relation: str,
    subject: str,
    object_: str,
    *evidence_refs: str,
) -> FactRecord:
    return FactRecord(
        fact_id=fact_id,
        relation=relation,
        subject=subject,
        object=object_,
        status="admitted",
        evidence_refs=tuple(evidence_refs),
        source_kind="synthetic_exp96",
        source_ref="exp96_fixture",
        admission_ref="admission:exp96",
    )


def build_world():
    world = build_personal_physics_v0()

    for entity_id, entity_type in (
        ("me", "Person"),
        ("person_b", "Person"),
        ("event_focus", "Event"),
        ("event_meeting", "Event"),
        ("project_x", "Project"),
        ("goal_x", "Goal"),
        ("route_call", "ToolCall"),
        ("maps", "Tool"),
        ("route_evidence", "Evidence"),
        ("travel_claim", "Claim"),
    ):
        world.register_entity(entity_id, entity_type)

    for fact in (
        admitted(
            "f-attend-focus",
            "attends",
            "me",
            "event_focus",
            "calendar:focus",
        ),
        admitted(
            "f-attend-meeting",
            "attends",
            "me",
            "event_meeting",
            "calendar:meeting",
        ),
        admitted(
            "f-followup-requested",
            "requested_followup",
            "me",
            "person_b",
            "message:followup",
        ),
        admitted(
            "f-followup-open",
            "not_completed_followup",
            "me",
            "person_b",
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
            "goal:declaration",
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
        world.add_fact(fact)

    admit_interval_overlap(
        world,
        fact_id="f-overlap",
        event_a="event_focus",
        start_a_minute=600,
        end_a_minute=660,
        event_b="event_meeting",
        start_b_minute=645,
        end_b_minute=705,
        evidence_refs=(
            "calendar:focus",
            "calendar:meeting",
        ),
        admission_ref="admission:calendar",
    )
    admit_travel_gap(
        world,
        fact_id="f-travel-gap",
        event_a="event_focus",
        event_a_end_minute=660,
        event_b="event_meeting",
        event_b_start_minute=690,
        travel_minutes=43,
        evidence_refs=(
            "calendar:focus",
            "calendar:meeting",
            "receipt:route-call",
        ),
        admission_ref="admission:route",
        route_call_ref="toolcall:route_call",
    )
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="experiments/exp96_personal_physics_data/results.json",
    )
    args = parser.parse_args()

    world = build_world()

    queries = {
        "calendar_conflict": (
            "schedule_conflict",
            "event_focus",
            "event_meeting",
        ),
        "travel_infeasible": (
            "infeasible_transition",
            "event_focus",
            "event_meeting",
        ),
        "followup_open": (
            "unresolved_followup",
            "me",
            "person_b",
        ),
        "goal_work": (
            "active_goal_work",
            "me",
            "goal_x",
        ),
        "tool_backed_claim": (
            "tool_backed_claim",
            "maps",
            "travel_claim",
        ),
    }

    initial = {
        name: asdict(world.derive(*query))
        for name, query in queries.items()
    }

    before_retraction = initial["travel_infeasible"]
    world.set_fact_status("f-travel-gap", "retracted")
    after_retraction = asdict(
        world.derive(
            "infeasible_transition",
            "event_focus",
            "event_meeting",
        )
    )

    result = {
        "experiment": "exp96_personal_physics_v0",
        "provenance": {
            "commit": current_commit(),
            "fixture": "synthetic_no_personal_data",
            "engine": "tensor_logic.personal_physics",
        },
        "initial_world_digest": before_retraction[
            "world_digest"
        ],
        "derivations": initial,
        "retraction_check": {
            "retracted_fact_id": "f-travel-gap",
            "before": before_retraction,
            "after": after_retraction,
            "dependent_conclusion_removed": (
                before_retraction["entailed"]
                and not after_retraction["entailed"]
            ),
        },
        "admission_boundary": {
            "only_status_used_as_premise": "admitted",
            "unknown_is_false": False,
            "llm_candidate_auto_admitted": False,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("exp96: Personal Physics v0")
    for name, derivation in initial.items():
        print(
            f"{name}: entailed={derivation['entailed']} "
            f"digest={derivation['derivation_digest']}"
        )
    print(
        "retraction removed travel conclusion: "
        f"{result['retraction_check']['dependent_conclusion_removed']}"
    )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
