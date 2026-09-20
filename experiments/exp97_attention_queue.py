"""exp97: adaptable relationship attention over a fixed admitted world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensor_logic.personal_attention import (
    AttentionPolicy,
    AttentionSignals,
    DEFAULT_ATTENTION_POLICY,
    score_attention_candidates,
)
from tensor_logic.personal_physics import (
    FactRecord,
    build_personal_physics_v0,
)


def _admitted(
    fact_id: str,
    relation: str,
    subject: str,
    object_: str,
) -> FactRecord:
    return FactRecord(
        fact_id=fact_id,
        relation=relation,
        subject=subject,
        object=object_,
        status="admitted",
        evidence_refs=(f"fixture:{fact_id}",),
        source_kind="synthetic_exp97",
        source_ref="exp97_fixture",
        admission_ref="admission:exp97",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="experiments/exp97_attention_queue_data/results.json",
    )
    args = parser.parse_args()

    world = build_personal_physics_v0()
    for person in ("me", "person_a", "person_b", "person_c"):
        world.register_entity(person, "Person")

    world.add_fact(
        _admitted(
            "f-requested-a",
            "requested_followup",
            "me",
            "person_a",
        )
    )
    world.add_fact(
        _admitted(
            "f-open-a",
            "not_completed_followup",
            "me",
            "person_a",
        )
    )
    hard = world.derive(
        "unresolved_followup",
        "me",
        "person_a",
    )
    if not hard.entailed:
        raise AssertionError("fixture should contain an unresolved follow-up")

    signals = {
        "person_a": AttentionSignals(
            importance=0.50,
            staleness=0.20,
            unresolved_followup=1.0,
            shared_project=0.10,
            evidence_refs=("proof:unresolved-a",),
        ),
        "person_b": AttentionSignals(
            importance=0.90,
            staleness=0.80,
            unresolved_followup=0.0,
            shared_project=0.60,
            evidence_refs=("policy:user-importance",),
        ),
        "person_c": AttentionSignals(
            importance=0.40,
            staleness=0.90,
            unresolved_followup=0.0,
            shared_project=0.0,
            evidence_refs=("history:last-interaction",),
        ),
    }

    relationship_first = AttentionPolicy(
        version="relationship-first-v1",
        importance_weight=0.45,
        staleness_weight=0.30,
        unresolved_followup_weight=0.10,
        shared_project_weight=0.15,
    )

    world_digest_before = world.world_digest
    default_queue = score_attention_candidates(
        signals,
        DEFAULT_ATTENTION_POLICY,
    )
    relationship_queue = score_attention_candidates(
        signals,
        relationship_first,
    )
    world_digest_after = world.world_digest

    result = {
        "experiment": "exp97_attention_queue",
        "hard_fact": {
            "unresolved_followup_person_a": hard.entailed,
            "derivation_digest": hard.derivation_digest,
        },
        "world_digest_before": world_digest_before,
        "world_digest_after": world_digest_after,
        "world_mutated_by_policy": (
            world_digest_before != world_digest_after
        ),
        "default_policy": {
            "version": DEFAULT_ATTENTION_POLICY.version,
            "digest": DEFAULT_ATTENTION_POLICY.digest,
            "queue": [
                {
                    "person_id": item.person_id,
                    "score": item.score,
                    "contributions": item.contributions,
                    "candidate_only": item.candidate_only,
                }
                for item in default_queue
            ],
        },
        "relationship_first_policy": {
            "version": relationship_first.version,
            "digest": relationship_first.digest,
            "queue": [
                {
                    "person_id": item.person_id,
                    "score": item.score,
                    "contributions": item.contributions,
                    "candidate_only": item.candidate_only,
                }
                for item in relationship_queue
            ],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("exp97: adaptable attention queue")
    print(
        "hard unresolved follow-up person_a: "
        f"{hard.entailed}"
    )
    print(
        "default ranking: "
        + " > ".join(
            item.person_id for item in default_queue
        )
    )
    print(
        "relationship-first ranking: "
        + " > ".join(
            item.person_id for item in relationship_queue
        )
    )
    print(
        "policy mutated world: "
        f"{result['world_mutated_by_policy']}"
    )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
