from tensor_logic.personal_physics import FactRecord, build_personal_physics_v0
from tensor_logic.world_tensor import (
    CoordinateProvenance,
    build_world_tensor_schema,
    tensorize_personal_world,
)


def _admitted(fact_id, relation, subject, object_):
    return FactRecord(
        fact_id=fact_id,
        relation=relation,
        subject=subject,
        object=object_,
        status="admitted",
        evidence_refs=(f"evidence:{fact_id}",),
        source_kind="fixture",
        source_ref=f"source:{fact_id}",
        admission_ref="admission:test",
    )


def test_four_axis_travel_tensor_is_sparse_real_and_provenanced():
    world = build_world_tensor_schema(
        places=("home", "office"),
        time_buckets=("2026-09-20T09:00",),
        modes=("drive", "transit"),
    )
    travel = world.tensors["travel_minutes"]
    coordinate = (
        "home",
        "office",
        "2026-09-20T09:00",
        "drive",
    )
    travel.set(
        coordinate,
        37.0,
        provenance=CoordinateProvenance(
            evidence_refs=("route:call-17",),
            source_refs=("maps",),
            admission_ref="admission:route",
        ),
    )

    assert travel.shape == (2, 2, 1, 2)
    assert travel.get(coordinate) == 37.0
    assert travel.dense().sum().item() == 37.0
    assert travel.sparse()._nnz() == 1
    assert travel.provenance(coordinate).evidence_refs == (
        "route:call-17",
    )


def test_tensorize_personal_world_includes_only_admitted_facts():
    world = build_personal_physics_v0()
    world.register_entity("me", "Person")
    world.register_entity("person_a", "Person")

    world.add_fact(
        _admitted(
            "f-open",
            "requested_followup",
            "me",
            "person_a",
        )
    )
    world.add_fact(
        FactRecord(
            fact_id="f-candidate",
            relation="not_completed_followup",
            subject="me",
            object="person_a",
            status="candidate",
            evidence_refs=("llm:candidate",),
            source_kind="llm",
            source_ref="model:x",
            confidence=0.99,
        )
    )

    tensor_world = tensorize_personal_world(world)

    assert tensor_world.tensors["requested_followup"].get(
        ("me", "person_a")
    ) == 1.0
    assert tensor_world.tensors["not_completed_followup"].get(
        ("me", "person_a")
    ) == 0.0
    provenance = tensor_world.tensors[
        "requested_followup"
    ].provenance(("me", "person_a"))
    assert provenance.metadata["fact_id"] == "f-open"
    assert provenance.admission_ref == "admission:test"


def test_world_tensor_digest_is_reproducible():
    def build():
        world = build_world_tensor_schema(
            people=("me",),
            goals=("goal_x",),
        )
        world.tensors["goal_priority"].set(
            ("goal_x",),
            0.9,
            provenance=CoordinateProvenance(
                evidence_refs=("user:goal",),
                admission_ref="admission:user",
            ),
        )
        return world

    assert build().digest == build().digest
