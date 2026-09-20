from tensor_logic.personal_schema_v2 import (
    AXES,
    TENSORS,
    build_personal_semantic_schema,
)


def test_semantic_schema_covers_core_life_domains():
    world = build_personal_semantic_schema(
        {
            "Person": ("me", "person_a"),
            "Event": ("event_a",),
            "Place": ("home", "office"),
            "TimeBucket": ("2026-09-20T09:00",),
            "Mode": ("drive",),
            "Project": ("project_x",),
            "Goal": ("goal_x",),
            "Task": ("task_x",),
            "Commitment": ("commitment_x",),
            "Message": ("message_x",),
            "Document": ("doc_x",),
            "Note": ("note_x",),
            "Decision": ("decision_x",),
            "Policy": ("attention-v1",),
            "Signal": ("importance", "staleness"),
            "Tool": ("maps",),
            "ToolCall": ("call_x",),
            "Evidence": ("evidence_x",),
            "Claim": ("claim_x",),
            "Action": ("action_x",),
            "Receipt": ("receipt_x",),
            "Provider": ("google_calendar",),
            "Source": ("gmail",),
            "Account": ("default",),
        }
    )

    assert set(world.axes) == set(AXES)
    assert set(world.tensors) == set(TENSORS)

    assert world.tensors["travel_minutes"].shape == (2, 2, 1, 1)
    assert world.tensors["attention_signal"].shape == (2, 2)
    assert world.tensors["task_requires"].shape == (1, 1)


def test_semantic_schema_supports_hard_and_soft_values():
    world = build_personal_semantic_schema(
        {
            "Person": ("me",),
            "Event": ("event_a",),
            "Goal": ("goal_x",),
            "Signal": ("importance",),
            "Policy": ("p1",),
        }
    )

    world.tensors["attends"].set(("me", "event_a"), 1.0)
    world.tensors["goal_priority"].set(("goal_x",), 0.9)
    world.tensors["attention_signal"].set(
        ("me", "importance"),
        0.7,
    )
    world.tensors["policy_weight"].set(
        ("p1", "importance"),
        1.0,
    )

    assert world.tensors["attends"].get(("me", "event_a")) == 1.0
    assert world.tensors["goal_priority"].get(("goal_x",)) == 0.9
    assert world.tensors["attention_signal"].get(
        ("me", "importance")
    ) == 0.7
