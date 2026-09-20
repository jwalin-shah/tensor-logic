"""Canonical semantic tensor schema for the Personal Physics life model.

The generic projection in universal_context.py guarantees coverage for arbitrary
fields. This module supplies stable semantic axes and tensors for high-value
life domains so queries and rules do not depend on raw JSON property paths.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .world_tensor import TensorWorld


AXES = (
    "Person",
    "Interaction",
    "Conversation",
    "Event",
    "TimeBucket",
    "Place",
    "Mode",
    "Project",
    "Goal",
    "Task",
    "Commitment",
    "Message",
    "Document",
    "Note",
    "Decision",
    "Policy",
    "Signal",
    "Tool",
    "ToolCall",
    "Evidence",
    "Claim",
    "Action",
    "Receipt",
    "Provider",
    "Source",
    "Account",
)


TENSORS = {
    # People and interactions.
    "interaction_person": (("Interaction", "Person"), "boolean"),
    "interaction_time": (("Interaction", "TimeBucket"), "boolean"),
    "interaction_source": (("Interaction", "Source"), "boolean"),
    "relationship_importance": (("Person",), "real"),
    "last_interaction_age_days": (("Person",), "real"),
    # Calendar and place.
    "attends": (("Person", "Event"), "boolean"),
    "located_at": (("Event", "Place"), "boolean"),
    "starts_at": (("Event", "TimeBucket"), "boolean"),
    "ends_at": (("Event", "TimeBucket"), "boolean"),
    "event_project": (("Event", "Project"), "boolean"),
    "event_optional": (("Event",), "boolean"),
    # Travel.
    "travel_minutes": (
        ("Place", "Place", "TimeBucket", "Mode"),
        "real",
    ),
    "distance_km": (("Place", "Place", "Mode"), "real"),
    # Work, goals, tasks, commitments.
    "works_on": (("Person", "Project"), "boolean"),
    "project_supports_goal": (("Project", "Goal"), "boolean"),
    "goal_priority": (("Goal",), "real"),
    "task_project": (("Task", "Project"), "boolean"),
    "task_requires": (("Task", "Task"), "boolean"),
    "task_due": (("Task", "TimeBucket"), "boolean"),
    "task_completed": (("Task",), "boolean"),
    "commitment_owner": (("Commitment", "Person"), "boolean"),
    "commitment_target_person": (
        ("Commitment", "Person"),
        "boolean",
    ),
    "commitment_due": (("Commitment", "TimeBucket"), "boolean"),
    "commitment_completed": (("Commitment",), "boolean"),
    # Messages, conversations, documents, notes.
    "message_sender": (("Message", "Person"), "boolean"),
    "message_recipient": (("Message", "Person"), "boolean"),
    "message_conversation": (
        ("Message", "Conversation"),
        "boolean",
    ),
    "message_time": (("Message", "TimeBucket"), "boolean"),
    "message_source": (("Message", "Source"), "boolean"),
    "document_project": (("Document", "Project"), "boolean"),
    "note_project": (("Note", "Project"), "boolean"),
    # Decisions, policies, adaptable scoring.
    "decision_goal": (("Decision", "Goal"), "boolean"),
    "decision_time": (("Decision", "TimeBucket"), "boolean"),
    "attention_signal": (("Person", "Signal"), "real"),
    "policy_weight": (("Policy", "Signal"), "real"),
    # Tools, evidence, claims, actions, receipts.
    "invoked": (("ToolCall", "Tool"), "boolean"),
    "toolcall_account": (("ToolCall", "Account"), "boolean"),
    "produced_evidence": (("ToolCall", "Evidence"), "boolean"),
    "supports_claim": (("Evidence", "Claim"), "boolean"),
    "contradicts_claim": (("Evidence", "Claim"), "boolean"),
    "action_toolcall": (("Action", "ToolCall"), "boolean"),
    "action_receipt": (("Action", "Receipt"), "boolean"),
    "receipt_toolcall": (("Receipt", "ToolCall"), "boolean"),
    # Provider/source health constrains query plans.
    "provider_readable": (("Provider",), "boolean"),
    "provider_syncable": (("Provider",), "boolean"),
    "provider_writable": (("Provider",), "boolean"),
}


def build_personal_semantic_schema(
    symbols: Mapping[str, Iterable[str]] | None = None,
) -> TensorWorld:
    """Build the canonical life tensor schema with caller-provided symbols."""
    supplied = dict(symbols or {})
    world = TensorWorld()
    for axis_name in AXES:
        world.add_axis(
            axis_name,
            axis_name,
            supplied.get(axis_name, ()),
        )

    for tensor_name, (
        axis_names,
        value_kind,
    ) in TENSORS.items():
        world.add_tensor(
            tensor_name,
            axis_names,
            value_kind=value_kind,
        )
    return world
