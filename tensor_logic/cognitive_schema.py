"""Tensor schema for a bounded cognitive architecture.

These tensors represent externally inspectable cognitive state, not hidden
chain-of-thought. They are designed to support working memory, episodic memory,
procedural memory, prediction/error, metacognition, planning, and language/action
interfaces on top of the Personal Physics world model.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .world_tensor import TensorWorld


COGNITIVE_AXES = (
    "Agent",
    "TaskContext",
    "MemoryItem",
    "Episode",
    "CognitiveState",
    "Operator",
    "Skill",
    "Goal",
    "Subgoal",
    "Prediction",
    "Outcome",
    "Hypothesis",
    "Claim",
    "Evidence",
    "Resource",
    "ErrorSignal",
    "Plan",
    "PlanStep",
    "Policy",
    "ModelVersion",
    "Utterance",
    "GenerationStep",
    "Token",
    "TimeBucket",
)


COGNITIVE_TENSORS = {
    # Working memory / global workspace.
    "working_memory": (
        ("TaskContext", "MemoryItem"),
        "boolean",
    ),
    "memory_activation": (
        ("TaskContext", "MemoryItem"),
        "real",
    ),
    "memory_relevance": (
        ("TaskContext", "MemoryItem"),
        "real",
    ),
    "workspace_broadcast": (
        ("TaskContext", "MemoryItem"),
        "boolean",
    ),
    "workspace_competition_score": (
        ("TaskContext", "MemoryItem"),
        "real",
    ),
    # Goals / task control.
    "active_goal": (("Agent", "Goal"), "boolean"),
    "goal_priority": (("Goal",), "real"),
    "goal_subgoal": (("Goal", "Subgoal"), "boolean"),
    "subgoal_satisfied": (("Subgoal",), "boolean"),
    "task_goal": (("TaskContext", "Goal"), "boolean"),
    # Episodic memory.
    "episode_state": (
        ("Episode", "CognitiveState"),
        "boolean",
    ),
    "episode_operator": (
        ("Episode", "Operator"),
        "boolean",
    ),
    "episode_outcome": (
        ("Episode", "Outcome"),
        "boolean",
    ),
    "episode_time": (
        ("Episode", "TimeBucket"),
        "boolean",
    ),
    "episode_reward": (("Episode",), "real"),
    "episode_confidence": (("Episode",), "real"),
    # Procedural / skill memory.
    "operator_precondition": (
        ("Operator", "Claim"),
        "boolean",
    ),
    "operator_effect": (
        ("Operator", "Claim"),
        "boolean",
    ),
    "operator_skill": (
        ("Operator", "Skill"),
        "boolean",
    ),
    "operator_success_rate": (("Operator",), "real"),
    "operator_value": (
        ("TaskContext", "Operator"),
        "real",
    ),
    "operator_confidence": (
        ("TaskContext", "Operator"),
        "real",
    ),
    "skill_activation": (
        ("TaskContext", "Skill"),
        "real",
    ),
    # Prediction / predictive processing.
    "prediction_claim": (
        ("Prediction", "Claim"),
        "boolean",
    ),
    "prediction_probability": (("Prediction",), "real"),
    "prediction_model": (
        ("Prediction", "ModelVersion"),
        "boolean",
    ),
    "prediction_error": (
        ("Prediction", "ErrorSignal"),
        "real",
    ),
    "error_magnitude": (("ErrorSignal",), "real"),
    # Hypotheses / learned candidate structure.
    "hypothesis_claim": (
        ("Hypothesis", "Claim"),
        "boolean",
    ),
    "hypothesis_support": (
        ("Hypothesis", "Evidence"),
        "real",
    ),
    "hypothesis_confidence": (("Hypothesis",), "real"),
    "hypothesis_model": (
        ("Hypothesis", "ModelVersion"),
        "boolean",
    ),
    # Metacognition / control.
    "task_confidence": (("TaskContext",), "real"),
    "task_uncertainty": (("TaskContext",), "real"),
    "contradiction_signal": (("TaskContext",), "real"),
    "escalation_score": (("TaskContext",), "real"),
    "resource_budget": (
        ("TaskContext", "Resource"),
        "real",
    ),
    "operator_resource_cost": (
        ("Operator", "Resource"),
        "real",
    ),
    # Planning / counterfactual simulation.
    "plan_task": (("Plan", "TaskContext"), "boolean"),
    "plan_step": (("Plan", "PlanStep"), "boolean"),
    "plan_step_operator": (
        ("PlanStep", "Operator"),
        "boolean",
    ),
    "plan_step_state": (
        ("PlanStep", "CognitiveState"),
        "boolean",
    ),
    "plan_step_parent": (
        ("PlanStep", "PlanStep"),
        "boolean",
    ),
    "plan_step_value": (("PlanStep",), "real"),
    "plan_step_probability": (("PlanStep",), "real"),
    "plan_value": (("Plan",), "real"),
    # Policies.
    "policy_operator_weight": (
        ("Policy", "Operator"),
        "real",
    ),
    "policy_skill_weight": (
        ("Policy", "Skill"),
        "real",
    ),
    # Language / communication is an output interface, not core truth.
    "utterance_claim": (
        ("Utterance", "Claim"),
        "boolean",
    ),
    "utterance_confidence": (("Utterance",), "real"),
    "utterance_candidate": (("Utterance",), "boolean"),
    "generation_token_probability": (
        ("GenerationStep", "Token"),
        "real",
    ),
    "generation_step_utterance": (
        ("GenerationStep", "Utterance"),
        "boolean",
    ),
}


def build_cognitive_tensor_schema(
    symbols: Mapping[str, Iterable[str]] | None = None,
) -> TensorWorld:
    supplied = dict(symbols or {})
    world = TensorWorld()

    for axis_name in COGNITIVE_AXES:
        world.add_axis(
            axis_name,
            axis_name,
            supplied.get(axis_name, ()),
        )

    for tensor_name, (
        axis_names,
        value_kind,
    ) in COGNITIVE_TENSORS.items():
        world.add_tensor(
            tensor_name,
            axis_names,
            value_kind=value_kind,
        )

    return world
