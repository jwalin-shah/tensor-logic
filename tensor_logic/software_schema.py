"""Semantic tensor schema for self-healing software systems.

The goal is to represent software execution and repair as a persistent world
model: static structure, dynamic traces, failures, hypotheses, patches, tests,
deployments, and receipts can all be queried and composed.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .world_tensor import TensorWorld


SOFTWARE_AXES = (
    "Repository",
    "Commit",
    "File",
    "Symbol",
    "Service",
    "Endpoint",
    "Dependency",
    "Test",
    "Build",
    "Deployment",
    "Incident",
    "Error",
    "Trace",
    "TraceFrame",
    "Invariant",
    "Metric",
    "Patch",
    "RepairHypothesis",
    "AgentRun",
    "ToolCall",
    "Receipt",
    "TimeBucket",
)


SOFTWARE_TENSORS = {
    # Static program / system structure.
    "repo_commit": (("Repository", "Commit"), "boolean"),
    "commit_file": (("Commit", "File"), "boolean"),
    "file_symbol": (("File", "Symbol"), "boolean"),
    "symbol_calls": (("Symbol", "Symbol"), "boolean"),
    "service_symbol": (("Service", "Symbol"), "boolean"),
    "service_endpoint": (("Service", "Endpoint"), "boolean"),
    "service_depends_on": (("Service", "Service"), "boolean"),
    "test_covers_symbol": (("Test", "Symbol"), "boolean"),
    "test_covers_endpoint": (("Test", "Endpoint"), "boolean"),
    "invariant_symbol": (("Invariant", "Symbol"), "boolean"),
    # Dynamic execution.
    "trace_incident": (("Trace", "Incident"), "boolean"),
    "trace_frame": (("Trace", "TraceFrame"), "boolean"),
    "frame_symbol": (("TraceFrame", "Symbol"), "boolean"),
    "frame_parent": (("TraceFrame", "TraceFrame"), "boolean"),
    "frame_time": (("TraceFrame", "TimeBucket"), "boolean"),
    "error_frame": (("Error", "TraceFrame"), "boolean"),
    "incident_error": (("Incident", "Error"), "boolean"),
    "incident_service": (("Incident", "Service"), "boolean"),
    "metric_service": (("Metric", "Service"), "boolean"),
    "metric_anomaly_score": (("Metric",), "real"),
    "invariant_violated": (("Invariant",), "boolean"),
    # Build / deploy history.
    "build_commit": (("Build", "Commit"), "boolean"),
    "build_test": (("Build", "Test"), "boolean"),
    "test_passed": (("Test",), "boolean"),
    "deployment_build": (("Deployment", "Build"), "boolean"),
    "deployment_service": (("Deployment", "Service"), "boolean"),
    "deployment_time": (("Deployment", "TimeBucket"), "boolean"),
    # Repair hypotheses and patches.
    "hypothesis_incident": (
        ("RepairHypothesis", "Incident"),
        "boolean",
    ),
    "hypothesis_symbol": (
        ("RepairHypothesis", "Symbol"),
        "boolean",
    ),
    "hypothesis_confidence": (("RepairHypothesis",), "real"),
    "patch_hypothesis": (("Patch", "RepairHypothesis"), "boolean"),
    "patch_commit": (("Patch", "Commit"), "boolean"),
    "patch_test": (("Patch", "Test"), "boolean"),
    "patch_validation_score": (("Patch",), "real"),
    # Agent/tool/receipt provenance.
    "agent_patch": (("AgentRun", "Patch"), "boolean"),
    "agent_toolcall": (("AgentRun", "ToolCall"), "boolean"),
    "toolcall_receipt": (("ToolCall", "Receipt"), "boolean"),
    "patch_receipt": (("Patch", "Receipt"), "boolean"),
}


def build_software_tensor_schema(
    symbols: Mapping[str, Iterable[str]] | None = None,
) -> TensorWorld:
    supplied = dict(symbols or {})
    world = TensorWorld()

    for axis_name in SOFTWARE_AXES:
        world.add_axis(
            axis_name,
            axis_name,
            supplied.get(axis_name, ()),
        )

    for tensor_name, (
        axis_names,
        value_kind,
    ) in SOFTWARE_TENSORS.items():
        world.add_tensor(
            tensor_name,
            axis_names,
            value_kind=value_kind,
        )

    return world
