"""Replayable external cognitive-process traces.

This is not model chain-of-thought. A cognitive program records only
machine-checkable state transitions between artifacts: observations,
normalized candidates, admitted facts, derivations, scores, comparisons,
proposals, actions, receipts, outcomes, and revisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable


COGNITIVE_OPERATORS = {
    "OBSERVE",
    "NORMALIZE",
    "ADMIT",
    "RETRIEVE",
    "DERIVE",
    "SCORE",
    "COMPARE",
    "PROPOSE",
    "ACT",
    "VERIFY",
    "REVISE",
}


@dataclass(frozen=True)
class CognitiveArtifact:
    artifact_id: str
    kind: str
    payload_digest: str
    source_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CognitiveStep:
    step_id: str
    operator: str
    operator_id: str
    operator_version: str
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    parent_step_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    receipt_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return _digest(
            {
                "step_id": self.step_id,
                "operator": self.operator,
                "operator_id": self.operator_id,
                "operator_version": self.operator_version,
                "inputs": list(self.input_artifact_ids),
                "outputs": list(self.output_artifact_ids),
                "parents": list(self.parent_step_ids),
                "evidence_refs": list(self.evidence_refs),
                "receipt_refs": list(self.receipt_refs),
                "metadata": self.metadata,
            }
        )


class CognitiveProgramTrace:
    """A deterministic artifact-and-step DAG for consequential reasoning."""

    def __init__(self, program_id: str) -> None:
        self.program_id = program_id
        self.artifacts: dict[str, CognitiveArtifact] = {}
        self.steps: dict[str, CognitiveStep] = {}
        self._artifact_payloads: dict[str, Any] = {}

    def add_artifact(
        self,
        artifact_id: str,
        *,
        kind: str,
        payload: Any,
        source_refs: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> CognitiveArtifact:
        if artifact_id in self.artifacts:
            raise ValueError(
                f"duplicate artifact_id: {artifact_id}"
            )
        artifact = CognitiveArtifact(
            artifact_id=artifact_id,
            kind=kind,
            payload_digest=_digest(payload),
            source_refs=tuple(source_refs),
            metadata=dict(metadata or {}),
        )
        self.artifacts[artifact_id] = artifact
        self._artifact_payloads[artifact_id] = payload
        return artifact

    def add_step(
        self,
        step_id: str,
        *,
        operator: str,
        operator_id: str,
        operator_version: str,
        input_artifact_ids: Iterable[str],
        output_artifact_ids: Iterable[str],
        parent_step_ids: Iterable[str] = (),
        evidence_refs: Iterable[str] = (),
        receipt_refs: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> CognitiveStep:
        if step_id in self.steps:
            raise ValueError(f"duplicate step_id: {step_id}")
        if operator not in COGNITIVE_OPERATORS:
            raise ValueError(f"unknown cognitive operator: {operator}")

        inputs = tuple(input_artifact_ids)
        outputs = tuple(output_artifact_ids)
        parents = tuple(parent_step_ids)
        receipts = tuple(receipt_refs)

        for artifact_id in (*inputs, *outputs):
            if artifact_id not in self.artifacts:
                raise ValueError(
                    f"unknown artifact referenced by {step_id}: "
                    f"{artifact_id}"
                )
        for parent_id in parents:
            if parent_id not in self.steps:
                raise ValueError(
                    f"unknown parent step for {step_id}: {parent_id}"
                )

        if operator == "ACT" and not receipts:
            raise ValueError(
                "ACT steps require at least one execution receipt reference"
            )
        if operator == "ADMIT" and not evidence_refs:
            raise ValueError(
                "ADMIT steps require evidence references"
            )

        step = CognitiveStep(
            step_id=step_id,
            operator=operator,
            operator_id=operator_id,
            operator_version=operator_version,
            input_artifact_ids=inputs,
            output_artifact_ids=outputs,
            parent_step_ids=parents,
            evidence_refs=tuple(evidence_refs),
            receipt_refs=receipts,
            metadata=dict(metadata or {}),
        )
        self.steps[step_id] = step
        return step

    def payload(self, artifact_id: str) -> Any:
        return self._artifact_payloads[artifact_id]

    @property
    def process_digest(self) -> str:
        payload = {
            "program_id": self.program_id,
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "payload_digest": artifact.payload_digest,
                    "source_refs": list(artifact.source_refs),
                    "metadata": artifact.metadata,
                }
                for artifact in sorted(
                    self.artifacts.values(),
                    key=lambda item: item.artifact_id,
                )
            ],
            "steps": [
                {
                    "step_id": step.step_id,
                    "digest": step.digest,
                }
                for step in sorted(
                    self.steps.values(),
                    key=lambda item: item.step_id,
                )
            ],
        }
        return _digest(payload)

    def replay_manifest(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "process_digest": self.process_digest,
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "kind": artifact.kind,
                    "payload_digest": artifact.payload_digest,
                    "source_refs": list(artifact.source_refs),
                    "metadata": artifact.metadata,
                }
                for artifact in sorted(
                    self.artifacts.values(),
                    key=lambda item: item.artifact_id,
                )
            ],
            "steps": [
                {
                    "step_id": step.step_id,
                    "operator": step.operator,
                    "operator_id": step.operator_id,
                    "operator_version": step.operator_version,
                    "inputs": list(step.input_artifact_ids),
                    "outputs": list(step.output_artifact_ids),
                    "parents": list(step.parent_step_ids),
                    "evidence_refs": list(step.evidence_refs),
                    "receipt_refs": list(step.receipt_refs),
                    "step_digest": step.digest,
                    "metadata": step.metadata,
                }
                for step in sorted(
                    self.steps.values(),
                    key=lambda item: item.step_id,
                )
            ],
        }


def _digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
