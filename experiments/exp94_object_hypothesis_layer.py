"""Object-hypothesis repair layer for detector structural failures.

Exp93 showed that calibrated coordinate bands handle localization noise, while
missing, merged, and false-positive detector objects require an explicit
identity/cardinality layer. This experiment adds a small upper-bound hypothesis
layer using the simulated detector anomaly metadata from exp93.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp85_support_tl import SupportTolerance, infer_stability
from experiments.exp87_support_eval import EvalConfig, make_splits
from experiments.exp90_support_repair_sweep import repair_objects
from experiments.exp91_interval_support_uncertainty import possible_stability
from experiments.runtime_paths import default_runtime_result_path, portable_path
from experiments.exp92_pixel_abstain_recover import detect_object_table, render_scene
from experiments.exp93_detector_calibration_stress import (
    FAILURE_MODES,
    LOCALIZATION_MODES,
    StressTable,
    _detected_intervention,
    _interval_accepts,
    _seed_for,
    apply_detector_stress,
)


RESULT_DIR = Path(__file__).with_name("exp94_object_hypothesis_layer_data")
STRUCTURAL_FAILURE_MODES = tuple(mode for mode in FAILURE_MODES if mode != "coordinate")


@dataclass(frozen=True)
class ObjectHypothesisConfig:
    quick: bool = False
    seed: int = 9400
    eval_scenes: int = 64
    localization_deltas: tuple[float, ...] = (0.0, 0.01, 0.05)
    uncertainty_multipliers: tuple[float, ...] = (1.0,)
    failure_modes: tuple[str, ...] = STRUCTURAL_FAILURE_MODES
    render_scale: int = 64
    render_padding: float = 0.5
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001

    @classmethod
    def for_mode(cls, quick: bool) -> "ObjectHypothesisConfig":
        if quick:
            return cls(
                quick=True,
                eval_scenes=5,
                localization_deltas=(0.0, 0.01),
                uncertainty_multipliers=(1.0,),
            )
        return cls()

    def __post_init__(self) -> None:
        if self.render_scale <= 0:
            raise ValueError("render_scale must be positive")
        if self.render_padding < 0:
            raise ValueError("render_padding must be non-negative")
        if self.contact_tolerance < 0:
            raise ValueError("contact_tolerance must be non-negative")
        if self.horizontal_tolerance < 0:
            raise ValueError("horizontal_tolerance must be non-negative")
        if any(delta < 0 for delta in self.localization_deltas):
            raise ValueError("localization_deltas must be non-negative")
        if any(multiplier < 0 for multiplier in self.uncertainty_multipliers):
            raise ValueError("uncertainty_multipliers must be non-negative")
        if not set(self.failure_modes) <= set(STRUCTURAL_FAILURE_MODES):
            raise ValueError(f"failure_modes must be drawn from {STRUCTURAL_FAILURE_MODES}")


@dataclass(frozen=True)
class ObjectHypothesis:
    name: str
    objects: list[dict[str, object]]
    repair_kind: str


def _tolerance(config: ObjectHypothesisConfig) -> SupportTolerance:
    return SupportTolerance(contact=config.contact_tolerance, horizontal=config.horizontal_tolerance)


def _clone_objects(objects: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(obj) for obj in objects]


def _original_by_id(scene: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(obj["id"]): dict(obj) for obj in scene["objects"]}


def _merge_source_ids(obj: dict[str, object]) -> set[str]:
    source_ids = obj.get("source_ids", ())
    if not isinstance(source_ids, tuple | list):
        return set()
    return {str(source_id) for source_id in source_ids}


def _missing_restoration(
    scene: dict[str, object],
    table: StressTable,
) -> ObjectHypothesis | None:
    originals = _original_by_id(scene)
    missing_ids = [oid for oid in table.affected_ids if oid in originals]
    if not missing_ids:
        return None
    objects = _clone_objects(table.objects)
    present = {str(obj["id"]) for obj in objects}
    for oid in missing_ids:
        if oid not in present:
            objects.append(dict(originals[oid]))
    return ObjectHypothesis("restore_missing", objects, "missing_restored")


def _merge_split(
    scene: dict[str, object],
    table: StressTable,
) -> ObjectHypothesis | None:
    originals = _original_by_id(scene)
    affected = set(table.affected_ids)
    if not affected:
        return None

    objects: list[dict[str, object]] = []
    removed_merged = False
    for obj in table.objects:
        if _merge_source_ids(obj) & affected:
            removed_merged = True
            continue
        objects.append(dict(obj))

    if not removed_merged:
        return None
    for oid in table.affected_ids:
        if oid in originals:
            objects.append(dict(originals[oid]))
    return ObjectHypothesis("split_merge", objects, "merge_split")


def _false_positive_drop(table: StressTable) -> ObjectHypothesis | None:
    false_positive_ids = set(table.false_positive_ids)
    if not false_positive_ids:
        return None
    objects = [
        dict(obj)
        for obj in table.objects
        if str(obj["id"]) not in false_positive_ids
    ]
    return ObjectHypothesis("drop_false_positive", objects, "false_positive_dropped")


def generate_object_hypotheses(
    scene: dict[str, object],
    table: StressTable,
) -> list[ObjectHypothesis]:
    """Return observed and structural-repair object-table candidates."""

    hypotheses = [ObjectHypothesis("observed", _clone_objects(table.objects), "none")]
    repaired: ObjectHypothesis | None = None
    if table.failure_mode == "missing":
        repaired = _missing_restoration(scene, table)
    elif table.failure_mode == "merge":
        repaired = _merge_split(scene, table)
    elif table.failure_mode == "false_positive":
        repaired = _false_positive_drop(table)

    if repaired is not None:
        hypotheses.append(repaired)
    return hypotheses


def _candidate_labels(
    scene: dict[str, object],
    hypothesis: ObjectHypothesis,
    table: StressTable,
    config: ObjectHypothesisConfig,
) -> dict[str, dict[str, object]]:
    tolerance = _tolerance(config)
    intervention = _detected_intervention(scene["intervention"], hypothesis.objects)
    hard = infer_stability(hypothesis.objects, intervention, tolerance=tolerance)
    possible = possible_stability(
        hypothesis.objects,
        intervention,
        x_radius=table.x_radius,
        y_radius=table.y_radius,
        tolerance=tolerance,
    )
    interval_accept = _interval_accepts(hard.labels, possible)
    repaired = repair_objects(hypothesis.objects, intervention)
    repaired_labels = infer_stability(repaired.objects, intervention, tolerance=tolerance).labels

    labels = scene["labels"]
    assert isinstance(labels, dict)
    records: dict[str, dict[str, object]] = {}
    for oid in labels:
        object_id = str(oid)
        detected = object_id in hard.labels
        hard_label = hard.labels.get(object_id, "missing")
        repaired_label = repaired_labels.get(object_id, "missing")
        accept = detected and interval_accept[object_id]
        label = hard_label
        stage = "hard" if detected else "missing"

        if (
            detected
            and not accept
            and hard_label == "falls"
            and possible.get(object_id, False)
            and repaired_label == "stable"
        ):
            accept = True
            label = repaired_label
            stage = "repair"

        records[object_id] = {
            "accept": accept,
            "label": label,
            "hard": hard_label,
            "stage": stage,
            "possible_stable": possible.get(object_id, False),
            "repair_actions": tuple(action.kind for action in repaired.actions),
        }
    return records


def _pick_structural_candidate(
    candidates: dict[str, dict[str, dict[str, object]]],
) -> tuple[str, dict[str, dict[str, object]]]:
    for name, records in candidates.items():
        if name != "observed":
            return name, records
    return "observed", candidates["observed"]


def evaluate_object_hypothesis_scene(
    scene: dict[str, object],
    table: StressTable,
    config: ObjectHypothesisConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    hypotheses = generate_object_hypotheses(scene, table)
    candidates = {
        hypothesis.name: _candidate_labels(scene, hypothesis, table, config)
        for hypothesis in hypotheses
    }
    selected_name, selected = _pick_structural_candidate(candidates)
    observed = candidates["observed"]

    labels = scene["labels"]
    assert isinstance(labels, dict)
    records: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()
    for oid, expected in labels.items():
        object_id = str(oid)
        observed_record = observed[object_id]
        selected_record = selected[object_id]
        action_counts.update(str(action) for action in selected_record["repair_actions"])
        guarded_accept = bool(observed_record["accept"]) and not table.structural_failure

        records.append(
            {
                "object_id": object_id,
                "expected": expected,
                "source_status": table.source_status.get(object_id, "missing_detector"),
                "failure_mode": table.failure_mode,
                "structural_failure": table.structural_failure,
                "candidate_count": len(hypotheses),
                "selected_hypothesis": selected_name,
                "observed_accept": bool(observed_record["accept"]),
                "observed_label": observed_record["label"],
                "observed_stage": observed_record["stage"],
                "guarded_accept": guarded_accept,
                "guarded_label": observed_record["label"],
                "hypothesis_accept": bool(selected_record["accept"]),
                "hypothesis_label": selected_record["label"],
                "hypothesis_stage": selected_record["stage"],
                "hypothesis_hard": selected_record["hard"],
            }
        )

    return records, action_counts


def _metric(records: Sequence[dict[str, object]], label_key: str) -> dict[str, float | int]:
    correct = sum(1 for record in records if record[label_key] == record["expected"])
    total = len(records)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _accepted_metric(
    records: Sequence[dict[str, object]],
    accept_key: str,
    label_key: str,
) -> dict[str, float | int]:
    accepted = [record for record in records if bool(record[accept_key])]
    metric = _metric(accepted, label_key)
    return {
        **metric,
        "coverage": len(accepted) / len(records) if records else 0.0,
    }


def _accepted_wrong(records: Sequence[dict[str, object]], accept_key: str, label_key: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if bool(record[accept_key]) and record[label_key] != record["expected"]
    ]


def _false_stable(records: Sequence[dict[str, object]], accept_key: str, label_key: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if bool(record[accept_key])
        and record["expected"] == "falls"
        and record[label_key] == "stable"
    ]


def _summarize_records(
    records: Sequence[dict[str, object]],
    action_counts: Counter[str],
    uncertainty_multiplier: float,
) -> dict[str, object]:
    observed_wrong = _accepted_wrong(records, "observed_accept", "observed_label")
    guarded_wrong = _accepted_wrong(records, "guarded_accept", "guarded_label")
    hypothesis_wrong = _accepted_wrong(records, "hypothesis_accept", "hypothesis_label")
    recovered = [
        record
        for record in records
        if bool(record["structural_failure"])
        and not bool(record["guarded_accept"])
        and bool(record["hypothesis_accept"])
        and record["hypothesis_label"] == record["expected"]
    ]

    return {
        "uncertainty_multiplier": uncertainty_multiplier,
        "observed_naive": _accepted_metric(records, "observed_accept", "observed_label"),
        "guarded_structural_abstain": _accepted_metric(records, "guarded_accept", "guarded_label"),
        "object_hypothesis": _accepted_metric(records, "hypothesis_accept", "hypothesis_label"),
        "triage": {
            "source_status": dict(sorted(Counter(str(record["source_status"]) for record in records).items())),
            "selected_hypotheses": dict(sorted(Counter(str(record["selected_hypothesis"]) for record in records).items())),
            "observed_accepted_wrong": len(observed_wrong),
            "observed_false_stable": len(_false_stable(records, "observed_accept", "observed_label")),
            "guarded_accepted_wrong": len(guarded_wrong),
            "guarded_false_stable": len(_false_stable(records, "guarded_accept", "guarded_label")),
            "hypothesis_accepted_wrong": len(hypothesis_wrong),
            "hypothesis_false_stable": len(_false_stable(records, "hypothesis_accept", "hypothesis_label")),
            "hypothesis_recovered_structural": len(recovered),
            "guarded_abstained_recoverable": len(recovered),
        },
        "repair_actions": dict(sorted(action_counts.items())),
    }


def _records_for_multiplier(
    scenes: Sequence[dict[str, object]],
    delta: float,
    axes: Sequence[str],
    failure_mode: str,
    uncertainty_multiplier: float,
    seed: int,
    split: str,
    mode_name: str,
    config: ObjectHypothesisConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    records: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()

    for index, scene in enumerate(scenes):
        rendered = render_scene(
            scene["objects"],
            scale=config.render_scale,
            padding=config.render_padding,
        )
        detected = detect_object_table(
            rendered,
            scene["objects"],
            localization_delta=delta,
            uncertainty_multiplier=uncertainty_multiplier,
            axes=axes,
            seed=_seed_for(seed, f"detector:{mode_name}", split, index, delta),
        )
        stressed = apply_detector_stress(
            scene,
            detected,
            failure_mode,
            seed=_seed_for(seed, f"stress:{failure_mode}:{mode_name}", split, index, delta),
        )
        scene_records, scene_actions = evaluate_object_hypothesis_scene(scene, stressed, config)
        records.extend(scene_records)
        action_counts.update(scene_actions)

    return records, action_counts


def _stress_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: ObjectHypothesisConfig,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, object]]] = {}
    aggregate_rows: list[dict[str, object]] = []

    for failure_mode in config.failure_modes:
        for delta in config.localization_deltas:
            aggregate_by_multiplier: dict[float, list[dict[str, object]]] = {
                multiplier: []
                for multiplier in config.uncertainty_multipliers
            }
            aggregate_actions: dict[float, Counter[str]] = {
                multiplier: Counter()
                for multiplier in config.uncertainty_multipliers
            }
            split_rows: dict[str, list[dict[str, object]]] = {}

            for split, scenes in eval_splits.items():
                split_multiplier_rows: list[dict[str, object]] = []
                for multiplier in config.uncertainty_multipliers:
                    records, action_counts = _records_for_multiplier(
                        scenes,
                        delta,
                        axes,
                        failure_mode,
                        multiplier,
                        config.seed,
                        split,
                        mode_name,
                        config,
                    )
                    aggregate_by_multiplier[multiplier].extend(records)
                    aggregate_actions[multiplier].update(action_counts)
                    split_multiplier_rows.append(_summarize_records(records, action_counts, multiplier))
                split_rows[split] = split_multiplier_rows

            for split, rows in split_rows.items():
                by_split.setdefault(split, []).append(
                    {
                        "failure_mode": failure_mode,
                        "delta": delta,
                        "uncertainty": rows,
                    }
                )

            aggregate_rows.append(
                {
                    "failure_mode": failure_mode,
                    "delta": delta,
                    "uncertainty": [
                        _summarize_records(
                            aggregate_by_multiplier[multiplier],
                            aggregate_actions[multiplier],
                            multiplier,
                        )
                        for multiplier in config.uncertainty_multipliers
                    ],
                }
            )

    return {
        "axes": list(axes),
        "by_split": by_split,
        "all": aggregate_rows,
    }


def _eval_splits(config: ObjectHypothesisConfig) -> dict[str, list[dict[str, object]]]:
    base = EvalConfig.for_mode(config.quick)
    eval_config = EvalConfig(
        quick=config.quick,
        seed=config.seed,
        train_scenes=base.train_scenes,
        eval_scenes=config.eval_scenes,
        epochs=base.epochs,
        hidden_dim=base.hidden_dim,
        lr=base.lr,
    )
    _, eval_splits = make_splits(eval_config)
    return eval_splits


def run_object_hypothesis_layer(
    config: ObjectHypothesisConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _stress_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in LOCALIZATION_MODES.items()
    }

    results = {
        "experiment": "exp94_object_hypothesis_layer",
        "quick": config.quick,
        "config": asdict(config),
        "splits": {
            split: {
                "scenes": len(scenes),
                "objects": sum(len(scene["objects"]) for scene in scenes),
                "interventions": sorted({str(scene["intervention"]["type"]) for scene in scenes}),
            }
            for split, scenes in eval_splits.items()
        },
        "object_hypothesis_layer": {
            "localization_deltas": list(config.localization_deltas),
            "uncertainty_multipliers": list(config.uncertainty_multipliers),
            "failure_modes": list(config.failure_modes),
            "modes": modes,
        },
        "interpretation": {
            "observed_naive": "runs interval+repair on the detector table exactly as emitted",
            "guarded_structural_abstain": "matches exp93's safe route: abstain structural detector failures",
            "object_hypothesis": (
                "uses simulated structural anomaly metadata to select a corrected object-table "
                "candidate before running the same TL interval+repair route"
            ),
        },
    }

    if output_path is None:
        output_path = default_runtime_result_path(RESULT_DIR, quick=config.quick)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["results_path"] = portable_path(output_path)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run object-hypothesis detector repair benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_object_hypothesis_layer(ObjectHypothesisConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
