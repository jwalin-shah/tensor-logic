"""Scored non-oracle object hypotheses for detector structural failures.

Exp94 showed an upper bound: if the detector hands us true structural anomaly
metadata, an object-hypothesis layer can recover missing, merged, and
false-positive failures. This experiment removes that oracle selection step.
It generates plausible object-table candidates from the observed stressed table
and ranks them with deterministic geometry/TL consistency signals.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
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
from experiments.exp92_pixel_abstain_recover import detect_object_table, render_scene
from experiments.exp93_detector_calibration_stress import (
    FAILURE_MODES,
    LOCALIZATION_MODES,
    StressTable,
    _detected_intervention,
    _interval_accepts,
    _removed_id,
    _seed_for,
    apply_detector_stress,
)
from experiments.exp94_object_hypothesis_layer import (
    ObjectHypothesisConfig,
    evaluate_object_hypothesis_scene,
)
from experiments.runtime_paths import portable_path, result_path


RESULT_DIR = Path(__file__).with_name("exp95_scored_object_hypotheses_data")
EXPERIMENT_NAME = "exp95_scored_object_hypotheses"
STRUCTURAL_FAILURE_MODES = tuple(mode for mode in FAILURE_MODES if mode != "coordinate")


@dataclass(frozen=True)
class ScoredHypothesisConfig:
    quick: bool = False
    seed: int = 9500
    eval_scenes: int = 64
    localization_deltas: tuple[float, ...] = (0.0, 0.01, 0.05)
    uncertainty_multipliers: tuple[float, ...] = (1.0,)
    failure_modes: tuple[str, ...] = STRUCTURAL_FAILURE_MODES
    render_scale: int = 64
    render_padding: float = 0.5
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001
    max_candidates_per_kind: int = 6

    @classmethod
    def for_mode(cls, quick: bool) -> "ScoredHypothesisConfig":
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
        if self.max_candidates_per_kind <= 0:
            raise ValueError("max_candidates_per_kind must be positive")
        if any(delta < 0 for delta in self.localization_deltas):
            raise ValueError("localization_deltas must be non-negative")
        if any(multiplier < 0 for multiplier in self.uncertainty_multipliers):
            raise ValueError("uncertainty_multipliers must be non-negative")
        if not set(self.failure_modes) <= set(STRUCTURAL_FAILURE_MODES):
            raise ValueError(f"failure_modes must be drawn from {STRUCTURAL_FAILURE_MODES}")


@dataclass(frozen=True)
class NonOracleHypothesis:
    name: str
    objects: list[dict[str, object]]
    repair_kind: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedHypothesis:
    hypothesis: NonOracleHypothesis
    score: float
    score_parts: dict[str, float | int | str]


def _tolerance(config: ScoredHypothesisConfig) -> SupportTolerance:
    return SupportTolerance(contact=config.contact_tolerance, horizontal=config.horizontal_tolerance)


def _clone_objects(objects: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(obj) for obj in objects]


def _safe_id(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def _effective_objects(
    intervention: dict[str, str],
    objects: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    removed = _removed_id(intervention)
    if removed is None:
        return _clone_objects(objects)
    return [dict(obj) for obj in objects if str(obj["id"]) != removed]


def _base_height(objects: Sequence[dict[str, object]]) -> float:
    heights = sorted(float(obj["h"]) for obj in objects if float(obj["h"]) > 0)
    if not heights:
        return 1.0
    return heights[len(heights) // 2]


def _add_missing_support_candidates(
    table: StressTable,
    intervention: dict[str, str],
    config: ScoredHypothesisConfig,
) -> list[NonOracleHypothesis]:
    removed = _removed_id(intervention)
    candidates: list[NonOracleHypothesis] = []
    effective = _effective_objects(intervention, table.objects)
    observed_intervention = _detected_intervention(intervention, table.objects)
    observed_labels = infer_stability(
        table.objects,
        observed_intervention,
        tolerance=_tolerance(config),
    ).labels
    targets = [
        obj
        for obj in effective
        if str(obj["id"]) != removed
        and observed_labels.get(str(obj["id"])) == "falls"
        and float(obj["y"]) > config.contact_tolerance
    ]
    targets = sorted(targets, key=lambda obj: (float(obj["y"]), float(obj["x"])))

    for target in targets[:1]:
        target_id = _safe_id(target["id"])
        support_height = max(float(target["y"]), _base_height(effective) * 0.5)
        support = {
            "id": f"hyp_missing_support_{target_id}",
            "x": float(target["x"]),
            "y": 0.0,
            "w": float(target["w"]),
            "h": support_height,
            "hypothesis_kind": "missing_support",
        }
        objects = _clone_objects(table.objects)
        objects.append(support)
        candidates.append(
            NonOracleHypothesis(
                name=f"add_missing_support_under_{target_id}",
                objects=objects,
                repair_kind="missing_support_added",
                metadata={"added_supports": 1, "target_y": float(target["y"])},
            )
        )
    return candidates


def _add_merge_split_candidates(
    table: StressTable,
    config: ScoredHypothesisConfig,
) -> list[NonOracleHypothesis]:
    candidates: list[NonOracleHypothesis] = []
    base_h = _base_height(table.objects)
    objects_by_height = sorted(table.objects, key=lambda obj: float(obj["h"]), reverse=True)

    for obj in objects_by_height[: config.max_candidates_per_kind]:
        height = float(obj["h"])
        split_threshold = base_h * 1.5 if len(table.objects) > 1 else base_h * 0.75
        if height <= max(split_threshold, config.contact_tolerance):
            continue
        obj_id = _safe_id(obj["id"])
        lower_h = height / 2
        upper_h = height - lower_h
        lower = {
            "id": f"hyp_split_{obj_id}_lower",
            "x": float(obj["x"]),
            "y": float(obj["y"]),
            "w": float(obj["w"]),
            "h": lower_h,
            "hypothesis_kind": "merge_split",
        }
        upper = {
            "id": f"hyp_split_{obj_id}_upper",
            "x": float(obj["x"]),
            "y": float(obj["y"]) + lower_h,
            "w": float(obj["w"]),
            "h": upper_h,
            "hypothesis_kind": "merge_split",
        }
        objects = [dict(candidate) for candidate in table.objects if str(candidate["id"]) != str(obj["id"])]
        objects.extend([lower, upper])
        candidates.append(
            NonOracleHypothesis(
                name=f"split_compound_box_{obj_id}",
                objects=objects,
                repair_kind="merge_split_candidate",
                metadata={"split_count": 1, "compound_height": height, "base_height": base_h},
            )
        )
    return candidates


def _add_false_positive_drop_candidates(
    table: StressTable,
    intervention: dict[str, str],
    config: ScoredHypothesisConfig,
) -> list[NonOracleHypothesis]:
    removed = _removed_id(intervention)
    candidates: list[NonOracleHypothesis] = []
    objects = [
        obj
        for obj in table.objects
        if str(obj["id"]) != removed
    ]
    objects = sorted(objects, key=lambda obj: (float(obj["y"]), -float(obj["h"]), float(obj["x"])))

    for obj in objects[: config.max_candidates_per_kind]:
        dropped_id = str(obj["id"])
        remaining = [dict(candidate) for candidate in table.objects if str(candidate["id"]) != dropped_id]
        if not remaining:
            continue
        candidates.append(
            NonOracleHypothesis(
                name=f"drop_candidate_{_safe_id(dropped_id)}",
                objects=remaining,
                repair_kind="false_positive_drop_candidate",
                metadata={
                    "dropped_ground_contact": float(obj["y"]) <= config.contact_tolerance,
                    "dropped_height": float(obj["h"]),
                    "dropped_width": float(obj["w"]),
                },
            )
        )
    return candidates


def generate_non_oracle_hypotheses(
    scene: dict[str, object],
    table: StressTable,
    config: ScoredHypothesisConfig,
) -> list[NonOracleHypothesis]:
    """Generate candidates without using true affected/source/false-positive IDs."""

    hypotheses = [NonOracleHypothesis("observed", _clone_objects(table.objects), "none")]
    intervention = scene["intervention"]
    assert isinstance(intervention, dict)

    if table.failure_mode == "missing":
        hypotheses.extend(_add_missing_support_candidates(table, intervention, config))
    elif table.failure_mode == "merge":
        hypotheses.extend(_add_merge_split_candidates(table, config))
    elif table.failure_mode == "false_positive":
        hypotheses.extend(_add_false_positive_drop_candidates(table, intervention, config))

    return hypotheses


def _candidate_route(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str],
    table: StressTable,
    config: ScoredHypothesisConfig,
) -> dict[str, dict[str, object]]:
    tolerance = _tolerance(config)
    detected_intervention = _detected_intervention(intervention, objects)
    hard = infer_stability(objects, detected_intervention, tolerance=tolerance)
    possible = possible_stability(
        objects,
        detected_intervention,
        x_radius=table.x_radius,
        y_radius=table.y_radius,
        tolerance=tolerance,
    )
    interval_accept = _interval_accepts(hard.labels, possible)
    repaired = repair_objects(objects, detected_intervention)
    repaired_labels = infer_stability(repaired.objects, detected_intervention, tolerance=tolerance).labels

    routed: dict[str, dict[str, object]] = {}
    for oid, hard_label in hard.labels.items():
        object_id = str(oid)
        accept = interval_accept[object_id]
        label = hard_label
        stage = "hard"
        if (
            not accept
            and hard_label == "falls"
            and possible.get(object_id, False)
            and repaired_labels.get(object_id) == "stable"
        ):
            accept = True
            label = "stable"
            stage = "repair"
        routed[object_id] = {
            "accept": accept,
            "label": label,
            "stage": stage,
            "possible_stable": possible.get(object_id, False),
        }
    return routed


def _original_records(
    scene: dict[str, object],
    hypothesis: NonOracleHypothesis,
    table: StressTable,
    config: ScoredHypothesisConfig,
) -> dict[str, dict[str, object]]:
    intervention = scene["intervention"]
    assert isinstance(intervention, dict)
    routed = _candidate_route(hypothesis.objects, intervention, table, config)

    labels = scene["labels"]
    assert isinstance(labels, dict)
    records: dict[str, dict[str, object]] = {}
    for oid in labels:
        object_id = str(oid)
        routed_record = routed.get(object_id)
        if routed_record is None:
            records[object_id] = {
                "accept": False,
                "label": "missing",
                "stage": "missing",
            }
        else:
            records[object_id] = dict(routed_record)
    return records


def _score_hypothesis(
    scene: dict[str, object],
    table: StressTable,
    hypothesis: NonOracleHypothesis,
    config: ScoredHypothesisConfig,
) -> RankedHypothesis:
    intervention = scene["intervention"]
    assert isinstance(intervention, dict)
    routed = _candidate_route(hypothesis.objects, intervention, table, config)
    total = len(routed)
    accepted = sum(1 for record in routed.values() if bool(record["accept"]))
    stable = sum(1 for record in routed.values() if record["label"] == "stable")
    synthetic = sum(1 for obj in hypothesis.objects if str(obj["id"]).startswith("hyp_"))
    coverage = accepted / total if total else 0.0
    stable_fraction = stable / total if total else 0.0

    mode_bonus = 0.0
    if table.failure_mode == "missing":
        mode_bonus = (
            0.3 * int(hypothesis.repair_kind == "missing_support_added")
            + 0.25 * stable_fraction
            - 0.03 * synthetic
        )
    elif table.failure_mode == "merge":
        mode_bonus = (
            0.35 * int(hypothesis.repair_kind == "merge_split_candidate")
            + 0.15 * stable_fraction
            - 0.02 * synthetic
        )
    elif table.failure_mode == "false_positive":
        dropped_ground = bool(hypothesis.metadata.get("dropped_ground_contact", False))
        mode_bonus = (
            0.35 * int(hypothesis.repair_kind == "false_positive_drop_candidate")
            + 0.25 * int(dropped_ground)
            - 0.08 * stable_fraction
            - 0.02 * total
        )

    score = coverage + mode_bonus
    return RankedHypothesis(
        hypothesis=hypothesis,
        score=score,
        score_parts={
            "score": score,
            "coverage": coverage,
            "stable_fraction": stable_fraction,
            "accepted": accepted,
            "total": total,
            "synthetic": synthetic,
            "mode_bonus": mode_bonus,
            "repair_kind": hypothesis.repair_kind,
        },
    )


def rank_non_oracle_hypotheses(
    scene: dict[str, object],
    table: StressTable,
    config: ScoredHypothesisConfig,
) -> list[RankedHypothesis]:
    ranked = [
        _score_hypothesis(scene, table, hypothesis, config)
        for hypothesis in generate_non_oracle_hypotheses(scene, table, config)
    ]
    return sorted(
        ranked,
        key=lambda item: (
            item.score,
            item.score_parts["coverage"],
            -int(item.hypothesis.name == "observed"),
            item.hypothesis.name,
        ),
        reverse=True,
    )


def evaluate_scored_hypothesis_scene(
    scene: dict[str, object],
    table: StressTable,
    config: ScoredHypothesisConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    ranked = rank_non_oracle_hypotheses(scene, table, config)
    selected = ranked[0]
    observed = next(item for item in ranked if item.hypothesis.name == "observed")
    observed_records = _original_records(scene, observed.hypothesis, table, config)
    selected_records = _original_records(scene, selected.hypothesis, table, config)

    oracle_records, _ = evaluate_object_hypothesis_scene(
        scene,
        table,
        ObjectHypothesisConfig(
            quick=config.quick,
            seed=config.seed,
            eval_scenes=config.eval_scenes,
            localization_deltas=config.localization_deltas,
            uncertainty_multipliers=config.uncertainty_multipliers,
            failure_modes=config.failure_modes,
            render_scale=config.render_scale,
            render_padding=config.render_padding,
            contact_tolerance=config.contact_tolerance,
            horizontal_tolerance=config.horizontal_tolerance,
        ),
    )
    oracle_by_id = {str(record["object_id"]): record for record in oracle_records}

    labels = scene["labels"]
    assert isinstance(labels, dict)
    action_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    for oid, expected in labels.items():
        object_id = str(oid)
        observed_record = observed_records[object_id]
        selected_record = selected_records[object_id]
        oracle_record = oracle_by_id[object_id]
        guarded_accept = bool(observed_record["accept"]) and not table.structural_failure
        action_counts.update([str(selected_record["stage"])])

        records.append(
            {
                "object_id": object_id,
                "expected": expected,
                "source_status": table.source_status.get(object_id, "missing_detector"),
                "failure_mode": table.failure_mode,
                "structural_failure": table.structural_failure,
                "candidate_count": len(ranked),
                "selected_hypothesis": selected.hypothesis.name,
                "selected_repair_kind": selected.hypothesis.repair_kind,
                "selected_score": selected.score,
                "observed_accept": bool(observed_record["accept"]),
                "observed_label": observed_record["label"],
                "observed_stage": observed_record["stage"],
                "guarded_accept": guarded_accept,
                "guarded_label": observed_record["label"],
                "oracle_accept": bool(oracle_record["hypothesis_accept"]),
                "oracle_label": oracle_record["hypothesis_label"],
                "oracle_hypothesis": oracle_record["selected_hypothesis"],
                "scored_accept": bool(selected_record["accept"]),
                "scored_label": selected_record["label"],
                "scored_stage": selected_record["stage"],
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


def _recovered(records: Sequence[dict[str, object]], accept_key: str, label_key: str) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if bool(record["structural_failure"])
        and not bool(record["guarded_accept"])
        and bool(record[accept_key])
        and record[label_key] == record["expected"]
    ]


def _summarize_records(
    records: Sequence[dict[str, object]],
    action_counts: Counter[str],
    uncertainty_multiplier: float,
) -> dict[str, object]:
    observed_wrong = _accepted_wrong(records, "observed_accept", "observed_label")
    guarded_wrong = _accepted_wrong(records, "guarded_accept", "guarded_label")
    oracle_wrong = _accepted_wrong(records, "oracle_accept", "oracle_label")
    scored_wrong = _accepted_wrong(records, "scored_accept", "scored_label")
    oracle_recovered = _recovered(records, "oracle_accept", "oracle_label")
    scored_recovered = _recovered(records, "scored_accept", "scored_label")

    return {
        "uncertainty_multiplier": uncertainty_multiplier,
        "observed_naive": _accepted_metric(records, "observed_accept", "observed_label"),
        "guarded_structural_abstain": _accepted_metric(records, "guarded_accept", "guarded_label"),
        "oracle_object_hypothesis": _accepted_metric(records, "oracle_accept", "oracle_label"),
        "scored_non_oracle": _accepted_metric(records, "scored_accept", "scored_label"),
        "triage": {
            "source_status": dict(sorted(Counter(str(record["source_status"]) for record in records).items())),
            "selected_hypotheses": dict(sorted(Counter(str(record["selected_hypothesis"]) for record in records).items())),
            "selected_repair_kinds": dict(sorted(Counter(str(record["selected_repair_kind"]) for record in records).items())),
            "observed_accepted_wrong": len(observed_wrong),
            "observed_false_stable": len(_false_stable(records, "observed_accept", "observed_label")),
            "guarded_accepted_wrong": len(guarded_wrong),
            "guarded_false_stable": len(_false_stable(records, "guarded_accept", "guarded_label")),
            "oracle_accepted_wrong": len(oracle_wrong),
            "oracle_false_stable": len(_false_stable(records, "oracle_accept", "oracle_label")),
            "oracle_recovered_structural": len(oracle_recovered),
            "scored_accepted_wrong": len(scored_wrong),
            "scored_false_stable": len(_false_stable(records, "scored_accept", "scored_label")),
            "scored_recovered_structural": len(scored_recovered),
            "scored_recovery_gap_vs_oracle": len(oracle_recovered) - len(scored_recovered),
        },
        "selected_stages": dict(sorted(action_counts.items())),
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
    config: ScoredHypothesisConfig,
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
        scene_records, scene_actions = evaluate_scored_hypothesis_scene(scene, stressed, config)
        records.extend(scene_records)
        action_counts.update(scene_actions)

    return records, action_counts


def _stress_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: ScoredHypothesisConfig,
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


def _eval_splits(config: ScoredHypothesisConfig) -> dict[str, list[dict[str, object]]]:
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


def run_scored_object_hypotheses(
    config: ScoredHypothesisConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _stress_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in LOCALIZATION_MODES.items()
    }

    results = {
        "experiment": "exp95_scored_object_hypotheses",
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
        "scored_object_hypotheses": {
            "localization_deltas": list(config.localization_deltas),
            "uncertainty_multipliers": list(config.uncertainty_multipliers),
            "failure_modes": list(config.failure_modes),
            "modes": modes,
        },
        "interpretation": {
            "observed_naive": "runs interval+repair on the detector table exactly as emitted",
            "guarded_structural_abstain": "abstains structural detector failures, matching exp93's safe route",
            "oracle_object_hypothesis": "exp94-style upper bound using true structural anomaly metadata",
            "scored_non_oracle": (
                "generates and ranks object-table candidates without affected_ids, source_ids, "
                "or false_positive_ids; remaining gaps expose identity/cardinality ranking work"
            ),
        },
    }

    if output_path is None:
        output_path = result_path(EXPERIMENT_NAME, quick=config.quick)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["results_path"] = portable_path(output_path)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scored non-oracle object-hypothesis benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_scored_object_hypotheses(ScoredHypothesisConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
