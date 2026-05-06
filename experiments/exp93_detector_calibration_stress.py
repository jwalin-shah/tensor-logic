"""Detector calibration stress benchmark for pixel-facing TL.

Exp92 proved that a clean segmentation-style front end can pass calibrated
coordinate bands into TL and recover useful accepted accuracy. This experiment
stresses that interface with detector-level failures: missing boxes, merged
boxes, false-positive supports, and under-calibrated uncertainty.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp85_support_tl import SupportTolerance, infer_stability
from experiments.exp87_support_eval import EvalConfig, make_splits
from experiments.exp89_support_primitive_confidence import prediction_confidences
from experiments.exp90_support_repair_sweep import repair_objects
from experiments.exp91_interval_support_uncertainty import possible_stability
from experiments.exp92_pixel_abstain_recover import (
    DetectedTable,
    detect_object_table,
    render_scene,
)


RESULT_DIR = Path(__file__).with_name("exp93_detector_calibration_stress_data")
LOCALIZATION_MODES = {
    "x_only": ("x",),
    "y_only": ("y",),
    "xy": ("x", "y"),
}
FAILURE_MODES = ("coordinate", "missing", "merge", "false_positive")


@dataclass(frozen=True)
class DetectorStressConfig:
    quick: bool = False
    seed: int = 9300
    eval_scenes: int = 64
    localization_deltas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05)
    uncertainty_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    failure_modes: tuple[str, ...] = FAILURE_MODES
    render_scale: int = 64
    render_padding: float = 0.5
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001
    confidence_threshold: float = 2 / 3
    near_multiplier: float = 20.0

    @classmethod
    def for_mode(cls, quick: bool) -> "DetectorStressConfig":
        if quick:
            return cls(
                quick=True,
                eval_scenes=5,
                localization_deltas=(0.0, 0.01),
                uncertainty_multipliers=(0.5, 1.0),
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
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.near_multiplier < 1:
            raise ValueError("near_multiplier must be at least 1")
        if any(delta < 0 for delta in self.localization_deltas):
            raise ValueError("localization_deltas must be non-negative")
        if any(multiplier < 0 for multiplier in self.uncertainty_multipliers):
            raise ValueError("uncertainty_multipliers must be non-negative")
        if not set(self.failure_modes) <= set(FAILURE_MODES):
            raise ValueError(f"failure_modes must be drawn from {FAILURE_MODES}")


@dataclass(frozen=True)
class StressTable:
    objects: list[dict[str, object]]
    x_radius: float
    y_radius: float
    failure_mode: str
    structural_failure: bool
    source_status: dict[str, str]
    affected_ids: tuple[str, ...]
    false_positive_ids: tuple[str, ...]
    localization_delta: float
    uncertainty_multiplier: float


def _tolerance(config: DetectorStressConfig) -> SupportTolerance:
    return SupportTolerance(contact=config.contact_tolerance, horizontal=config.horizontal_tolerance)


def _seed_for(base_seed: int, mode: str, split: str, index: int, level: float) -> int:
    text_score = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{mode}:{split}"))
    level_score = int(round(level * 1_000_000))
    return base_seed + text_score + index * 9973 + level_score * 37


def _removed_id(intervention: dict[str, str] | None) -> str | None:
    if not intervention or intervention.get("type") == "none":
        return None
    return intervention.get("object_id")


def _detected_intervention(
    intervention: dict[str, str],
    objects: Sequence[dict[str, object]],
) -> dict[str, str]:
    removed = _removed_id(intervention)
    if removed is None:
        return {"type": "none"}
    detected_ids = {str(obj["id"]) for obj in objects}
    if removed not in detected_ids:
        return {"type": "none"}
    return dict(intervention)


def _active_original_ids(scene: dict[str, object]) -> set[str]:
    removed = _removed_id(scene["intervention"])
    return {
        str(obj["id"])
        for obj in scene["objects"]
        if str(obj["id"]) != removed
    }


def _choose_missing_target(scene: dict[str, object], detected: Sequence[dict[str, object]]) -> str | None:
    labels = scene["labels"]
    assert isinstance(labels, dict)
    active_ids = _active_original_ids(scene)
    candidates = [
        obj
        for obj in detected
        if str(obj["id"]) in active_ids and labels.get(str(obj["id"])) == "stable"
    ]
    if not candidates:
        return None
    target = min(candidates, key=lambda obj: (float(obj["y"]), float(obj["x"])))
    return str(target["id"])


def _choose_merge_pair(scene: dict[str, object], detected: Sequence[dict[str, object]]) -> tuple[str, str] | None:
    active_ids = _active_original_ids(scene)
    originals = [obj for obj in detected if str(obj["id"]) in active_ids]
    if len(originals) < 2:
        return None

    def pair_score(pair: tuple[dict[str, object], dict[str, object]]) -> tuple[float, float]:
        a, b = pair
        ax0 = float(a["x"])
        ax1 = ax0 + float(a["w"])
        bx0 = float(b["x"])
        bx1 = bx0 + float(b["w"])
        horizontal_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
        atop = float(a["y"]) + float(a["h"])
        btop = float(b["y"]) + float(b["h"])
        vertical_gap = min(abs(float(a["y"]) - btop), abs(float(b["y"]) - atop))
        return vertical_gap, horizontal_gap

    pairs = [
        (a, b)
        for i, a in enumerate(originals)
        for b in originals[i + 1 :]
    ]
    a, b = min(pairs, key=pair_score)
    return str(a["id"]), str(b["id"])


def _merge_objects(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    x0 = min(float(a["x"]), float(b["x"]))
    y0 = min(float(a["y"]), float(b["y"]))
    x1 = max(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"]))
    y1 = max(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"]))
    aid = str(a["id"])
    bid = str(b["id"])
    return {
        "id": f"merged_{aid}_{bid}",
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
        "source_ids": (aid, bid),
    }


def _choose_false_positive_target(scene: dict[str, object], detected: Sequence[dict[str, object]]) -> dict[str, object] | None:
    labels = scene["labels"]
    assert isinstance(labels, dict)
    removed = _removed_id(scene["intervention"])
    candidates = [
        obj
        for obj in detected
        if str(obj["id"]) != removed and labels.get(str(obj["id"])) == "falls"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda obj: (float(obj["y"]), float(obj["x"])))


def _false_positive_for(target: dict[str, object] | None, rng: random.Random) -> dict[str, object]:
    if target is None:
        return {
            "id": f"fp_{rng.randrange(1_000_000)}",
            "x": 20.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "source_ids": (),
        }

    y = float(target["y"])
    h = max(y, 0.5)
    return {
        "id": f"fp_{rng.randrange(1_000_000)}",
        "x": float(target["x"]),
        "y": 0.0,
        "w": float(target["w"]),
        "h": h,
        "source_ids": (),
    }


def apply_detector_stress(
    scene: dict[str, object],
    table: DetectedTable,
    failure_mode: str,
    seed: int,
) -> StressTable:
    """Apply one detector failure mode to a pixel-derived object table."""

    if failure_mode not in FAILURE_MODES:
        raise ValueError(f"unknown failure mode: {failure_mode}")

    objects = [dict(obj) for obj in table.objects]
    source_status = {str(obj["id"]): "detected" for obj in scene["objects"]}
    affected: list[str] = []
    false_positive_ids: list[str] = []
    structural_failure = failure_mode != "coordinate"
    rng = random.Random(seed)

    if failure_mode == "missing":
        target = _choose_missing_target(scene, objects)
        if target is not None:
            objects = [obj for obj in objects if str(obj["id"]) != target]
            source_status[target] = "missing_detector"
            affected.append(target)

    elif failure_mode == "merge":
        pair = _choose_merge_pair(scene, objects)
        if pair is not None:
            by_id = {str(obj["id"]): obj for obj in objects}
            merged = _merge_objects(by_id[pair[0]], by_id[pair[1]])
            objects = [
                obj
                for obj in objects
                if str(obj["id"]) not in set(pair)
            ]
            objects.append(merged)
            for oid in pair:
                source_status[oid] = "merged_source"
                affected.append(oid)

    elif failure_mode == "false_positive":
        target = _choose_false_positive_target(scene, objects)
        fp = _false_positive_for(target, rng)
        objects.append(fp)
        false_positive_ids.append(str(fp["id"]))
        if target is not None:
            affected.append(str(target["id"]))

    return StressTable(
        objects=objects,
        x_radius=table.x_radius,
        y_radius=table.y_radius,
        failure_mode=failure_mode,
        structural_failure=structural_failure,
        source_status=source_status,
        affected_ids=tuple(affected),
        false_positive_ids=tuple(false_positive_ids),
        localization_delta=table.localization_delta,
        uncertainty_multiplier=table.uncertainty_multiplier,
    )


def _interval_accepts(labels: dict[str, str], possible: dict[str, bool]) -> dict[str, bool]:
    return {
        oid: (label == "stable" and possible[oid]) or (label == "falls" and not possible[oid])
        for oid, label in labels.items()
    }


def evaluate_stress_scene(
    scene: dict[str, object],
    table: StressTable,
    config: DetectorStressConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    tolerance = _tolerance(config)
    intervention = _detected_intervention(scene["intervention"], table.objects)
    hard = infer_stability(table.objects, intervention, tolerance=tolerance)
    confidences = prediction_confidences(
        table.objects,
        intervention,
        tolerance=tolerance,
        near_multiplier=config.near_multiplier,
    )
    possible = possible_stability(
        table.objects,
        intervention,
        x_radius=table.x_radius,
        y_radius=table.y_radius,
        tolerance=tolerance,
    )
    interval_accept = _interval_accepts(hard.labels, possible)
    repaired = repair_objects(table.objects, intervention)
    repaired_labels = infer_stability(repaired.objects, intervention, tolerance=tolerance).labels
    action_counts = Counter(action.kind for action in repaired.actions)

    labels = scene["labels"]
    assert isinstance(labels, dict)
    records: list[dict[str, object]] = []
    for oid, expected in labels.items():
        object_id = str(oid)
        status = table.source_status.get(object_id, "missing_detector")
        hard_label = hard.labels.get(object_id, "missing")
        repaired_label = repaired_labels.get(object_id, "missing")
        detected = object_id in hard.labels
        primitive_accept = detected and confidences[object_id] >= config.confidence_threshold
        interval_ok = detected and interval_accept[object_id]
        naive_accept = interval_ok
        naive_label = hard_label
        naive_stage = "hard"

        if (
            detected
            and not naive_accept
            and hard_label == "falls"
            and possible[object_id]
            and repaired_label == "stable"
        ):
            naive_accept = True
            naive_label = repaired_label
            naive_stage = "repair"

        guarded_accept = naive_accept and not table.structural_failure

        records.append(
            {
                "object_id": object_id,
                "expected": expected,
                "source_status": status,
                "failure_mode": table.failure_mode,
                "structural_failure": table.structural_failure,
                "hard": hard_label,
                "primitive_accept": primitive_accept,
                "possible_stable": possible.get(object_id, False),
                "interval_accept": interval_ok,
                "repaired": repaired_label,
                "naive_accept": naive_accept,
                "naive_label": naive_label,
                "naive_stage": naive_stage,
                "guarded_accept": guarded_accept,
                "guarded_label": naive_label,
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
    wrong_hard = [record for record in records if record["hard"] != record["expected"]]
    structural = [record for record in records if bool(record["structural_failure"])]
    status_counts = Counter(str(record["source_status"]) for record in records)
    naive_wrong = _accepted_wrong(records, "naive_accept", "naive_label")
    guarded_wrong = _accepted_wrong(records, "guarded_accept", "guarded_label")

    return {
        "uncertainty_multiplier": uncertainty_multiplier,
        "hard": _metric(records, "hard"),
        "primitive_confidence": _accepted_metric(records, "primitive_accept", "hard"),
        "interval_feasibility": _accepted_metric(records, "interval_accept", "hard"),
        "repaired": _metric(records, "repaired"),
        "naive_abstain_recover": _accepted_metric(records, "naive_accept", "naive_label"),
        "guarded_abstain_recover": _accepted_metric(records, "guarded_accept", "guarded_label"),
        "triage": {
            "wrong_hard": len(wrong_hard),
            "structural_records": len(structural),
            "source_status": dict(sorted(status_counts.items())),
            "interval_flagged_wrong": sum(1 for record in wrong_hard if not bool(record["interval_accept"])),
            "interval_wrong_coverage": (
                sum(1 for record in wrong_hard if not bool(record["interval_accept"])) / len(wrong_hard)
                if wrong_hard
                else 0.0
            ),
            "naive_repaired": sum(1 for record in records if record["naive_stage"] == "repair"),
            "naive_accepted_wrong": len(naive_wrong),
            "naive_false_stable": len(_false_stable(records, "naive_accept", "naive_label")),
            "guarded_accepted_wrong": len(guarded_wrong),
            "guarded_false_stable": len(_false_stable(records, "guarded_accept", "guarded_label")),
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
    config: DetectorStressConfig,
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
        scene_records, scene_actions = evaluate_stress_scene(scene, stressed, config)
        records.extend(scene_records)
        action_counts.update(scene_actions)

    return records, action_counts


def _stress_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: DetectorStressConfig,
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


def _eval_splits(config: DetectorStressConfig) -> dict[str, list[dict[str, object]]]:
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


def run_detector_stress(
    config: DetectorStressConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _stress_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in LOCALIZATION_MODES.items()
    }

    results = {
        "experiment": "exp93_detector_calibration_stress",
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
        "detector_stress": {
            "localization_deltas": list(config.localization_deltas),
            "uncertainty_multipliers": list(config.uncertainty_multipliers),
            "failure_modes": list(config.failure_modes),
            "modes": modes,
        },
        "interpretation": {
            "naive_abstain_recover": (
                "uses coordinate intervals and repair but does not know when the detector "
                "has emitted a structural anomaly such as a missing or merged object"
            ),
            "guarded_abstain_recover": (
                "requires the detector to expose a scene-level structural-anomaly flag; "
                "it abstains all object labels in such scenes"
            ),
        },
    }

    if output_path is None:
        output_path = RESULT_DIR / ("results_quick.json" if config.quick else "results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        results["results_path"] = str(output_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        results["results_path"] = str(output_path)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector calibration stress benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_detector_stress(DetectorStressConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
