"""Low-confidence repair sweep for noisy support/stability scenes.

Exp89 showed that primitive confidence can identify many unreliable noisy
geometry predictions. This experiment tests whether small deterministic
object-table repairs recover those low-confidence failures without widening
the support/stability engine's tolerance or introducing false supports.
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
from experiments.exp88_support_noisy_relations import perturb_objects
from experiments.exp89_support_primitive_confidence import prediction_confidences


RESULT_DIR = Path(__file__).with_name("exp90_support_repair_sweep_data")
GEOMETRY_MODES = {
    "x_only": ("x",),
    "y_only": ("y",),
    "xy": ("x", "y"),
}


@dataclass(frozen=True)
class RepairConfig:
    quick: bool = False
    seed: int = 9000
    eval_scenes: int = 80
    geometry_deltas: tuple[float, ...] = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05)
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001
    repair_contact_radius: float = 0.02
    repair_horizontal_radius: float = 0.02
    confidence_threshold: float = 2 / 3
    near_multiplier: float = 20.0

    @classmethod
    def for_mode(cls, quick: bool) -> "RepairConfig":
        if quick:
            return cls(
                quick=True,
                eval_scenes=6,
                geometry_deltas=(0.0, 0.0001, 0.001, 0.01),
            )
        return cls()

    def __post_init__(self) -> None:
        if self.contact_tolerance < 0:
            raise ValueError("contact_tolerance must be non-negative")
        if self.horizontal_tolerance < 0:
            raise ValueError("horizontal_tolerance must be non-negative")
        if self.repair_contact_radius < self.contact_tolerance:
            raise ValueError("repair_contact_radius must be at least contact_tolerance")
        if self.repair_horizontal_radius < self.horizontal_tolerance:
            raise ValueError("repair_horizontal_radius must be at least horizontal_tolerance")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.near_multiplier < 1:
            raise ValueError("near_multiplier must be at least 1")


@dataclass(frozen=True)
class RepairAction:
    kind: str
    object_id: str
    detail: str


@dataclass(frozen=True)
class RepairResult:
    objects: list[dict[str, object]]
    actions: tuple[RepairAction, ...]


def _tolerance(config: RepairConfig) -> SupportTolerance:
    return SupportTolerance(contact=config.contact_tolerance, horizontal=config.horizontal_tolerance)


def _interval(obj: dict[str, object]) -> tuple[float, float]:
    x = float(obj["x"])
    return x, x + float(obj["w"])


def _top(obj: dict[str, object]) -> float:
    return float(obj["y"]) + float(obj["h"])


def _removed_id(intervention: dict[str, str] | None) -> str | None:
    if not intervention or intervention.get("type") == "none":
        return None
    return intervention.get("object_id")


def _objects_by_id(objects: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(obj["id"]): dict(obj) for obj in objects}


def _horizontal_gap(a: dict[str, object], b: dict[str, object]) -> float:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    return max(0.0, max(a0, b0) - min(a1, b1))


def _overlap_interval(a: dict[str, object], b: dict[str, object]) -> tuple[float, float] | None:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return lo, hi


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _coverage_gap(
    intervals: Sequence[tuple[float, float]],
    left: float,
    right: float,
) -> tuple[float, float]:
    pos = left
    total_gap = 0.0
    max_gap = 0.0
    for lo, hi in _merge_intervals(intervals):
        if hi <= pos:
            continue
        if lo > pos:
            gap = lo - pos
            total_gap += gap
            max_gap = max(max_gap, gap)
        pos = max(pos, hi)
        if pos >= right:
            return total_gap, max_gap
    if pos < right:
        gap = right - pos
        total_gap += gap
        max_gap = max(max_gap, gap)
    return total_gap, max_gap


def _padded_covers(
    intervals: Sequence[tuple[float, float]],
    left: float,
    right: float,
    radius: float,
) -> bool:
    padded = [(lo - radius, hi + radius) for lo, hi in intervals]
    total_gap, _ = _coverage_gap(padded, left, right)
    return total_gap <= 1e-9


def _support_intervals(
    obj: dict[str, object],
    supporters: Sequence[dict[str, object]],
) -> list[tuple[float, float]]:
    intervals = []
    for supporter in supporters:
        overlap = _overlap_interval(obj, supporter)
        if overlap is not None:
            intervals.append(overlap)
    return intervals


def _near_supporters(
    obj: dict[str, object],
    by_id: dict[str, dict[str, object]],
    supporter_ids: Sequence[str],
    contact_radius: float,
    horizontal_radius: float,
) -> list[dict[str, object]]:
    out = []
    for sid in supporter_ids:
        supporter = by_id[sid]
        if abs(float(obj["y"]) - _top(supporter)) <= contact_radius:
            if _horizontal_gap(obj, supporter) <= horizontal_radius:
                out.append(supporter)
    return out


def classify_failures(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str] | None,
    expected_labels: dict[str, str],
    predicted_labels: dict[str, str],
    config: RepairConfig,
) -> dict[str, str]:
    """Classify wrong baseline predictions by repair-relevant geometry cause."""

    by_id = _objects_by_id(objects)
    removed = _removed_id(intervention)
    active_expected_stable = [
        oid
        for oid, label in expected_labels.items()
        if label == "stable" and oid != removed and oid in by_id
    ]
    categories: dict[str, str] = {}

    for oid, expected in expected_labels.items():
        predicted = predicted_labels[str(oid)]
        if predicted == expected:
            continue
        obj = by_id[str(oid)]
        if expected == "falls" and predicted == "stable":
            categories[str(oid)] = "false_support"
            continue

        if expected != "stable" or predicted != "falls":
            categories[str(oid)] = "other"
            continue

        if config.contact_tolerance < abs(float(obj["y"])) <= config.repair_contact_radius:
            categories[str(oid)] = "false_fall_missed_ground"
            continue

        supporter_ids = [sid for sid in active_expected_stable if sid != str(oid)]
        near = _near_supporters(
            obj,
            by_id,
            supporter_ids,
            config.repair_contact_radius,
            config.repair_horizontal_radius,
        )
        if not near:
            categories[str(oid)] = "false_fall_other"
            continue

        min_contact_gap = min(abs(float(obj["y"]) - _top(supporter)) for supporter in near)
        intervals = _support_intervals(obj, near)
        left, right = _interval(obj)
        _, max_gap = _coverage_gap(intervals, left, right)
        if min_contact_gap > config.contact_tolerance:
            categories[str(oid)] = "false_fall_missed_contact"
        elif max_gap > config.horizontal_tolerance and _padded_covers(
            intervals,
            left,
            right,
            config.repair_horizontal_radius,
        ):
            categories[str(oid)] = "false_fall_support_gap"
        else:
            categories[str(oid)] = "false_fall_other"

    return categories


def _low_confidence_falls(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str] | None,
    config: RepairConfig,
) -> tuple[dict[str, str], dict[str, float], set[str]]:
    tolerance = _tolerance(config)
    result = infer_stability(objects, intervention, tolerance=tolerance)
    confidences = prediction_confidences(
        objects,
        intervention,
        tolerance=tolerance,
        near_multiplier=config.near_multiplier,
    )
    candidates = {
        oid
        for oid, label in result.labels.items()
        if label == "falls" and confidences[oid] <= config.confidence_threshold
    }
    return result.labels, confidences, candidates


def _bridge_supporter_gaps(
    repaired: list[dict[str, object]],
    candidate: dict[str, object],
    supporters: Sequence[dict[str, object]],
    config: RepairConfig,
) -> list[RepairAction]:
    by_id = _objects_by_id(repaired)
    actions: list[RepairAction] = []
    ordered = sorted(supporters, key=lambda obj: float(obj["x"]))
    for left, right in zip(ordered, ordered[1:]):
        left_right = _interval(left)[1]
        right_left = _interval(right)[0]
        gap = right_left - left_right
        if config.horizontal_tolerance < gap <= config.repair_horizontal_radius:
            rid = str(right["id"])
            by_id[rid]["x"] = float(by_id[rid]["x"]) - gap
            actions.append(
                RepairAction(
                    "bridge_support_gap",
                    rid,
                    f"closed gap {gap:.6g} while repairing {candidate['id']}",
                )
            )

    if actions:
        id_to_obj = {str(obj["id"]): obj for obj in repaired}
        for oid, obj in by_id.items():
            id_to_obj[oid].update(obj)
    return actions


def repair_objects(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str] | None = None,
    config: RepairConfig | None = None,
) -> RepairResult:
    """Apply deterministic repairs only to low-confidence falling predictions."""

    cfg = config if config is not None else RepairConfig()
    tolerance = _tolerance(cfg)
    repaired = [dict(obj) for obj in objects]
    actions: list[RepairAction] = []

    for _ in range(4):
        labels, _, candidate_ids = _low_confidence_falls(repaired, intervention, cfg)
        if not candidate_ids:
            break

        by_id = _objects_by_id(repaired)
        changed = False

        for oid in sorted(candidate_ids, key=lambda item: float(by_id[item]["y"])):
            obj = by_id[oid]
            if abs(float(obj["y"])) <= cfg.repair_contact_radius:
                before = float(obj["y"])
                if abs(before) > tolerance.contact:
                    obj["y"] = 0.0
                    actions.append(RepairAction("snap_to_ground", oid, f"y {before:.6g} -> 0"))
                    changed = True

        if changed:
            repaired = [by_id[str(obj["id"])] for obj in repaired]
            continue

        stable_ids = [oid for oid, label in labels.items() if label == "stable"]
        for oid in sorted(candidate_ids, key=lambda item: float(by_id[item]["y"])):
            obj = by_id[oid]
            supporters = _near_supporters(
                obj,
                by_id,
                stable_ids,
                cfg.repair_contact_radius,
                cfg.repair_horizontal_radius,
            )
            if not supporters:
                continue

            intervals = _support_intervals(obj, supporters)
            left, right = _interval(obj)
            if not _padded_covers(intervals, left, right, cfg.repair_horizontal_radius):
                continue

            bridge_actions = _bridge_supporter_gaps(repaired, obj, supporters, cfg)
            if bridge_actions:
                actions.extend(bridge_actions)
                changed = True
                by_id = _objects_by_id(repaired)
                obj = by_id[oid]
                supporters = _near_supporters(
                    obj,
                    by_id,
                    stable_ids,
                    cfg.repair_contact_radius,
                    cfg.repair_horizontal_radius,
                )

            target_top = min(supporters, key=lambda supporter: abs(float(obj["y"]) - _top(supporter)))
            before_y = float(obj["y"])
            new_y = _top(target_top)
            if abs(before_y - new_y) > tolerance.contact:
                obj["y"] = new_y
                actions.append(
                    RepairAction(
                        "snap_to_support_top",
                        oid,
                        f"y {before_y:.6g} -> {new_y:.6g}",
                    )
                )
                changed = True

            merged = _merge_intervals(_support_intervals(obj, supporters))
            if merged:
                support_left = merged[0][0]
                support_right = merged[-1][1]
                obj_left, obj_right = _interval(obj)
                before_x = float(obj["x"])
                support_width = support_right - support_left
                if support_width >= float(obj["w"]) - tolerance.horizontal:
                    if 0 < support_left - obj_left <= cfg.repair_horizontal_radius:
                        obj["x"] = support_left
                    elif 0 < obj_right - support_right <= cfg.repair_horizontal_radius:
                        obj["x"] = support_right - float(obj["w"])
                if float(obj["x"]) != before_x:
                    actions.append(
                        RepairAction(
                            "snap_horizontal_edge",
                            oid,
                            f"x {before_x:.6g} -> {float(obj['x']):.6g}",
                        )
                    )
                    changed = True

        repaired = [by_id[str(obj["id"])] for obj in repaired]
        if not changed:
            break

    return RepairResult(objects=repaired, actions=tuple(actions))


def _seed_for(base_seed: int, mode: str, split: str, index: int, level: float) -> int:
    text_score = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{mode}:{split}"))
    level_score = int(round(level * 1_000_000))
    return base_seed + text_score + index * 9973 + level_score * 37


def _object_records(
    scenes: Sequence[dict[str, object]],
    delta: float,
    axes: Sequence[str],
    seed: int,
    split: str,
    mode_name: str,
    config: RepairConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    tolerance = _tolerance(config)
    records: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()

    for index, scene in enumerate(scenes):
        noisy_objects = perturb_objects(
            scene["objects"],
            delta=delta,
            seed=_seed_for(seed, f"repair:{mode_name}", split, index, delta),
            axes=axes,
        )
        baseline = infer_stability(noisy_objects, scene["intervention"], tolerance=tolerance)
        confidences = prediction_confidences(
            noisy_objects,
            scene["intervention"],
            tolerance=tolerance,
            near_multiplier=config.near_multiplier,
        )
        labels = scene["labels"]
        assert isinstance(labels, dict)
        taxonomy = classify_failures(noisy_objects, scene["intervention"], labels, baseline.labels, config)
        repaired = repair_objects(noisy_objects, scene["intervention"], config)
        repaired_labels = infer_stability(repaired.objects, scene["intervention"], tolerance=tolerance).labels
        action_counts.update(action.kind for action in repaired.actions)

        for oid, expected in labels.items():
            object_id = str(oid)
            records.append(
                {
                    "expected": expected,
                    "baseline": baseline.labels[object_id],
                    "repaired": repaired_labels[object_id],
                    "confidence": confidences[object_id],
                    "category": taxonomy.get(object_id, "correct"),
                }
            )
    return records, action_counts


def _accuracy(records: Sequence[dict[str, object]], key: str) -> dict[str, float | int]:
    correct = sum(1 for record in records if record[key] == record["expected"])
    total = len(records)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _summarize_records(
    records: Sequence[dict[str, object]],
    action_counts: Counter[str],
    confidence_threshold: float,
) -> dict[str, object]:
    wrong_before = [record for record in records if record["baseline"] != record["expected"]]
    recovered = [
        record
        for record in wrong_before
        if record["repaired"] == record["expected"]
    ]
    regressions = [
        record
        for record in records
        if record["baseline"] == record["expected"] and record["repaired"] != record["expected"]
    ]
    introduced_false_supports = [
        record
        for record in records
        if record["expected"] == "falls"
        and record["baseline"] == record["expected"]
        and record["repaired"] == "stable"
    ]
    low_confidence = [
        record
        for record in records
        if float(record["confidence"]) <= confidence_threshold
    ]
    high_confidence = [
        record
        for record in records
        if float(record["confidence"]) > confidence_threshold
    ]
    taxonomy = Counter(str(record["category"]) for record in wrong_before)
    recovered_by_category = Counter(str(record["category"]) for record in recovered)

    return {
        "baseline": _accuracy(records, "baseline"),
        "repaired": _accuracy(records, "repaired"),
        "recovery": {
            "wrong_before": len(wrong_before),
            "recovered": len(recovered),
            "recovery_rate": len(recovered) / len(wrong_before) if wrong_before else 0.0,
            "regressions": len(regressions),
            "introduced_false_supports": len(introduced_false_supports),
        },
        "triage": {
            "low_confidence": _accuracy(low_confidence, "baseline"),
            "low_confidence_coverage": len(low_confidence) / len(records) if records else 0.0,
            "high_confidence": _accuracy(high_confidence, "baseline"),
            "high_confidence_coverage": len(high_confidence) / len(records) if records else 0.0,
        },
        "taxonomy": dict(sorted(taxonomy.items())),
        "recovered_by_category": dict(sorted(recovered_by_category.items())),
        "repair_actions": dict(sorted(action_counts.items())),
    }


def _repair_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: RepairConfig,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, object]]] = {}
    aggregate_rows: list[dict[str, object]] = []

    for delta in config.geometry_deltas:
        aggregate_records: list[dict[str, object]] = []
        aggregate_actions: Counter[str] = Counter()
        for split, scenes in eval_splits.items():
            records, action_counts = _object_records(
                scenes,
                delta,
                axes,
                config.seed,
                split,
                mode_name,
                config,
            )
            aggregate_records.extend(records)
            aggregate_actions.update(action_counts)
            by_split.setdefault(split, []).append(
                {
                    "delta": delta,
                    **_summarize_records(records, action_counts, config.confidence_threshold),
                }
            )
        aggregate_rows.append(
            {
                "delta": delta,
                **_summarize_records(
                    aggregate_records,
                    aggregate_actions,
                    config.confidence_threshold,
                ),
            }
        )

    return {
        "axes": list(axes),
        "by_split": by_split,
        "all": aggregate_rows,
    }


def _eval_splits(config: RepairConfig) -> dict[str, list[dict[str, object]]]:
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


def run_repair_evaluation(
    config: RepairConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _repair_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in GEOMETRY_MODES.items()
    }

    results = {
        "experiment": "exp90_support_repair_sweep",
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
        "repair_noise": {
            "deltas": list(config.geometry_deltas),
            "modes": modes,
        },
        "interpretation": {
            "repair": (
                "deterministic object-table repair is a low-confidence recovery path, "
                "not a replacement for perception uncertainty"
            )
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
    parser = argparse.ArgumentParser(description="Run support/stability low-confidence repair sweep.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_repair_evaluation(RepairConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
