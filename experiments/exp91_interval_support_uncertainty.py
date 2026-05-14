"""Interval object-table uncertainty for support/stability predictions.

Exp89 added point-estimate primitive confidence, and exp90 showed that
low-confidence object-table repairs can recover many false falls. This
experiment asks whether explicit coordinate bands can expose the same
uncertainty before committing noisy object tables to hard primitive facts.
"""

from __future__ import annotations

import argparse
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
from experiments.runtime_paths import portable_path, result_path


RESULT_DIR = Path(__file__).with_name("exp91_interval_support_uncertainty_data")
EXPERIMENT_NAME = "exp91_interval_support_uncertainty"
GEOMETRY_MODES = {
    "x_only": ("x",),
    "y_only": ("y",),
    "xy": ("x", "y"),
}


@dataclass(frozen=True)
class IntervalConfig:
    quick: bool = False
    seed: int = 9100
    eval_scenes: int = 80
    geometry_deltas: tuple[float, ...] = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05)
    band_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001
    confidence_threshold: float = 2 / 3
    near_multiplier: float = 20.0

    @classmethod
    def for_mode(cls, quick: bool) -> "IntervalConfig":
        if quick:
            return cls(
                quick=True,
                eval_scenes=6,
                geometry_deltas=(0.0, 0.0001, 0.001, 0.01),
                band_multipliers=(1.0, 2.0),
            )
        return cls()

    def __post_init__(self) -> None:
        if self.contact_tolerance < 0:
            raise ValueError("contact_tolerance must be non-negative")
        if self.horizontal_tolerance < 0:
            raise ValueError("horizontal_tolerance must be non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.near_multiplier < 1:
            raise ValueError("near_multiplier must be at least 1")
        if any(multiplier < 0 for multiplier in self.band_multipliers):
            raise ValueError("band_multipliers must be non-negative")


@dataclass(frozen=True)
class ObjectBand:
    object_id: str
    x: tuple[float, float]
    y: tuple[float, float]
    w: float
    h: float

    @property
    def top(self) -> tuple[float, float]:
        return self.y[0] + self.h, self.y[1] + self.h

    @property
    def x_radius(self) -> float:
        return (self.x[1] - self.x[0]) / 2


def _tolerance(config: IntervalConfig) -> SupportTolerance:
    return SupportTolerance(contact=config.contact_tolerance, horizontal=config.horizontal_tolerance)


def _interval(obj: dict[str, object]) -> tuple[float, float]:
    x = float(obj["x"])
    return x, x + float(obj["w"])


def _removed_id(intervention: dict[str, str] | None) -> str | None:
    if not intervention or intervention.get("type") == "none":
        return None
    return intervention.get("object_id")


def _objects_by_id(objects: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for obj in objects:
        oid = str(obj["id"])
        if oid in by_id:
            raise ValueError(f"duplicate object id: {oid}")
        by_id[oid] = dict(obj)
    return by_id


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


def _covers_width(
    intervals: Sequence[tuple[float, float]],
    left: float,
    right: float,
    tolerance: float,
) -> bool:
    pos = left
    for lo, hi in _merge_intervals(intervals):
        if lo > pos + tolerance:
            return False
        pos = max(pos, hi)
        if pos >= right - tolerance:
            return True
    return pos >= right - tolerance


def _ranges_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
    tolerance: float,
) -> bool:
    return first[0] <= second[1] + tolerance and second[0] <= first[1] + tolerance


def object_bands(
    objects: Sequence[dict[str, object]],
    x_radius: float = 0.0,
    y_radius: float = 0.0,
) -> dict[str, ObjectBand]:
    """Wrap point objects in coordinate uncertainty bands."""

    if x_radius < 0 or y_radius < 0:
        raise ValueError("interval radii must be non-negative")

    bands: dict[str, ObjectBand] = {}
    for obj in objects:
        oid = str(obj["id"])
        x = float(obj["x"])
        y = float(obj["y"])
        if oid in bands:
            raise ValueError(f"duplicate object id: {oid}")
        bands[oid] = ObjectBand(
            object_id=oid,
            x=(x - x_radius, x + x_radius),
            y=(y - y_radius, y + y_radius),
            w=float(obj["w"]),
            h=float(obj["h"]),
        )
    return bands


def _possible_on_ground(band: ObjectBand, tolerance: SupportTolerance) -> bool:
    return band.y[0] <= tolerance.contact and band.y[1] >= -tolerance.contact


def _possible_support_interval(
    upper: dict[str, object],
    lower: dict[str, object],
    upper_band: ObjectBand,
    lower_band: ObjectBand,
    tolerance: SupportTolerance,
) -> tuple[float, float] | None:
    if not _ranges_overlap(upper_band.y, lower_band.top, tolerance.contact):
        return None

    left, right = _interval(upper)
    lower_left = float(lower["x"])
    lower_right = lower_left + float(lower["w"])
    slack = upper_band.x_radius + lower_band.x_radius + tolerance.horizontal
    lo = max(left, lower_left - slack)
    hi = min(right, lower_right + slack)
    if hi <= lo:
        return None
    return lo, hi


def possible_stability(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str] | None = None,
    x_radius: float = 0.0,
    y_radius: float = 0.0,
    tolerance: SupportTolerance | None = None,
) -> dict[str, bool]:
    """Return whether each object can be stable within coordinate bands."""

    tol = tolerance if tolerance is not None else SupportTolerance(contact=0.001, horizontal=0.001)
    by_id = _objects_by_id(objects)
    bands = object_bands(objects, x_radius=x_radius, y_radius=y_radius)
    removed = _removed_id(intervention)
    active_ids = [oid for oid in by_id if oid != removed]

    possible = {
        oid
        for oid in active_ids
        if _possible_on_ground(bands[oid], tol)
    }

    changed = True
    while changed:
        changed = False
        for oid in active_ids:
            if oid in possible:
                continue

            obj = by_id[oid]
            intervals: list[tuple[float, float]] = []
            for sid in possible:
                if sid == oid:
                    continue
                support = _possible_support_interval(
                    obj,
                    by_id[sid],
                    bands[oid],
                    bands[sid],
                    tol,
                )
                if support is not None:
                    intervals.append(support)

            left, right = _interval(obj)
            if _covers_width(intervals, left, right, max(tol.horizontal, 1e-9)):
                possible.add(oid)
                changed = True

    return {oid: oid in possible for oid in by_id}


def _seed_for(base_seed: int, mode: str, split: str, index: int, level: float) -> int:
    text_score = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{mode}:{split}"))
    level_score = int(round(level * 1_000_000))
    return base_seed + text_score + index * 9973 + level_score * 37


def _metric(records: Sequence[dict[str, object]]) -> dict[str, float | int]:
    correct = sum(1 for record in records if record["predicted"] == record["expected"])
    total = len(records)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _accepted_metric(
    records: Sequence[dict[str, object]],
    accept_key: str,
) -> dict[str, float | int]:
    accepted = [record for record in records if bool(record[accept_key])]
    metric = _metric(accepted)
    return {
        **metric,
        "coverage": len(accepted) / len(records) if records else 0.0,
    }


def _summarize_band(records: Sequence[dict[str, object]], band_multiplier: float) -> dict[str, object]:
    wrong = [record for record in records if record["predicted"] != record["expected"]]
    interval_abstained = [record for record in records if not bool(record["interval_accept"])]
    primitive_abstained = [record for record in records if not bool(record["primitive_accept"])]
    false_falls = [
        record
        for record in wrong
        if record["expected"] == "stable" and record["predicted"] == "falls"
    ]
    false_supports = [
        record
        for record in wrong
        if record["expected"] == "falls" and record["predicted"] == "stable"
    ]
    interval_flagged_wrong = [
        record
        for record in wrong
        if not bool(record["interval_accept"])
    ]

    return {
        "band_multiplier": band_multiplier,
        "point": _metric(records),
        "primitive_confidence": _accepted_metric(records, "primitive_accept"),
        "interval_feasibility": _accepted_metric(records, "interval_accept"),
        "triage": {
            "wrong_before": len(wrong),
            "interval_abstained": len(interval_abstained),
            "interval_abstained_wrong": len(interval_flagged_wrong),
            "interval_wrong_coverage": len(interval_flagged_wrong) / len(wrong) if wrong else 0.0,
            "primitive_abstained": len(primitive_abstained),
            "primitive_abstained_wrong": sum(
                1 for record in wrong if not bool(record["primitive_accept"])
            ),
            "false_falls": len(false_falls),
            "false_falls_interval_flagged": sum(
                1 for record in false_falls if not bool(record["interval_accept"])
            ),
            "false_supports": len(false_supports),
            "false_supports_interval_flagged": sum(
                1 for record in false_supports if not bool(record["interval_accept"])
            ),
        },
    }


def _records_for_band(
    scenes: Sequence[dict[str, object]],
    delta: float,
    axes: Sequence[str],
    band_multiplier: float,
    seed: int,
    split: str,
    mode_name: str,
    config: IntervalConfig,
) -> list[dict[str, object]]:
    tolerance = _tolerance(config)
    x_radius = delta * band_multiplier if "x" in axes else 0.0
    y_radius = delta * band_multiplier if "y" in axes else 0.0
    records: list[dict[str, object]] = []

    for index, scene in enumerate(scenes):
        noisy_objects = perturb_objects(
            scene["objects"],
            delta=delta,
            seed=_seed_for(seed, f"interval:{mode_name}", split, index, delta),
            axes=axes,
        )
        result = infer_stability(noisy_objects, scene["intervention"], tolerance=tolerance)
        confidences = prediction_confidences(
            noisy_objects,
            scene["intervention"],
            tolerance=tolerance,
            near_multiplier=config.near_multiplier,
        )
        possible = possible_stability(
            noisy_objects,
            scene["intervention"],
            x_radius=x_radius,
            y_radius=y_radius,
            tolerance=tolerance,
        )
        labels = scene["labels"]
        assert isinstance(labels, dict)

        for oid, expected in labels.items():
            object_id = str(oid)
            predicted = result.labels[object_id]
            possible_stable = possible[object_id]
            interval_accept = (
                predicted == "stable" and possible_stable
            ) or (
                predicted == "falls" and not possible_stable
            )
            records.append(
                {
                    "expected": expected,
                    "predicted": predicted,
                    "primitive_confidence": confidences[object_id],
                    "primitive_accept": confidences[object_id] >= config.confidence_threshold,
                    "possible_stable": possible_stable,
                    "interval_accept": interval_accept,
                }
            )

    return records


def _interval_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: IntervalConfig,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, object]]] = {}
    aggregate_rows: list[dict[str, object]] = []

    for delta in config.geometry_deltas:
        split_rows: dict[str, list[dict[str, object]]] = {}
        aggregate_by_band: dict[float, list[dict[str, object]]] = {
            multiplier: []
            for multiplier in config.band_multipliers
        }

        for split, scenes in eval_splits.items():
            split_band_rows: list[dict[str, object]] = []
            for multiplier in config.band_multipliers:
                records = _records_for_band(
                    scenes,
                    delta,
                    axes,
                    multiplier,
                    config.seed,
                    split,
                    mode_name,
                    config,
                )
                aggregate_by_band[multiplier].extend(records)
                split_band_rows.append(_summarize_band(records, multiplier))
            split_rows[split] = split_band_rows

        for split, rows in split_rows.items():
            by_split.setdefault(split, []).append({"delta": delta, "bands": rows})

        aggregate_rows.append(
            {
                "delta": delta,
                "bands": [
                    _summarize_band(aggregate_by_band[multiplier], multiplier)
                    for multiplier in config.band_multipliers
                ],
            }
        )

    return {
        "axes": list(axes),
        "by_split": by_split,
        "all": aggregate_rows,
    }


def _eval_splits(config: IntervalConfig) -> dict[str, list[dict[str, object]]]:
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


def run_interval_evaluation(
    config: IntervalConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _interval_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in GEOMETRY_MODES.items()
    }

    results = {
        "experiment": "exp91_interval_support_uncertainty",
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
        "interval_noise": {
            "deltas": list(config.geometry_deltas),
            "band_multipliers": list(config.band_multipliers),
            "modes": modes,
        },
        "interpretation": {
            "interval_feasibility": (
                "coordinate bands are used as an abstain/recover signal around "
                "the unchanged hard TL point prediction"
            ),
            "accepted_prediction": (
                "a point fall is accepted only when no stable proof is feasible "
                "inside the interval band"
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
    parser = argparse.ArgumentParser(description="Run support/stability interval uncertainty sweep.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_interval_evaluation(IntervalConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
