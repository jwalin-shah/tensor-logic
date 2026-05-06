"""Primitive-confidence triage for noisy support/stability scenes.

Exp88 showed that fixed geometry tolerance removes microscopic jitter failures,
but still breaks once noise exceeds that tolerance scale. This experiment keeps
the hard tolerant support/stability engine unchanged and asks whether simple
distance-to-boundary primitive confidences separate reliable predictions from
fragile ones.
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


RESULT_DIR = Path(__file__).with_name("exp89_support_primitive_confidence_data")
GEOMETRY_MODES = {
    "x_only": ("x",),
    "y_only": ("y",),
    "xy": ("x", "y"),
}
BUCKETS = {
    "low": (0.0, 1 / 3),
    "medium": (1 / 3, 2 / 3),
    "high": (2 / 3, 1.000000001),
}


@dataclass(frozen=True)
class ConfidenceConfig:
    quick: bool = False
    seed: int = 8900
    eval_scenes: int = 80
    geometry_deltas: tuple[float, ...] = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05)
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001
    near_multiplier: float = 20.0
    coverage_thresholds: tuple[float, ...] = (0.0, 1 / 3, 2 / 3, 0.9)

    @classmethod
    def for_mode(cls, quick: bool) -> "ConfidenceConfig":
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
        if self.near_multiplier < 1:
            raise ValueError("near_multiplier must be at least 1")


def _interval(obj: dict[str, object]) -> tuple[float, float]:
    x = float(obj["x"])
    return x, x + float(obj["w"])


def _overlap_interval(a: dict[str, object], b: dict[str, object]) -> tuple[float, float] | None:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return lo, hi


def _horizontal_gap(a: dict[str, object], b: dict[str, object]) -> float:
    a0, a1 = _interval(a)
    b0, b1 = _interval(b)
    return max(0.0, max(a0, b0) - min(a1, b1))


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


def _inside_score(distance: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 1.0 if distance <= 0 else 0.0
    return max(0.0, min(1.0, (tolerance - distance) / tolerance))


def _coverage_confidence(
    intervals: Sequence[tuple[float, float]],
    left: float,
    right: float,
    tolerance: float,
) -> float:
    merged = _merge_intervals(intervals)
    if not merged:
        return 0.0

    pos = left
    confidence = 1.0
    for lo, hi in merged:
        gap = max(0.0, lo - pos)
        gap_score = _inside_score(gap, tolerance)
        if gap_score <= 0:
            return 0.0
        confidence = min(confidence, gap_score)
        pos = max(pos, hi)
        if pos >= right:
            return confidence

    final_gap = max(0.0, right - pos)
    final_score = _inside_score(final_gap, tolerance)
    if final_score <= 0:
        return 0.0
    return min(confidence, final_score)


def _support_relation_confidence(
    upper: dict[str, object],
    lower: dict[str, object],
    tolerance: SupportTolerance,
) -> float:
    lower_top = float(lower["y"]) + float(lower["h"])
    contact_score = _inside_score(abs(float(upper["y"]) - lower_top), tolerance.contact)
    horizontal_score = _inside_score(_horizontal_gap(upper, lower), tolerance.horizontal)
    return min(contact_score, horizontal_score)


def _removed_id(intervention: dict[str, str] | None) -> str | None:
    if not intervention or intervention.get("type") == "none":
        return None
    return intervention.get("object_id")


def stability_confidences(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str] | None = None,
    tolerance: SupportTolerance | None = None,
) -> dict[str, float]:
    """Score each object's best tolerant stability proof on [0, 1]."""

    tol = tolerance if tolerance is not None else SupportTolerance(contact=0.001, horizontal=0.001)
    by_id = {str(obj["id"]): dict(obj) for obj in objects}
    removed = _removed_id(intervention)
    active_ids = [oid for oid in by_id if oid != removed]

    scores = {
        oid: _inside_score(abs(float(by_id[oid]["y"])), tol.contact)
        for oid in active_ids
    }

    changed = True
    while changed:
        changed = False
        for oid in active_ids:
            obj = by_id[oid]
            intervals: list[tuple[float, float]] = []
            support_scores: list[float] = []
            for sid in active_ids:
                if sid == oid or scores[sid] <= 0:
                    continue
                supporter = by_id[sid]
                relation_score = _support_relation_confidence(obj, supporter, tol)
                if relation_score <= 0:
                    continue
                overlap = _overlap_interval(obj, supporter)
                if overlap is None:
                    continue
                intervals.append(overlap)
                support_scores.append(min(scores[sid], relation_score))

            left, right = _interval(obj)
            coverage_score = _coverage_confidence(intervals, left, right, tol.horizontal)
            if support_scores and coverage_score > 0:
                candidate = min(min(support_scores), coverage_score)
                if candidate > scores[oid]:
                    scores[oid] = candidate
                    changed = True

    if removed is not None and removed in by_id:
        scores[removed] = 0.0
    return {oid: scores.get(oid, 0.0) for oid in by_id}


def prediction_confidences(
    objects: Sequence[dict[str, object]],
    intervention: dict[str, str] | None = None,
    tolerance: SupportTolerance | None = None,
    near_multiplier: float = 2.0,
) -> dict[str, float]:
    """Return a confidence for the tolerant engine's hard label per object."""

    tol = tolerance if tolerance is not None else SupportTolerance(contact=0.001, horizontal=0.001)
    result = infer_stability(objects, intervention, tolerance=tol)
    stable_scores = stability_confidences(objects, intervention, tolerance=tol)
    near_tol = SupportTolerance(
        contact=tol.contact * near_multiplier,
        horizontal=tol.horizontal * near_multiplier,
    )
    near_stable_scores = stability_confidences(objects, intervention, tolerance=near_tol)

    confidences: dict[str, float] = {}
    for oid, label in result.labels.items():
        if label == "stable":
            confidences[oid] = stable_scores.get(oid, 0.0)
        else:
            confidences[oid] = 1.0 - near_stable_scores.get(oid, 0.0)
    return confidences


def _seed_for(base_seed: int, mode: str, split: str, index: int, level: float) -> int:
    text_score = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{mode}:{split}"))
    level_score = int(round(level * 1_000_000))
    return base_seed + text_score + index * 9973 + level_score * 37


def _metric(records: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    correct = sum(int(record["correct"]) for record in records)
    total = len(records)
    mean_confidence = sum(float(record["confidence"]) for record in records) / total if total else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "mean_confidence": mean_confidence,
    }


def _summarize_records(
    records: Sequence[dict[str, float | int]],
    thresholds: Sequence[float],
) -> dict[str, object]:
    bucket_rows: dict[str, dict[str, float | int]] = {}
    total = len(records)
    for name, (lo, hi) in BUCKETS.items():
        bucket_records = [
            record
            for record in records
            if lo <= float(record["confidence"]) < hi
        ]
        bucket_rows[name] = {
            **_metric(bucket_records),
            "coverage": len(bucket_records) / total if total else 0.0,
        }

    coverage_rows = []
    for threshold in thresholds:
        accepted = [record for record in records if float(record["confidence"]) >= threshold]
        coverage_rows.append(
            {
                "threshold": threshold,
                **_metric(accepted),
                "coverage": len(accepted) / total if total else 0.0,
            }
        )

    high = bucket_rows["high"]
    low = bucket_rows["low"]
    high_low_delta = None
    if high["total"] and low["total"]:
        high_low_delta = float(high["accuracy"]) - float(low["accuracy"])

    return {
        "overall": _metric(records),
        "buckets": bucket_rows,
        "coverage_at_thresholds": coverage_rows,
        "high_minus_low_accuracy": high_low_delta,
    }


def _confidence_records(
    scenes: Sequence[dict[str, object]],
    delta: float,
    axes: Sequence[str],
    seed: int,
    split: str,
    mode_name: str,
    config: ConfidenceConfig,
) -> list[dict[str, float | int]]:
    tolerance = SupportTolerance(
        contact=config.contact_tolerance,
        horizontal=config.horizontal_tolerance,
    )
    records: list[dict[str, float | int]] = []
    for index, scene in enumerate(scenes):
        noisy_objects = perturb_objects(
            scene["objects"],
            delta=delta,
            seed=_seed_for(seed, f"confidence:{mode_name}", split, index, delta),
            axes=axes,
        )
        result = infer_stability(noisy_objects, scene["intervention"], tolerance=tolerance)
        confidences = prediction_confidences(
            noisy_objects,
            scene["intervention"],
            tolerance=tolerance,
            near_multiplier=config.near_multiplier,
        )
        labels = scene["labels"]
        assert isinstance(labels, dict)
        for oid, expected in labels.items():
            object_id = str(oid)
            records.append(
                {
                    "confidence": confidences[object_id],
                    "correct": int(result.labels[object_id] == expected),
                }
            )
    return records


def _confidence_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: ConfidenceConfig,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, object]]] = {}
    aggregate_by_delta: list[dict[str, object]] = []

    for delta in config.geometry_deltas:
        aggregate_records: list[dict[str, float | int]] = []
        for split, scenes in eval_splits.items():
            records = _confidence_records(scenes, delta, axes, config.seed, split, mode_name, config)
            aggregate_records.extend(records)
            by_split.setdefault(split, []).append(
                {"delta": delta, **_summarize_records(records, config.coverage_thresholds)}
            )
        aggregate_by_delta.append(
            {"delta": delta, **_summarize_records(aggregate_records, config.coverage_thresholds)}
        )

    return {
        "axes": list(axes),
        "by_split": by_split,
        "all": aggregate_by_delta,
    }


def _eval_splits(config: ConfidenceConfig) -> dict[str, list[dict[str, object]]]:
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


def run_confidence_evaluation(
    config: ConfidenceConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _confidence_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in GEOMETRY_MODES.items()
    }

    results = {
        "experiment": "exp89_support_primitive_confidence",
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
        "confidence_noise": {
            "deltas": list(config.geometry_deltas),
            "modes": modes,
        },
        "interpretation": {
            "confidence": (
                "distance-to-threshold primitive confidence is a triage signal, "
                "not a replacement for hard TL labels"
            ),
            "near_multiplier": (
                "fall confidence uses a wider near-support band so plausible "
                "perception jitter is not treated as a certain fall"
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
    parser = argparse.ArgumentParser(description="Run support/stability primitive-confidence sweep.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_confidence_evaluation(ConfidenceConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
