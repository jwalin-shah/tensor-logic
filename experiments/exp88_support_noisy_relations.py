"""Noisy primitive robustness sweep for support/stability TL.

The clean V1 support/stability result assumes perfect object tables and exact
primitive relation extraction. This experiment keeps the clean oracle labels
fixed, then corrupts either object geometry or primitive relation facts before
rerunning the deterministic TL support engine.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp85_support_tl import PrimitiveRelations, SupportTolerance, extract_primitives, infer_stability
from experiments.exp87_support_eval import EvalConfig, make_splits


RESULT_DIR = Path(__file__).with_name("exp88_support_noisy_relations_data")
BINARY_RELATIONS = ("touching", "above", "horiz_overlap")
GEOMETRY_MODES = {
    "x_only": ("x",),
    "y_only": ("y",),
    "xy": ("x", "y"),
}


@dataclass(frozen=True)
class NoiseConfig:
    quick: bool = False
    seed: int = 8800
    eval_scenes: int = 80
    geometry_deltas: tuple[float, ...] = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05)
    relation_flip_probabilities: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
    tolerant_contact_tolerance: float = 0.001
    tolerant_horizontal_tolerance: float = 0.001

    @classmethod
    def for_mode(cls, quick: bool) -> "NoiseConfig":
        if quick:
            return cls(
                quick=True,
                eval_scenes=6,
                geometry_deltas=(0.0, 0.0001, 0.001, 0.01),
                relation_flip_probabilities=(0.0, 0.01, 0.05, 0.1),
            )
        return cls()


def perturb_objects(
    objects: Sequence[dict[str, object]],
    delta: float,
    seed: int,
    axes: Sequence[str] = ("x", "y"),
) -> list[dict[str, object]]:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    axis_set = set(axes)
    if not axis_set <= {"x", "y"}:
        raise ValueError("axes must contain only 'x' and/or 'y'")
    if delta == 0:
        return [dict(obj) for obj in objects]

    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for obj in objects:
        noisy = dict(obj)
        for axis in axis_set:
            noisy[axis] = float(noisy[axis]) + rng.uniform(-delta, delta)
        out.append(noisy)
    return out


def flip_primitives(
    primitives: PrimitiveRelations,
    object_ids: Sequence[str],
    probability: float,
    seed: int,
) -> PrimitiveRelations:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if probability == 0.0:
        return primitives

    rng = random.Random(seed)
    active_ids = [oid for oid in object_ids if oid not in primitives.removed]
    pair_universe = [(a, b) for a in active_ids for b in active_ids if a != b]

    flipped_binary: dict[str, frozenset[tuple[str, str]]] = {}
    for relation in BINARY_RELATIONS:
        facts = set(getattr(primitives, relation))
        for fact in pair_universe:
            if rng.random() < probability:
                if fact in facts:
                    facts.remove(fact)
                else:
                    facts.add(fact)
        flipped_binary[relation] = frozenset(facts)

    on_ground = set(primitives.on_ground)
    for oid in active_ids:
        if rng.random() < probability:
            if oid in on_ground:
                on_ground.remove(oid)
            else:
                on_ground.add(oid)

    return PrimitiveRelations(
        touching=flipped_binary["touching"],
        above=flipped_binary["above"],
        horiz_overlap=flipped_binary["horiz_overlap"],
        on_ground=frozenset(on_ground),
        removed=primitives.removed,
    )


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


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
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


def _covers_width(intervals: Iterable[tuple[float, float]], left: float, right: float) -> bool:
    pos = left
    for lo, hi in _merge_intervals(intervals):
        if lo > pos:
            return False
        pos = max(pos, hi)
        if pos >= right:
            return True
    return pos >= right


def infer_labels_from_primitives(
    objects: Sequence[dict[str, object]],
    primitives: PrimitiveRelations,
) -> dict[str, str]:
    by_id = {str(obj["id"]): dict(obj) for obj in objects}
    if len(by_id) != len(objects):
        raise ValueError("duplicate object id")

    active_ids = [oid for oid in by_id if oid not in primitives.removed]
    stable: set[str] = set()
    changed = True
    while changed:
        changed = False
        for oid in active_ids:
            if oid in stable:
                continue
            if oid in primitives.on_ground:
                stable.add(oid)
                changed = True
                continue

            obj = by_id[oid]
            support_intervals: list[tuple[float, float]] = []
            for sid in stable:
                contact = (oid, sid)
                if contact not in primitives.touching:
                    continue
                if contact not in primitives.above:
                    continue
                if contact not in primitives.horiz_overlap:
                    continue
                overlap = _overlap_interval(obj, by_id[sid])
                if overlap is not None:
                    support_intervals.append(overlap)

            left, right = _interval(obj)
            if _covers_width(support_intervals, left, right):
                stable.add(oid)
                changed = True

    return {oid: ("stable" if oid in stable else "falls") for oid in by_id}


def object_accuracy(
    scenes: Sequence[dict[str, object]],
    predictions: Sequence[dict[str, str]],
) -> dict[str, float | int]:
    if len(scenes) != len(predictions):
        raise ValueError("scenes and predictions must have the same length")
    correct = 0
    total = 0
    for scene, predicted in zip(scenes, predictions):
        labels = scene["labels"]
        assert isinstance(labels, dict)
        for oid, expected in labels.items():
            total += 1
            correct += int(predicted[str(oid)] == expected)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _seed_for(base_seed: int, mode: str, split: str, index: int, level: float) -> int:
    text_score = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{mode}:{split}"))
    level_score = int(round(level * 1_000_000))
    return base_seed + text_score + index * 9973 + level_score * 37


def _first_below_100(rows: Sequence[dict[str, float | int]], level_key: str) -> dict[str, float | int] | None:
    for row in rows:
        if float(row["accuracy"]) < 1.0:
            return {
                level_key: row[level_key],
                "accuracy": row["accuracy"],
                "correct": row["correct"],
                "total": row["total"],
            }
    return None


def _with_aggregate(
    by_split: dict[str, list[dict[str, float | int]]],
    levels: Sequence[float],
    level_key: str,
) -> dict[str, list[dict[str, float | int]]]:
    aggregate: list[dict[str, float | int]] = []
    for index, level in enumerate(levels):
        correct = sum(int(rows[index]["correct"]) for rows in by_split.values())
        total = sum(int(rows[index]["total"]) for rows in by_split.values())
        aggregate.append({level_key: level, "accuracy": correct / total if total else 0.0, "correct": correct, "total": total})
    return {**by_split, "all": aggregate}


def _geometry_noise_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    deltas: Sequence[float],
    axes: Sequence[str],
    seed: int,
    mode_name: str,
    tolerance: SupportTolerance | None = None,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, float | int]]] = {}
    for split, scenes in eval_splits.items():
        rows: list[dict[str, float | int]] = []
        for delta in deltas:
            predictions = []
            for index, scene in enumerate(scenes):
                noisy_objects = perturb_objects(
                    scene["objects"],
                    delta=delta,
                    seed=_seed_for(seed, f"geometry:{mode_name}", split, index, delta),
                    axes=axes,
                )
                predictions.append(infer_stability(noisy_objects, scene["intervention"], tolerance=tolerance).labels)
            rows.append({"delta": delta, **object_accuracy(scenes, predictions)})
        by_split[split] = rows

    by_split = _with_aggregate(by_split, deltas, "delta")
    return {
        "axes": list(axes),
        "by_split": by_split,
        "first_below_100": {split: _first_below_100(rows, "delta") for split, rows in by_split.items()},
    }


def evaluate_geometry_noise(
    eval_splits: dict[str, list[dict[str, object]]],
    deltas: Sequence[float],
    seed: int,
    tolerance: SupportTolerance | None = None,
) -> dict[str, object]:
    return {
        "deltas": list(deltas),
        "modes": {
            mode_name: _geometry_noise_mode(eval_splits, deltas, axes, seed, mode_name, tolerance=tolerance)
            for mode_name, axes in GEOMETRY_MODES.items()
        },
    }


def evaluate_relation_flip_noise(
    eval_splits: dict[str, list[dict[str, object]]],
    probabilities: Sequence[float],
    seed: int,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, float | int]]] = {}
    for split, scenes in eval_splits.items():
        rows: list[dict[str, float | int]] = []
        for probability in probabilities:
            predictions = []
            for index, scene in enumerate(scenes):
                primitives = extract_primitives(scene["objects"], scene["intervention"])
                object_ids = [str(obj["id"]) for obj in scene["objects"]]
                noisy_primitives = flip_primitives(
                    primitives,
                    object_ids=object_ids,
                    probability=probability,
                    seed=_seed_for(seed, "relation_flip", split, index, probability),
                )
                predictions.append(infer_labels_from_primitives(scene["objects"], noisy_primitives))
            rows.append({"probability": probability, **object_accuracy(scenes, predictions)})
        by_split[split] = rows

    by_split = _with_aggregate(by_split, probabilities, "probability")
    return {
        "probabilities": list(probabilities),
        "by_split": by_split,
        "first_below_100": {split: _first_below_100(rows, "probability") for split, rows in by_split.items()},
    }


def _eval_splits(config: NoiseConfig) -> dict[str, list[dict[str, object]]]:
    base = EvalConfig.for_mode(config.quick)
    eval_config = replace(base, seed=config.seed, eval_scenes=config.eval_scenes)
    _, eval_splits = make_splits(eval_config)
    return eval_splits


def run_noise_evaluation(config: NoiseConfig, output_path: Path | None = None) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    geometry = evaluate_geometry_noise(eval_splits, config.geometry_deltas, config.seed)
    tolerant_geometry = evaluate_geometry_noise(
        eval_splits,
        config.geometry_deltas,
        config.seed,
        tolerance=SupportTolerance(
            contact=config.tolerant_contact_tolerance,
            horizontal=config.tolerant_horizontal_tolerance,
        ),
    )
    relation_flips = evaluate_relation_flip_noise(eval_splits, config.relation_flip_probabilities, config.seed + 1000)

    results = {
        "experiment": "exp88_support_noisy_relations",
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
        "geometry_noise": geometry,
        "tolerant_geometry_noise": {
            "tolerance": {
                "contact": config.tolerant_contact_tolerance,
                "horizontal": config.tolerant_horizontal_tolerance,
            },
            **tolerant_geometry,
        },
        "relation_flip_noise": relation_flips,
        "interpretation": {
            "geometry": "strict primitive extraction is brittle to non-zero y jitter and x jitter that opens microscopic support-width gaps",
            "relations": "relation flips measure discrete primitive extraction errors after object geometry is otherwise perfect",
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
    parser = argparse.ArgumentParser(description="Run support/stability noisy relation robustness sweep.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_noise_evaluation(NoiseConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
