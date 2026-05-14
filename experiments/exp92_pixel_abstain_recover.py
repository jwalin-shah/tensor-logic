"""Pixel-facing abstain/recover benchmark for support/stability.

Exp91 showed that calibrated coordinate bands let the TL support engine avoid
forcing false hard falls. This experiment moves one step upstream: render
synthetic object tables into a pixel-space segmentation stub, recover object
candidate boxes with detector-style localization error, attach uncertainty
bands, then route accepted cases to TL and ambiguous false-fall candidates to
deterministic repair.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
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
from experiments.runtime_paths import default_runtime_result_path, portable_path


RESULT_DIR = Path(__file__).with_name("exp92_pixel_abstain_recover_data")
LOCALIZATION_MODES = {
    "x_only": ("x",),
    "y_only": ("y",),
    "xy": ("x", "y"),
}


@dataclass(frozen=True)
class PixelBenchmarkConfig:
    quick: bool = False
    seed: int = 9200
    eval_scenes: int = 80
    localization_deltas: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05)
    uncertainty_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    render_scale: int = 64
    render_padding: float = 0.5
    contact_tolerance: float = 0.001
    horizontal_tolerance: float = 0.001
    confidence_threshold: float = 2 / 3
    near_multiplier: float = 20.0

    @classmethod
    def for_mode(cls, quick: bool) -> "PixelBenchmarkConfig":
        if quick:
            return cls(
                quick=True,
                eval_scenes=6,
                localization_deltas=(0.0, 0.001, 0.01),
                uncertainty_multipliers=(1.0, 2.0),
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


@dataclass(frozen=True)
class PixelBox:
    object_id: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class RenderedScene:
    width: int
    height: int
    scale: int
    origin_x: float
    origin_y: float
    boxes: tuple[PixelBox, ...]
    pixels: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class DetectedTable:
    objects: list[dict[str, object]]
    x_radius: float
    y_radius: float
    pixel_radius: float
    localization_delta: float
    uncertainty_multiplier: float


def _tolerance(config: PixelBenchmarkConfig) -> SupportTolerance:
    return SupportTolerance(contact=config.contact_tolerance, horizontal=config.horizontal_tolerance)


def _seed_for(base_seed: int, mode: str, split: str, index: int, level: float) -> int:
    text_score = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{mode}:{split}"))
    level_score = int(round(level * 1_000_000))
    return base_seed + text_score + index * 9973 + level_score * 37


def _scene_bounds(objects: Sequence[dict[str, object]], padding: float) -> tuple[float, float, float, float]:
    min_x = min(float(obj["x"]) for obj in objects) - padding
    min_y = min(float(obj["y"]) for obj in objects) - padding
    max_x = max(float(obj["x"]) + float(obj["w"]) for obj in objects) + padding
    max_y = max(float(obj["y"]) + float(obj["h"]) for obj in objects) + padding
    return min_x, min_y, max_x, max_y


def render_scene(
    objects: Sequence[dict[str, object]],
    scale: int = 64,
    padding: float = 0.5,
    include_pixels: bool = False,
) -> RenderedScene:
    """Rasterize object rectangles into a label-image coordinate frame."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    if not objects:
        raise ValueError("objects must not be empty")

    min_x, min_y, max_x, max_y = _scene_bounds(objects, padding)
    width = max(1, int(math.ceil((max_x - min_x) * scale)))
    height = max(1, int(math.ceil((max_y - min_y) * scale)))
    boxes: list[PixelBox] = []

    for obj in objects:
        x0 = int(math.floor((float(obj["x"]) - min_x) * scale))
        y0 = int(math.floor((float(obj["y"]) - min_y) * scale))
        x1 = int(math.ceil((float(obj["x"]) + float(obj["w"]) - min_x) * scale))
        y1 = int(math.ceil((float(obj["y"]) + float(obj["h"]) - min_y) * scale))
        boxes.append(
            PixelBox(
                object_id=str(obj["id"]),
                x0=max(0, min(width, x0)),
                y0=max(0, min(height, y0)),
                x1=max(0, min(width, x1)),
                y1=max(0, min(height, y1)),
            )
        )

    pixels: tuple[tuple[int, ...], ...] | None = None
    if include_pixels:
        label_by_id = {box.object_id: index + 1 for index, box in enumerate(boxes)}
        rows = [[0 for _ in range(width)] for _ in range(height)]
        for box in boxes:
            label = label_by_id[box.object_id]
            for y in range(box.y0, box.y1):
                row = rows[y]
                for x in range(box.x0, box.x1):
                    row[x] = label
        pixels = tuple(tuple(row) for row in rows)

    return RenderedScene(
        width=width,
        height=height,
        scale=scale,
        origin_x=min_x,
        origin_y=min_y,
        boxes=tuple(boxes),
        pixels=pixels,
    )


def detect_object_table(
    rendered: RenderedScene,
    original_objects: Sequence[dict[str, object]],
    localization_delta: float,
    uncertainty_multiplier: float,
    axes: Sequence[str],
    seed: int,
) -> DetectedTable:
    """Recover object boxes from rendered pixels and attach localization bands."""

    if localization_delta < 0:
        raise ValueError("localization_delta must be non-negative")
    if uncertainty_multiplier < 0:
        raise ValueError("uncertainty_multiplier must be non-negative")

    axis_set = set(axes)
    if not axis_set <= {"x", "y"}:
        raise ValueError("axes must contain only 'x' and/or 'y'")

    originals = {str(obj["id"]): dict(obj) for obj in original_objects}
    rng = random.Random(seed)
    detected: list[dict[str, object]] = []
    for box in rendered.boxes:
        obj = originals[box.object_id]
        x = rendered.origin_x + box.x0 / rendered.scale
        y = rendered.origin_y + box.y0 / rendered.scale
        w = (box.x1 - box.x0) / rendered.scale
        h = (box.y1 - box.y0) / rendered.scale
        if "x" in axis_set and localization_delta:
            x += rng.uniform(-localization_delta, localization_delta)
        if "y" in axis_set and localization_delta:
            y += rng.uniform(-localization_delta, localization_delta)
        detected.append(
            {
                "id": box.object_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "source_w": float(obj["w"]),
                "source_h": float(obj["h"]),
            }
        )

    pixel_radius = 1.0 / rendered.scale
    x_radius = pixel_radius + (localization_delta * uncertainty_multiplier if "x" in axis_set else 0.0)
    y_radius = pixel_radius + (localization_delta * uncertainty_multiplier if "y" in axis_set else 0.0)
    return DetectedTable(
        objects=detected,
        x_radius=x_radius,
        y_radius=y_radius,
        pixel_radius=pixel_radius,
        localization_delta=localization_delta,
        uncertainty_multiplier=uncertainty_multiplier,
    )


def _interval_accepts(
    labels: dict[str, str],
    possible: dict[str, bool],
) -> dict[str, bool]:
    return {
        oid: (label == "stable" and possible[oid]) or (label == "falls" and not possible[oid])
        for oid, label in labels.items()
    }


def _pipeline_records(
    scene: dict[str, object],
    table: DetectedTable,
    config: PixelBenchmarkConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    tolerance = _tolerance(config)
    hard = infer_stability(table.objects, scene["intervention"], tolerance=tolerance)
    confidences = prediction_confidences(
        table.objects,
        scene["intervention"],
        tolerance=tolerance,
        near_multiplier=config.near_multiplier,
    )
    possible = possible_stability(
        table.objects,
        scene["intervention"],
        x_radius=table.x_radius,
        y_radius=table.y_radius,
        tolerance=tolerance,
    )
    interval_accept = _interval_accepts(hard.labels, possible)
    repaired = repair_objects(table.objects, scene["intervention"])
    repaired_labels = infer_stability(repaired.objects, scene["intervention"], tolerance=tolerance).labels
    action_counts = Counter(action.kind for action in repaired.actions)

    labels = scene["labels"]
    assert isinstance(labels, dict)
    records: list[dict[str, object]] = []
    for oid, expected in labels.items():
        object_id = str(oid)
        hard_label = hard.labels[object_id]
        repaired_label = repaired_labels[object_id]
        primitive_accept = confidences[object_id] >= config.confidence_threshold
        pipeline_accept = interval_accept[object_id]
        pipeline_label = hard_label
        pipeline_stage = "hard"

        if not pipeline_accept and hard_label == "falls" and possible[object_id] and repaired_label == "stable":
            pipeline_accept = True
            pipeline_label = repaired_label
            pipeline_stage = "repair"

        records.append(
            {
                "expected": expected,
                "hard": hard_label,
                "primitive_confidence": confidences[object_id],
                "primitive_accept": primitive_accept,
                "possible_stable": possible[object_id],
                "interval_accept": interval_accept[object_id],
                "repaired": repaired_label,
                "pipeline_accept": pipeline_accept,
                "pipeline_label": pipeline_label,
                "pipeline_stage": pipeline_stage,
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


def _summarize_records(
    records: Sequence[dict[str, object]],
    action_counts: Counter[str],
    uncertainty_multiplier: float,
) -> dict[str, object]:
    wrong_hard = [record for record in records if record["hard"] != record["expected"]]
    interval_flagged_wrong = [
        record
        for record in wrong_hard
        if not bool(record["interval_accept"])
    ]
    pipeline_repaired = [
        record
        for record in records
        if record["pipeline_stage"] == "repair"
    ]
    pipeline_accepted = [record for record in records if bool(record["pipeline_accept"])]
    false_supports = [
        record
        for record in wrong_hard
        if record["expected"] == "falls" and record["hard"] == "stable"
    ]
    false_falls = [
        record
        for record in wrong_hard
        if record["expected"] == "stable" and record["hard"] == "falls"
    ]

    return {
        "uncertainty_multiplier": uncertainty_multiplier,
        "hard": _metric(records, "hard"),
        "primitive_confidence": _accepted_metric(records, "primitive_accept", "hard"),
        "interval_feasibility": _accepted_metric(records, "interval_accept", "hard"),
        "repaired": _metric(records, "repaired"),
        "abstain_recover": _accepted_metric(records, "pipeline_accept", "pipeline_label"),
        "triage": {
            "wrong_hard": len(wrong_hard),
            "false_falls": len(false_falls),
            "false_supports": len(false_supports),
            "interval_flagged_wrong": len(interval_flagged_wrong),
            "interval_wrong_coverage": len(interval_flagged_wrong) / len(wrong_hard) if wrong_hard else 0.0,
            "pipeline_repaired": len(pipeline_repaired),
            "pipeline_repaired_correct": sum(
                1 for record in pipeline_repaired if record["pipeline_label"] == record["expected"]
            ),
            "pipeline_accepted_wrong": sum(
                1 for record in pipeline_accepted if record["pipeline_label"] != record["expected"]
            ),
            "false_supports_interval_flagged": sum(
                1 for record in false_supports if not bool(record["interval_accept"])
            ),
        },
        "repair_actions": dict(sorted(action_counts.items())),
    }


def _records_for_multiplier(
    scenes: Sequence[dict[str, object]],
    delta: float,
    axes: Sequence[str],
    uncertainty_multiplier: float,
    seed: int,
    split: str,
    mode_name: str,
    config: PixelBenchmarkConfig,
) -> tuple[list[dict[str, object]], Counter[str]]:
    records: list[dict[str, object]] = []
    action_counts: Counter[str] = Counter()

    for index, scene in enumerate(scenes):
        rendered = render_scene(
            scene["objects"],
            scale=config.render_scale,
            padding=config.render_padding,
        )
        table = detect_object_table(
            rendered,
            scene["objects"],
            localization_delta=delta,
            uncertainty_multiplier=uncertainty_multiplier,
            axes=axes,
            seed=_seed_for(seed, f"pixel:{mode_name}", split, index, delta),
        )
        scene_records, scene_actions = _pipeline_records(scene, table, config)
        records.extend(scene_records)
        action_counts.update(scene_actions)

    return records, action_counts


def _pixel_mode(
    eval_splits: dict[str, list[dict[str, object]]],
    axes: Sequence[str],
    mode_name: str,
    config: PixelBenchmarkConfig,
) -> dict[str, object]:
    by_split: dict[str, list[dict[str, object]]] = {}
    aggregate_rows: list[dict[str, object]] = []

    for delta in config.localization_deltas:
        split_rows: dict[str, list[dict[str, object]]] = {}
        aggregate_by_multiplier: dict[float, list[dict[str, object]]] = {
            multiplier: []
            for multiplier in config.uncertainty_multipliers
        }
        aggregate_actions: dict[float, Counter[str]] = {
            multiplier: Counter()
            for multiplier in config.uncertainty_multipliers
        }

        for split, scenes in eval_splits.items():
            split_multiplier_rows: list[dict[str, object]] = []
            for multiplier in config.uncertainty_multipliers:
                records, action_counts = _records_for_multiplier(
                    scenes,
                    delta,
                    axes,
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
            by_split.setdefault(split, []).append({"delta": delta, "uncertainty": rows})

        aggregate_rows.append(
            {
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


def _eval_splits(config: PixelBenchmarkConfig) -> dict[str, list[dict[str, object]]]:
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


def run_pixel_benchmark(
    config: PixelBenchmarkConfig,
    output_path: Path | None = None,
) -> dict[str, object]:
    eval_splits = _eval_splits(config)
    modes = {
        mode_name: _pixel_mode(eval_splits, axes, mode_name, config)
        for mode_name, axes in LOCALIZATION_MODES.items()
    }

    results = {
        "experiment": "exp92_pixel_abstain_recover",
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
        "pixel_noise": {
            "localization_deltas": list(config.localization_deltas),
            "uncertainty_multipliers": list(config.uncertainty_multipliers),
            "modes": modes,
        },
        "interpretation": {
            "renderer": (
                "synthetic segmentation pixels provide object candidate boxes; "
                "detector-style localization error is applied before TL"
            ),
            "abstain_recover": (
                "hard interval-consistent labels are accepted, ambiguous falls "
                "may be recovered by deterministic object-table repair, and the "
                "rest abstain"
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
    parser = argparse.ArgumentParser(description="Run pixel-facing support/stability abstain/recover benchmark.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_pixel_benchmark(PixelBenchmarkConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
