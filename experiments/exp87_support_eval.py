"""V1 support/stability ID, OOD, and counterfactual evaluation harness."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Callable, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp84_support_data import compute_labels, generate_dataset
from experiments.exp85_support_tl import infer_stability
from experiments.exp86_support_baselines import (
    DeepSetsBaseline,
    PaddedMLP,
    evaluate_model,
    tensorize_scenes,
    train_model,
)
from experiments.runtime_paths import portable_path, result_path


RESULT_DIR = Path(__file__).with_name("exp87_support_data")
EXPERIMENT_NAME = "exp87_support_eval"
OOD_GATE_SPLITS = ("larger_ood", "deeper_ood")


@dataclass(frozen=True)
class EvalConfig:
    quick: bool = False
    seed: int = 8700
    train_scenes: int = 160
    eval_scenes: int = 80
    epochs: int = 80
    hidden_dim: int = 64
    lr: float = 0.01

    @classmethod
    def for_mode(cls, quick: bool) -> "EvalConfig":
        if quick:
            return cls(quick=True, train_scenes=16, eval_scenes=8, epochs=12, hidden_dim=48)
        return cls()


def _obj(i: int, x: float, y: float, w: float, h: float) -> dict[str, float | str]:
    return {"id": f"o{i}", "x": x, "y": y, "w": w, "h": h}


def _scene(objects: Sequence[dict[str, object]], split: str, intervention: dict[str, str]) -> dict[str, object]:
    scene_objects = [dict(obj) for obj in objects]
    return {
        "objects": scene_objects,
        "split": split,
        "intervention": dict(intervention),
        "labels": compute_labels(scene_objects, intervention),
    }


def _remove_target(objects: Sequence[dict[str, object]], rng: random.Random, policy: str) -> str:
    if policy == "bottom":
        return str(objects[0]["id"])
    if policy == "support":
        return str(rng.choice(objects[: max(1, min(3, len(objects)))])["id"])
    if policy == "any":
        return str(rng.choice(objects)["id"])
    raise ValueError(f"unknown removal policy: {policy}")


def _with_interventions(
    base_objects: Sequence[dict[str, object]],
    split: str,
    rng: random.Random,
    interventions: Sequence[str],
    removal_policy: str = "any",
) -> list[dict[str, object]]:
    scenes: list[dict[str, object]] = []
    for mode in interventions:
        if mode == "none":
            scenes.append(_scene(base_objects, split, {"type": "none"}))
        elif mode == "remove":
            scenes.append(
                _scene(
                    base_objects,
                    split,
                    {"type": "remove", "object_id": _remove_target(base_objects, rng, removal_policy)},
                )
            )
        else:
            raise ValueError(f"unknown intervention mode: {mode}")
    return scenes


def _deeper_stack_objects(rng: random.Random, index: int) -> list[dict[str, float | str]]:
    n = rng.randint(5, 8)
    base_w = rng.choice([2.4, 2.8, 3.2])
    y = 0.0
    objects: list[dict[str, float | str]] = []
    for i in range(n):
        h = rng.choice([0.8, 1.0, 1.2])
        w = max(0.9, base_w - 0.18 * i)
        x = 0.1 * (index % 3) + (base_w - w) / 2.0
        objects.append(_obj(i, x, y, w, h))
        y += h
    return objects


def _branching_objects(rng: random.Random, index: int) -> list[dict[str, float | str]]:
    left_w = rng.choice([1.2, 1.6, 2.0])
    right_w = rng.choice([1.1, 1.7, 2.1])
    h = 1.0
    objects: list[dict[str, float | str]] = [
        _obj(0, 0.0, 0.0, left_w, h),
        _obj(1, left_w, 0.0, right_w, h),
    ]
    bridge_w = left_w + right_w
    objects.append(_obj(2, 0.0, h, bridge_w, h))
    y = 2.0
    for i in range(3, rng.randint(5, 8)):
        prev = objects[-1]
        w = max(0.9, float(prev["w"]) - rng.choice([0.1, 0.2, 0.35]))
        x = float(prev["x"]) + (float(prev["w"]) - w) / 2.0 + (0.02 if index % 2 else 0.0)
        objects.append(_obj(i, x, y, w, h))
        y += h
    return objects


def _custom_dataset(
    num_scenes: int,
    split: str,
    seed: int,
    maker: Callable[[random.Random, int], list[dict[str, float | str]]],
    interventions: Sequence[str],
    removal_policy: str,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    scenes: list[dict[str, object]] = []
    for index in range(num_scenes):
        scenes.extend(_with_interventions(maker(rng, index), split, rng, interventions, removal_policy))
    return scenes


def make_splits(config: EvalConfig) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    train = generate_dataset(config.train_scenes, split="id", seed=config.seed, interventions=("none", "remove"))
    eval_splits = {
        "id": generate_dataset(config.eval_scenes, split="id", seed=config.seed + 1, interventions=("none", "remove")),
        "larger_ood": generate_dataset(
            config.eval_scenes, split="ood", seed=config.seed + 2, interventions=("none", "remove")
        ),
        "deeper_ood": _custom_dataset(
            config.eval_scenes,
            split="deeper_ood",
            seed=config.seed + 3,
            maker=_deeper_stack_objects,
            interventions=("none", "remove"),
            removal_policy="support",
        ),
        "branching_ood": _custom_dataset(
            config.eval_scenes,
            split="branching_ood",
            seed=config.seed + 4,
            maker=_branching_objects,
            interventions=("none", "remove"),
            removal_policy="support",
        ),
        "counterfactual": (
            generate_dataset(config.eval_scenes, split="id", seed=config.seed + 5, interventions=("remove",))
            + generate_dataset(config.eval_scenes, split="ood", seed=config.seed + 6, interventions=("remove",))
            + _custom_dataset(
                config.eval_scenes,
                split="deeper_counterfactual",
                seed=config.seed + 7,
                maker=_deeper_stack_objects,
                interventions=("remove",),
                removal_policy="bottom",
            )
            + _custom_dataset(
                config.eval_scenes,
                split="branching_counterfactual",
                seed=config.seed + 8,
                maker=_branching_objects,
                interventions=("remove",),
                removal_policy="support",
            )
        ),
    }
    return train, eval_splits


def _object_accuracy(
    scenes: Sequence[dict[str, object]],
    predict: Callable[[dict[str, object]], dict[str, str]],
) -> dict[str, float | int]:
    correct = 0
    total = 0
    for scene in scenes:
        predicted = predict(scene)
        labels = scene["labels"]
        assert isinstance(labels, dict)
        for oid, expected in labels.items():
            total += 1
            correct += int(predicted[str(oid)] == expected)
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _tl_metrics(scenes: Sequence[dict[str, object]]) -> dict[str, object]:
    missing_proofs = 0
    stable_proofs = 0
    falls_proofs = 0
    support_facts = 0

    def predict(scene: dict[str, object]) -> dict[str, str]:
        nonlocal missing_proofs, stable_proofs, falls_proofs, support_facts
        result = infer_stability(scene["objects"], scene["intervention"])
        support_facts += len(result.supports)
        for oid, label in result.labels.items():
            try:
                result.proof_for(label, oid)
            except KeyError:
                missing_proofs += 1
            else:
                if label == "stable":
                    stable_proofs += 1
                else:
                    falls_proofs += 1
        return result.labels

    accuracy = _object_accuracy(scenes, predict)
    return {
        **accuracy,
        "proofs": {
            "stable": stable_proofs,
            "falls": falls_proofs,
            "missing": missing_proofs,
            "support_facts": support_facts,
            "complete": missing_proofs == 0,
        },
    }


def _split_summary(scenes: Sequence[dict[str, object]]) -> dict[str, object]:
    counts = [len(scene["objects"]) for scene in scenes]
    intervention_types = sorted({str(scene["intervention"]["type"]) for scene in scenes})
    scene_splits = sorted({str(scene["split"]) for scene in scenes})
    return {
        "scenes": len(scenes),
        "objects": sum(counts),
        "object_count_range": [min(counts), max(counts)] if counts else [0, 0],
        "interventions": intervention_types,
        "scene_splits": scene_splits,
    }


def _train_and_eval_baselines(
    train: Sequence[dict[str, object]],
    eval_splits: dict[str, list[dict[str, object]]],
    config: EvalConfig,
) -> dict[str, dict[str, object]]:
    torch.manual_seed(0)
    all_eval = [scene for scenes in eval_splits.values() for scene in scenes]
    max_objects = max(len(scene["objects"]) for scene in list(train) + all_eval)
    train_batch = tensorize_scenes(train, max_objects=max_objects)

    mlp = PaddedMLP(max_objects=max_objects, hidden_dim=config.hidden_dim)
    mlp_losses = train_model(mlp, train_batch, epochs=config.epochs, lr=config.lr)

    deepsets = DeepSetsBaseline(hidden_dim=config.hidden_dim)
    deepsets_losses = train_model(deepsets, train_batch, epochs=config.epochs, lr=config.lr)

    results: dict[str, dict[str, object]] = {
        "mlp": {"loss_start": mlp_losses[0], "loss_end": mlp_losses[-1]},
        "deepsets": {"loss_start": deepsets_losses[0], "loss_end": deepsets_losses[-1]},
    }
    for split, scenes in eval_splits.items():
        batch = tensorize_scenes(scenes, max_objects=max_objects)
        results["mlp"][split] = evaluate_model(mlp, batch)
        results["deepsets"][split] = evaluate_model(deepsets, batch)
    results["metadata"] = {"max_objects": max_objects}
    return results


def _build_gates(tl: dict[str, object], baselines: dict[str, dict[str, object]]) -> dict[str, object]:
    deterministic_splits = ("id", "larger_ood", "deeper_ood", "branching_ood")
    det_correct = sum(int(tl[split]["correct"]) for split in deterministic_splits)
    det_total = sum(int(tl[split]["total"]) for split in deterministic_splits)
    det_acc = det_correct / det_total if det_total else 0.0

    cf_acc = float(tl["counterfactual"]["accuracy"])
    split_margins: dict[str, object] = {}
    for split in OOD_GATE_SPLITS:
        neural_scores = {
            "mlp": float(baselines["mlp"][split]["accuracy"]),
            "deepsets": float(baselines["deepsets"][split]["accuracy"]),
        }
        best_name = max(neural_scores, key=neural_scores.get)
        best_score = neural_scores[best_name]
        tl_score = float(tl[split]["accuracy"])
        split_margins[split] = {
            "tl_accuracy": tl_score,
            "best_neural": best_name,
            "best_neural_accuracy": best_score,
            "margin": tl_score - best_score,
            "passed": (tl_score - best_score) >= 0.10,
        }

    ood_passed = all(bool(item["passed"]) for item in split_margins.values())
    tl_det_passed = det_acc == 1.0
    tl_cf_passed = cf_acc == 1.0
    return {
        "tl_deterministic_label_accuracy_100": {
            "passed": tl_det_passed,
            "accuracy": det_acc,
            "correct": det_correct,
            "total": det_total,
            "threshold": 1.0,
        },
        "tl_counterfactual_retraction_accuracy_100": {
            "passed": tl_cf_passed,
            "accuracy": cf_acc,
            "correct": int(tl["counterfactual"]["correct"]),
            "total": int(tl["counterfactual"]["total"]),
            "threshold": 1.0,
        },
        "tl_ood_margin_vs_best_neural_at_least_10pp": {
            "passed": ood_passed,
            "threshold": 0.10,
            "splits": split_margins,
        },
        "v1_passed": tl_det_passed and tl_cf_passed and ood_passed,
    }


def run_evaluation(config: EvalConfig, output_path: Path | None = None) -> dict[str, object]:
    train, eval_splits = make_splits(config)
    split_summaries = {"train": _split_summary(train)}
    split_summaries.update({name: _split_summary(scenes) for name, scenes in eval_splits.items()})

    if split_summaries["id"]["object_count_range"] == split_summaries["larger_ood"]["object_count_range"]:
        raise RuntimeError("ID and larger OOD splits have the same object-count range")
    if not any("deeper" in split for split in split_summaries["deeper_ood"]["scene_splits"]):
        raise RuntimeError("deeper OOD split metadata missing")
    if not any("branching" in split for split in split_summaries["branching_ood"]["scene_splits"]):
        raise RuntimeError("branching OOD split metadata missing")

    tl = {name: _tl_metrics(scenes) for name, scenes in eval_splits.items()}
    baselines = _train_and_eval_baselines(train, eval_splits, config)
    gates = _build_gates(tl, baselines)

    if not bool(gates["tl_deterministic_label_accuracy_100"]["passed"]):
        raise RuntimeError("TL deterministic label accuracy fell below 100%")
    if not bool(gates["tl_counterfactual_retraction_accuracy_100"]["passed"]):
        raise RuntimeError("TL counterfactual retraction accuracy fell below 100%")
    ood_margin_gate = gates["tl_ood_margin_vs_best_neural_at_least_10pp"]
    if not bool(ood_margin_gate["passed"]):
        failed_splits = []
        for split, split_gate in ood_margin_gate["splits"].items():
            if not bool(split_gate["passed"]):
                failed_splits.append(
                    f"{split}: margin={float(split_gate['margin']):.3f}, "
                    f"threshold={float(ood_margin_gate['threshold']):.3f}"
                )
        details = "; ".join(failed_splits) if failed_splits else "no split details available"
        raise RuntimeError(f"TL OOD margin vs best neural fell below 10pp ({details})")

    results = {
        "experiment": "exp87_support_eval",
        "quick": config.quick,
        "config": asdict(config),
        "splits": split_summaries,
        "tl": tl,
        "mlp": baselines["mlp"],
        "deepsets": baselines["deepsets"],
        "baseline_metadata": baselines["metadata"],
        "gates": gates,
        "thesis": "passed" if gates["v1_passed"] else "falsified",
    }

    if output_path is None:
        output_path = result_path(EXPERIMENT_NAME, quick=config.quick)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["results_path"] = portable_path(output_path)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run support/stability V1 evaluation.")
    parser.add_argument("--quick", action="store_true", help="Use small deterministic smoke-run sizes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional results JSON path.")
    args = parser.parse_args()

    results = run_evaluation(EvalConfig.for_mode(args.quick), output_path=args.output)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
