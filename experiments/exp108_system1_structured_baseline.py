"""exp108: train/evaluate the structured MiniJev-style baseline.

This is a synthetic benchmark-mechanics result, not a LifeOps product claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensor_logic.system1_tasks import generate_pairs
from tensor_logic.system1_training import (
    evaluate_structured_baseline,
    train_structured_baseline,
)


def _metrics_payload(result):
    return {
        "split": result.split,
        "cases": result.cases,
        "decisions": result.decisions,
        "accuracy": result.metrics.accuracy,
        "brier": result.metrics.brier,
        "nll": result.metrics.nll,
        "ece": result.metrics.ece,
        "p50_latency_ms": result.metrics.p50_latency_ms,
        "p95_latency_ms": result.metrics.p95_latency_ms,
        "mean_soft_accuracy": result.mean_soft_accuracy,
        "selective_curve": [
            {
                "threshold": point.threshold,
                "coverage": point.coverage,
                "accuracy": point.accuracy,
                "risk": point.risk,
            }
            for point in result.metrics.selective_curve
        ],
    }


def _dataset_digest(pairs) -> str:
    payload = [
        {
            "scenario": pair.scenario.__dict__,
            "raw_case_id": pair.raw.case_id,
            "structured_case_id": pair.structured.case_id,
            "targets": dict(pair.structured.targets),
            "teacher": pair.structured.target_distributions,
        }
        for pair in pairs
    ]
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run(
    *,
    seed: int = 43,
    iid_count: int = 240,
    ood_per_family: int = 60,
    epochs: int = 300,
) -> dict:
    pairs = generate_pairs(
        seed=seed,
        iid_count=iid_count,
        ood_per_family=ood_per_family,
    )
    model, history = train_structured_baseline(
        pairs,
        epochs=epochs,
        learning_rate=0.06,
        seed=0,
    )

    evaluations = {}
    for split in (
        "test",
        "ood_compositional",
        "ood_source_failure",
        "ood_confidence_shift",
    ):
        evaluations[split] = _metrics_payload(
            evaluate_structured_baseline(
                model,
                pairs,
                split=split,
                model_version="exp108",
            )
        )

    return {
        "experiment": "exp108_system1_structured_baseline",
        "seed": seed,
        "iid_count": iid_count,
        "ood_per_family": ood_per_family,
        "pair_count": len(pairs),
        "dataset_digest": _dataset_digest(pairs),
        "epochs": epochs,
        "initial_loss": history.losses[0],
        "final_loss": history.losses[-1],
        "loss_reduction": history.losses[0] - history.losses[-1],
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "evaluations": evaluations,
        "claim_boundary": (
            "Synthetic teacher-generated tasks only. Use this to validate the "
            "benchmark and compare architecture sensitivity, not to claim "
            "real-world LifeOps decision quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp108.json")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    result = run(epochs=args.epochs)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
