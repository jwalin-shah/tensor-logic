"""exp109: paired raw-text vs structured-state System-1 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensor_logic.system1_raw_baseline import (
    evaluate_raw_text_baseline,
    train_raw_text_baseline,
)
from tensor_logic.system1_tasks import generate_pairs
from tensor_logic.system1_training import (
    evaluate_structured_baseline,
    train_structured_baseline,
)


def _metrics(result):
    return {
        "accuracy": result.metrics.accuracy,
        "brier": result.metrics.brier,
        "nll": result.metrics.nll,
        "ece": result.metrics.ece,
        "mean_soft_accuracy": result.mean_soft_accuracy,
        "cases": result.cases,
        "decisions": result.decisions,
    }


def run(
    *,
    seed: int = 53,
    iid_count: int = 240,
    ood_per_family: int = 60,
    epochs: int = 250,
    raw_dim: int = 512,
) -> dict:
    pairs = generate_pairs(
        seed=seed,
        iid_count=iid_count,
        ood_per_family=ood_per_family,
    )

    structured_model, structured_history = train_structured_baseline(
        pairs,
        epochs=epochs,
        learning_rate=0.06,
        seed=0,
    )
    raw_model, raw_losses = train_raw_text_baseline(
        pairs,
        dim=raw_dim,
        epochs=epochs,
        learning_rate=0.06,
        seed=0,
    )

    splits = (
        "test",
        "ood_compositional",
        "ood_source_failure",
        "ood_confidence_shift",
    )
    results = {}

    for split in splits:
        structured = evaluate_structured_baseline(
            structured_model,
            pairs,
            split=split,
            model_version="exp109-structured",
        )
        raw = evaluate_raw_text_baseline(
            raw_model,
            pairs,
            split=split,
            dim=raw_dim,
            model_version="exp109-raw-hash",
        )

        structured_payload = _metrics(structured)
        raw_payload = _metrics(raw)
        results[split] = {
            "structured": structured_payload,
            "raw_hashed": raw_payload,
            "delta_structured_minus_raw": {
                metric: structured_payload[metric] - raw_payload[metric]
                for metric in (
                    "accuracy",
                    "brier",
                    "nll",
                    "ece",
                    "mean_soft_accuracy",
                )
                if structured_payload[metric] is not None
                and raw_payload[metric] is not None
            },
        }

    digest_payload = [
        {
            "id": pair.scenario.scenario_id,
            "split": pair.scenario.split,
            "targets": dict(pair.structured.targets),
        }
        for pair in pairs
    ]
    dataset_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "experiment": "exp109_raw_vs_structured_system1",
        "seed": seed,
        "dataset_digest": dataset_digest,
        "pair_count": len(pairs),
        "epochs": epochs,
        "raw_hash_dim": raw_dim,
        "structured_loss": {
            "initial": structured_history.losses[0],
            "final": structured_history.losses[-1],
        },
        "raw_loss": {
            "initial": raw_losses[0],
            "final": raw_losses[-1],
        },
        "splits": results,
        "claim_boundary": (
            "Both models are small synthetic baselines. The comparison measures "
            "representation sensitivity under this generated task family, not "
            "Laya/Jev/LLM quality and not real LifeOps product performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/exp109.json")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--raw-dim", type=int, default=512)
    args = parser.parse_args()

    result = run(
        epochs=args.epochs,
        raw_dim=args.raw_dim,
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
