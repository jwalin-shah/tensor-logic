"""Neural baselines for the support/stability object-table task."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp84_support_data import generate_dataset


FEATURE_DIM = 5
LABEL_TO_ID = {"falls": 0, "stable": 1}


@dataclass(frozen=True)
class Batch:
    features: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    object_ids: list[list[str]]


def tensorize_scenes(scenes: Sequence[dict[str, object]], max_objects: int | None = None) -> Batch:
    if max_objects is None:
        max_objects = max(len(scene["objects"]) for scene in scenes)
    features = torch.zeros((len(scenes), max_objects, FEATURE_DIM), dtype=torch.float32)
    labels = torch.full((len(scenes), max_objects), -100, dtype=torch.long)
    mask = torch.zeros((len(scenes), max_objects), dtype=torch.bool)
    object_ids: list[list[str]] = []

    for bi, scene in enumerate(scenes):
        removed_id = None
        intervention = scene.get("intervention") or {}
        if intervention.get("type") == "remove":
            removed_id = intervention.get("object_id")
        ids: list[str] = []
        for oi, obj in enumerate(scene["objects"]):
            if oi >= max_objects:
                raise ValueError(f"scene has more than max_objects={max_objects} objects")
            oid = str(obj["id"])
            ids.append(oid)
            features[bi, oi] = torch.tensor(
                [
                    float(obj["x"]),
                    float(obj["y"]),
                    float(obj["w"]),
                    float(obj["h"]),
                    1.0 if oid == removed_id else 0.0,
                ],
                dtype=torch.float32,
            )
            labels[bi, oi] = LABEL_TO_ID[scene["labels"][oid]]
            mask[bi, oi] = True
        object_ids.append(ids)

    return Batch(features=features, labels=labels, mask=mask, object_ids=object_ids)


class PaddedMLP(nn.Module):
    def __init__(self, max_objects: int, hidden_dim: int = 64):
        super().__init__()
        self.max_objects = max_objects
        self.net = nn.Sequential(
            nn.Linear(max_objects * FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_objects * 2),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = features.shape[0]
        return self.net(features.reshape(batch_size, -1)).reshape(batch_size, self.max_objects, 2)


class DeepSetsBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        masked = encoded * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        pooled = masked.sum(dim=1) / denom
        context = pooled.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        return self.head(torch.cat([encoded, context], dim=-1))


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    pred = logits.argmax(dim=-1)
    valid = mask & (labels >= 0)
    total = int(valid.sum().item())
    correct = int(((pred == labels) & valid).sum().item())
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def _loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_logits = logits[mask]
    valid_labels = labels[mask]
    return F.cross_entropy(valid_logits, valid_labels)


def train_model(model: nn.Module, batch: Batch, epochs: int, lr: float = 0.01) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(batch.features, batch.mask)
        loss = _loss(logits, batch.labels, batch.mask)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    return losses


def evaluate_model(model: nn.Module, batch: Batch) -> dict[str, float | int]:
    with torch.no_grad():
        logits = model(batch.features, batch.mask)
    return masked_accuracy(logits, batch.labels, batch.mask)


def _make_splits(quick: bool) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_n = 16 if quick else 160
    eval_n = 8 if quick else 80
    train = generate_dataset(train_n, split="id", seed=100, interventions=("none", "remove"))
    eval_id = generate_dataset(eval_n, split="id", seed=200, interventions=("none", "remove"))
    eval_ood = generate_dataset(eval_n, split="ood", seed=300, interventions=("none", "remove"))
    return train, eval_id, eval_ood


def run_baselines(quick: bool = False) -> dict[str, object]:
    torch.manual_seed(0)
    train, eval_id, eval_ood = _make_splits(quick)
    max_objects = max(len(scene["objects"]) for scene in train + eval_id + eval_ood)
    train_batch = tensorize_scenes(train, max_objects=max_objects)
    id_batch = tensorize_scenes(eval_id, max_objects=max_objects)
    ood_batch = tensorize_scenes(eval_ood, max_objects=max_objects)
    epochs = 12 if quick else 80

    mlp = PaddedMLP(max_objects=max_objects)
    mlp_losses = train_model(mlp, train_batch, epochs=epochs)

    deepsets = DeepSetsBaseline()
    deepsets_losses = train_model(deepsets, train_batch, epochs=epochs)

    return {
        "quick": quick,
        "max_objects": max_objects,
        "mlp": {
            "id": evaluate_model(mlp, id_batch),
            "ood": evaluate_model(mlp, ood_batch),
            "loss_start": mlp_losses[0],
            "loss_end": mlp_losses[-1],
        },
        "deepsets": {
            "id": evaluate_model(deepsets, id_batch),
            "ood": evaluate_model(deepsets, ood_batch),
            "loss_start": deepsets_losses[0],
            "loss_end": deepsets_losses[-1],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run support/stability neural baselines.")
    parser.add_argument("--quick", action="store_true", help="Use a small fast smoke-run dataset.")
    args = parser.parse_args()
    print(json.dumps(run_baselines(quick=args.quick), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
