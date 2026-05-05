from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

Label = Literal["stable", "falls"]
Split = Literal["id", "ood"]
InterventionKind = Literal["none", "remove"]


@dataclass(frozen=True)
class ObjectState:
    id: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Intervention:
    kind: InterventionKind
    object_id: str | None = None


@dataclass(frozen=True)
class Scene:
    objects: list[ObjectState]
    intervention: Intervention
    labels: dict[str, Label]


def _is_supported_by(child: ObjectState, parent: ObjectState) -> bool:
    child_bottom = child.y
    parent_top = parent.y + parent.h
    if child_bottom != parent_top:
        return False
    child_left, child_right = child.x, child.x + child.w
    parent_left, parent_right = parent.x, parent.x + parent.w
    overlap = min(child_right, parent_right) - max(child_left, parent_left)
    return overlap > 0


def _rectangles_overlap(a: ObjectState, b: ObjectState) -> bool:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    x_overlap = min(ax2, bx2) - max(a.x, b.x)
    y_overlap = min(ay2, by2) - max(a.y, b.y)
    return x_overlap > 0 and y_overlap > 0


def _compute_labels(objects: list[ObjectState], intervention: Intervention) -> dict[str, Label]:
    removed = intervention.object_id if intervention.kind == "remove" else None
    id_to_obj = {obj.id: obj for obj in objects}

    stable: dict[str, bool] = {}
    changed = True
    while changed:
        changed = False
        for obj in objects:
            if obj.id == removed or obj.id in stable:
                continue
            if obj.y == 0:
                stable[obj.id] = True
                changed = True
                continue
            for parent in objects:
                if parent.id == removed or parent.id == obj.id:
                    continue
                if parent.id not in stable:
                    continue
                if _is_supported_by(obj, parent):
                    stable[obj.id] = True
                    changed = True
                    break

    labels: dict[str, Label] = {}
    for object_id in id_to_obj:
        if object_id == removed:
            continue
        labels[object_id] = "stable" if object_id in stable else "falls"
    return labels


def _generate_scene(rng: Random, split: Split, intervention: InterventionKind) -> Scene:
    if split == "id":
        n = rng.randint(2, 3)
        widths = [rng.randint(2, 4) for _ in range(n)]
        heights = [rng.randint(1, 2) for _ in range(n)]
    else:
        n = rng.randint(4, 8)
        widths = [rng.randint(1, 6) for _ in range(n)]
        heights = [rng.randint(1, 4) for _ in range(n)]

    # Build a deterministic support tree: each object above ground has exactly one parent.
    objects: list[ObjectState] = []
    for i in range(n):
        object_id = f"o{i}"
        w = widths[i]
        h = heights[i]
        if i == 0:
            x = 0
            y = 0
        else:
            parent_idx = rng.randint(0, i - 1)
            parent = objects[parent_idx]
            min_x = parent.x - w + 1
            max_x = parent.x + parent.w - 1
            y = parent.y + parent.h
            placed = False
            for _ in range(24):
                x = rng.randint(min_x, max_x)
                candidate = ObjectState(id=object_id, x=x, y=y, w=w, h=h)
                if all(not _rectangles_overlap(candidate, existing) for existing in objects):
                    objects.append(candidate)
                    placed = True
                    break
            if not placed:
                x = parent.x
                while True:
                    candidate = ObjectState(id=object_id, x=x, y=y, w=w, h=h)
                    if all(not _rectangles_overlap(candidate, existing) for existing in objects):
                        objects.append(candidate)
                        break
                    x += 1
                continue
            continue
        objects.append(ObjectState(id=object_id, x=x, y=y, w=w, h=h))

    scene_intervention = Intervention(kind="none")
    if intervention == "remove":
        removable = [obj.id for obj in objects if obj.y != 0]
        remove_id = rng.choice(removable) if removable else objects[0].id
        scene_intervention = Intervention(kind="remove", object_id=remove_id)

    labels = _compute_labels(objects, scene_intervention)
    return Scene(objects=objects, intervention=scene_intervention, labels=labels)


def generate_dataset(
    n_scenes: int,
    split: Split,
    intervention: InterventionKind = "none",
    seed: int = 0,
) -> list[dict[str, object]]:
    """Generate deterministic support scenes with exact stable/falls labels."""
    if split not in {"id", "ood"}:
        raise ValueError(f"Unknown split: {split}")
    if intervention not in {"none", "remove"}:
        raise ValueError(f"Unknown intervention: {intervention}")

    rng = Random(seed)
    dataset: list[dict[str, object]] = []
    for _ in range(n_scenes):
        scene = _generate_scene(rng, split, intervention)
        dataset.append(
            {
                "objects": [obj.__dict__.copy() for obj in scene.objects],
                "intervention": scene.intervention.__dict__.copy(),
                "labels": scene.labels.copy(),
            }
        )
    return dataset
