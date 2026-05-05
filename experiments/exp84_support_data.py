"""Deterministic support/stability V1 object-table generator.

Pure geometry, exact labels, no learned components.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Obj:
    id: str
    x: float
    y: float
    w: float
    h: float


def _overlap_interval(a0: float, a1: float, b0: float, b1: float) -> Optional[Tuple[float, float]]:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return (lo, hi)


def _merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    out = sorted(intervals)
    merged = [out[0]]
    for lo, hi in out[1:]:
        mlo, mhi = merged[-1]
        if lo <= mhi:
            merged[-1] = (mlo, max(mhi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _covers_width(intervals: Sequence[Tuple[float, float]], left: float, right: float) -> bool:
    merged = _merge_intervals(intervals)
    if not merged:
        return False
    pos = left
    for lo, hi in merged:
        if lo > pos:
            return False
        pos = max(pos, hi)
        if pos >= right:
            return True
    return pos >= right


def compute_labels(objects: Sequence[Dict[str, float]], intervention: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    objs = [Obj(id=o["id"], x=float(o["x"]), y=float(o["y"]), w=float(o["w"]), h=float(o["h"])) for o in objects]
    removed = intervention and intervention.get("type") == "remove"
    removed_id = intervention.get("object_id") if removed else None

    if removed_id is not None and removed_id not in {o.id for o in objs}:
        raise ValueError(f"unknown remove target: {removed_id}")

    active = [o for o in objs if o.id != removed_id]
    stable: Dict[str, bool] = {o.id: False for o in active}

    changed = True
    while changed:
        changed = False
        for o in active:
            if stable[o.id]:
                continue
            if o.y == 0:
                stable[o.id] = True
                changed = True
                continue
            support_intervals = []
            for s in active:
                if s.id == o.id or not stable[s.id]:
                    continue
                if abs((s.y + s.h) - o.y) > 1e-9:
                    continue
                overlap = _overlap_interval(o.x, o.x + o.w, s.x, s.x + s.w)
                if overlap is not None:
                    support_intervals.append(overlap)
            if _covers_width(support_intervals, o.x, o.x + o.w):
                stable[o.id] = True
                changed = True

    labels = {o.id: ("stable" if stable.get(o.id, False) else "falls") for o in active}
    if removed_id is not None:
        labels[removed_id] = "falls"
    return labels


def _obj(i: int, x: float, y: float, w: float, h: float) -> Dict[str, float]:
    return {"id": f"o{i}", "x": x, "y": y, "w": w, "h": h}


def _gen_id_scene(rng: random.Random) -> Dict[str, object]:
    n = rng.choice([2, 3])
    base_w = rng.choice([2.0, 3.0])
    h = 1.0
    x = 0.0
    objs = [_obj(0, x, 0.0, base_w, h)]
    for i in range(1, n):
        w = max(1.0, base_w - 0.5 * i)
        cx = x + (base_w - w) / 2.0
        objs.append(_obj(i, cx, float(i), w, h))
    return {"objects": objs, "split": "id", "intervention": {"type": "none"}}


def _gen_ood_scene(rng: random.Random) -> Dict[str, object]:
    n = rng.randint(4, 8)
    objs: List[Dict[str, float]] = []
    # two base supports for branching potential
    left_w = rng.choice([1.5, 2.5, 3.5])
    right_w = rng.choice([1.5, 2.5, 3.0])
    objs.append(_obj(0, 0.0, 0.0, left_w, 1.0))
    objs.append(_obj(1, left_w, 0.0, right_w, 1.0))
    next_id = 2
    # bridge block supported by both branches
    bridge_w = left_w + right_w
    objs.append(_obj(next_id, 0.0, 1.0, bridge_w, 1.0))
    next_id += 1
    top_y = 2.0
    while next_id < n:
        prev = objs[-1]
        w = rng.choice([0.8, 1.2, 1.8, 2.2])
        h = rng.choice([0.8, 1.0, 1.4])
        x = prev["x"] + max(0.0, (prev["w"] - w) / 2.0)
        objs.append(_obj(next_id, x, top_y, w, h))
        top_y += h
        next_id += 1
    return {"objects": objs, "split": "ood", "intervention": {"type": "none"}}


def generate_dataset(num_scenes: int, split: str, seed: int, interventions: Sequence[str] = ("none", "remove")) -> List[Dict[str, object]]:
    if split not in {"id", "ood"}:
        raise ValueError("split must be 'id' or 'ood'")
    rng = random.Random(seed)
    scenes: List[Dict[str, object]] = []
    for _ in range(num_scenes):
        scene = _gen_id_scene(rng) if split == "id" else _gen_ood_scene(rng)
        objects = scene["objects"]
        for mode in interventions:
            if mode == "none":
                intervention = {"type": "none"}
            elif mode == "remove":
                victim = rng.choice(objects)["id"]
                intervention = {"type": "remove", "object_id": victim}
            else:
                raise ValueError(f"unknown intervention mode: {mode}")
            scene_objects = [dict(obj) for obj in objects]
            labels = compute_labels(scene_objects, intervention)
            scenes.append({"objects": scene_objects, "split": split, "intervention": dict(intervention), "labels": labels})
    return scenes
