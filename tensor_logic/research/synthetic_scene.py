import numpy as np
import torch

from tensor_logic.research.utils import apply_body


CANVAS = 64
OBJ_SIZE = 10
OBJ_NAMES = ["R", "G", "B"]
COLORS = [(220, 50, 50), (50, 200, 50), (50, 50, 220)]
RELATIONS = ["above", "left_of", "touching", "occluded"]
N_OBJ = len(OBJ_NAMES)
PAIRS = [(a, b) for a in OBJ_NAMES for b in OBJ_NAMES if a != b]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}
OBJ_IDX = {n: i for i, n in enumerate(OBJ_NAMES)}
N_PAIRS = len(PAIRS)


def generate_colored_object_sequence(n_frames=8, rng=None):
    """Return frames and positions for the shared colored-object scene."""
    if rng is None:
        rng = np.random.default_rng()
    half = OBJ_SIZE // 2
    pos = rng.integers(half + 4, CANVAS - half - 4, size=(N_OBJ, 2)).astype(float)
    vel = rng.uniform(-3.0, 3.0, size=(N_OBJ, 2))

    frames_np, pos_hist = [], []
    for _ in range(n_frames):
        pos += vel
        for i in range(N_OBJ):
            for d in range(2):
                if pos[i, d] < half + 1:
                    pos[i, d] = half + 1.0
                    vel[i, d] = abs(vel[i, d])
                elif pos[i, d] > CANVAS - half - 1:
                    pos[i, d] = float(CANVAS - half - 1)
                    vel[i, d] = -abs(vel[i, d])
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        for color, p in zip(COLORS, pos):
            x, y = int(round(p[0])), int(round(p[1]))
            canvas[
                max(0, y - half) : min(CANVAS, y + half),
                max(0, x - half) : min(CANVAS, x + half),
            ] = color
        frames_np.append(canvas.copy())
        pos_hist.append(pos.copy())

    frames = torch.from_numpy(np.stack(frames_np)).permute(0, 3, 1, 2).float() / 255.0
    return frames, np.array(pos_hist)


def compute_pairwise_spatial_relations(pos):
    """Return all pairwise colored-object spatial facts for a scene."""
    rels = {}
    for i, a in enumerate(OBJ_NAMES):
        for j, b in enumerate(OBJ_NAMES):
            if i == j:
                continue
            pi, pj = pos[i], pos[j]
            rels[("above", a, b)] = bool(pi[1] < pj[1])
            rels[("left_of", a, b)] = bool(pi[0] < pj[0])
            gap_x = abs(pi[0] - pj[0]) - OBJ_SIZE
            gap_y = abs(pi[1] - pj[1]) - OBJ_SIZE
            rels[("touching", a, b)] = bool(max(gap_x, gap_y) < 2)
            bbox = abs(pi[0] - pj[0]) < OBJ_SIZE and abs(pi[1] - pj[1]) < OBJ_SIZE
            rels[("occluded", a, b)] = bool(bbox and i > j)
    return rels


def generate_labeled_scene_frames(n=500, seed=99):
    """Return a list of single-frame colored-object scenes with positions."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        frames, pos_hist = generate_colored_object_sequence(n_frames=1, rng=rng)
        out.append((frames[0], pos_hist[0]))
    return out


def build_spatial_tl_state(fact_dict):
    """Build base and derived TL tensors from pairwise spatial facts."""
    base = {r: torch.zeros(N_OBJ, N_OBJ) for r in RELATIONS}
    for (rel_name, a, b), val in fact_dict.items():
        if val and rel_name in base:
            base[rel_name][OBJ_IDX[a], OBJ_IDX[b]] = 1.0

    blocked_path = apply_body(["touching", "above"], base)
    blocked_path.fill_diagonal_(0)
    same_side = apply_body(["left_of", "left_of"], base)
    clear_above = 1.0 - base["above"].max(dim=0)[0]

    return {
        **base,
        "blocked_path": blocked_path,
        "same_side": same_side,
        "clear_above": clear_above,
    }


def remove_object_facts(fact_dict, obj):
    """Return facts with every relation touching obj removed."""
    return {k: v for k, v in fact_dict.items() if k[1] != obj and k[2] != obj}


def compute_ground_truth_spatial_derivations(pos, exclude_obj=None):
    """Compute derived spatial relations directly from geometry."""
    blocked = torch.zeros(N_OBJ, N_OBJ)
    same = torch.zeros(N_OBJ, N_OBJ)
    clear = torch.ones(N_OBJ)
    active = [i for i, nm in enumerate(OBJ_NAMES) if nm != exclude_obj]

    for xi in active:
        for zi in active:
            if xi == zi:
                continue
            for yi in active:
                if yi == xi or yi == zi:
                    continue
                if _touching(pos[xi], pos[yi]) and _above(pos[yi], pos[zi]):
                    blocked[xi, zi] = 1.0
                if _left_of(pos[xi], pos[yi]) and _left_of(pos[yi], pos[zi]):
                    same[xi, zi] = 1.0
        for yi in active:
            if yi != xi and _above(pos[yi], pos[xi]):
                clear[xi] = 0.0

    return {"blocked_path": blocked, "same_side": same, "clear_above": clear}


def _touching(pi, pj):
    return max(abs(pi[0] - pj[0]) - OBJ_SIZE, abs(pi[1] - pj[1]) - OBJ_SIZE) < 2


def _above(pi, pj):
    return pi[1] < pj[1]


def _left_of(pi, pj):
    return pi[0] < pj[0]
