"""Scenes on a drones x goals grid: the training generator and the fixed benchmark.

A scene = N drones at random start positions + a target shape sampled as M goal
points + a collision distance scaled to the scene's density.

Shapes are lists of strokes (polylines). Every scene must pass a feature-size
check: the swarm has to be able to draw the shape at all. Two parts of a shape
that are far apart along the outline (more than GAP_ARC drone spacings) must
also be apart in space (at least GAP_MIN drone spacings). This rejects
double outlines, letters too close together, and details finer than the drones.

    s     = outline length / N          (drone spacing if spread evenly)
    d_min = D_MIN_FRAC * s              (collision distance)
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from swarm.shapes import sample_polyline

EXTENT = 0.8                   # larger side of a shape at full size (box is [-0.5, 0.5])
D_MIN_FRAC = 0.4               # collision distance = 0.4 x drone spacing
GAP_ARC, GAP_MIN = 3.0, 0.8    # feature-size rule, in drone spacings

# (drones N, goals M)
GRID = [(8, 8), (8, 16), (8, 32), (8, 64),
        (16, 16), (16, 32), (16, 64),
        (32, 32), (32, 64),
        (64, 64), (64, 128)]

TRAIN_FAMILIES = ["circle", "square", "line", "triangle", "pentagon", "random_polygon", "random_polyline", "letters"]
HELDOUT_FAMILIES = ["star", "heart", "spiral", "word"]
WORDS = ["MARRY\nME", "UMD", "HI", "SWARM", "LOVE", "FIRE"]


# --- single-stroke font: each letter is (width, [(points, closed), ...]) on a height-1 grid -----

def _o(*pts):
    return (list(pts), False)


def _c(*pts):
    return (list(pts), True)


_P = [(0, 0), (0, 1), (.45, 1), (.65, .85), (.65, .65), (.45, .5), (0, .5)]
_C = [(.7, .85), (.5, 1), (.2, 1), (0, .75), (0, .25), (.2, 0), (.5, 0), (.7, .15)]
FONT = {
    "A": (.8, [_o((0, 0), (.4, 1), (.8, 0)), _o((.2, .5), (.6, .5))]),
    "B": (.65, [_o((0, 0), (0, 1), (.45, 1), (.6, .88), (.6, .62), (.45, .5), (0, .5)),
                _o((.45, .5), (.65, .38), (.65, .12), (.45, 0), (0, 0))]),
    "C": (.7, [_o(*_C)]),
    "D": (.7, [_c((0, 0), (0, 1), (.4, 1), (.7, .75), (.7, .25), (.4, 0))]),
    "E": (.6, [_o((.6, 1), (0, 1), (0, 0), (.6, 0)), _o((0, .5), (.45, .5))]),
    "F": (.6, [_o((.6, 1), (0, 1), (0, 0)), _o((0, .5), (.45, .5))]),
    "G": (.7, [_o(*_C, (.7, .45), (.4, .45))]),
    "H": (.7, [_o((0, 0), (0, 1)), _o((.7, 0), (.7, 1)), _o((0, .5), (.7, .5))]),
    "I": (0., [_o((0, 0), (0, 1))]),
    "J": (.6, [_o((.6, 1), (.6, .2), (.4, 0), (.15, 0), (0, .2))]),
    "K": (.7, [_o((0, 0), (0, 1)), _o((.7, 1), (0, .4)), _o((.25, .6), (.7, 0))]),
    "L": (.6, [_o((0, 1), (0, 0), (.6, 0))]),
    "M": (.9, [_o((0, 0), (0, 1), (.45, .45), (.9, 1), (.9, 0))]),
    "N": (.7, [_o((0, 0), (0, 1), (.7, 0), (.7, 1))]),
    "O": (.7, [_c((.2, 0), (.5, 0), (.7, .25), (.7, .75), (.5, 1), (.2, 1), (0, .75), (0, .25))]),
    "P": (.65, [_o(*_P)]),
    "Q": (.75, [_c((.2, 0), (.5, 0), (.7, .25), (.7, .75), (.5, 1), (.2, 1), (0, .75), (0, .25)),
                _o((.45, .25), (.75, -.05))]),
    "R": (.65, [_o(*_P), _o((.3, .5), (.65, 0))]),
    "S": (.65, [_o((.65, .85), (.45, 1), (.2, 1), (0, .85), (0, .65), (.2, .5), (.45, .5), (.65, .35),
                   (.65, .15), (.45, 0), (.2, 0), (0, .15))]),
    "T": (.8, [_o((0, 1), (.8, 1)), _o((.4, 1), (.4, 0))]),
    "U": (.7, [_o((0, 1), (0, .25), (.2, 0), (.5, 0), (.7, .25), (.7, 1))]),
    "V": (.8, [_o((0, 1), (.4, 0), (.8, 1))]),
    "W": (1., [_o((0, 1), (.25, 0), (.5, .6), (.75, 0), (1, 1))]),
    "X": (.7, [_o((0, 0), (.7, 1)), _o((0, 1), (.7, 0))]),
    "Y": (.8, [_o((0, 1), (.4, .5), (.8, 1)), _o((.4, .5), (.4, 0))]),
    "Z": (.7, [_o((0, 1), (.7, 1), (0, 0), (.7, 0))]),
}
LETTER_GAP, LINE_GAP = 0.4, 1.5


def text_strokes(text: str):
    """Single-stroke text as strokes. '\\n' starts a new line; lines are centered."""
    strokes = []
    for row, line in enumerate(text.upper().split("\n")):
        advances = [(FONT[ch][0] + LETTER_GAP) if ch != " " else 0.6 for ch in line]
        width = sum(advances) - LETTER_GAP
        x = -width / 2
        for ch, adv in zip(line, advances):
            if ch != " ":
                for pts, closed in FONT[ch][1]:
                    p = np.asarray(pts, float) + [x, -row * LINE_GAP]
                    strokes.append((p, closed))
            x += adv
    return strokes


# --- shape families --------------------------------------------------------------------------

def _poly(n, radii=(1.0,), phase=math.pi / 2):
    k = np.arange(n)
    r = np.asarray(radii)[k % len(radii)]
    a = phase + 2 * math.pi * k / n
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def base_strokes(family: str, rng: np.random.Generator):
    """Untransformed strokes for one shape of a family, plus whether it must stay upright."""
    if family == "circle":
        return [(_poly(96), True)], False
    if family == "square":
        return [(_poly(4, phase=math.pi / 4), True)], False
    if family == "line":
        return [(np.array([[-1.0, 0], [1, 0]]), False)], False
    if family == "triangle":
        return [(_poly(3), True)], False
    if family == "pentagon":
        return [(_poly(5), True)], False
    if family == "star":
        return [(_poly(10, radii=(1.0, 0.45)), True)], False
    if family == "heart":
        t = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return [(np.stack([x, y], 1), True)], True
    if family == "spiral":
        t = np.linspace(0.6 * math.pi, 4.5 * math.pi, 160)
        return [(np.stack([t * np.cos(t), t * np.sin(t)], 1), False)], False
    if family == "random_polygon":
        n = int(rng.integers(3, 8))
        a = np.sort(rng.random(n)) * 2 * math.pi
        r = 0.5 + 0.5 * rng.random(n)
        return [(np.stack([r * np.cos(a), r * np.sin(a)], 1), True)], False
    if family == "random_polyline":
        n = int(rng.integers(3, 6))
        return [(rng.random((n, 2)) * 2 - 1, False)], False
    if family == "letters":
        letters = rng.choice(list(FONT), size=int(rng.integers(1, 3)))
        return text_strokes("".join(letters)), True
    if family == "word":
        return text_strokes(WORDS[int(rng.integers(len(WORDS)))]), True
    raise ValueError(f"unknown family: {family!r}")


def transform(strokes, rng, upright: bool):
    """Random rotation, stretch, size and position; result fits inside the box."""
    pts = np.concatenate([p for p, _ in strokes])
    if upright:      # text and hearts stay readable
        a = rng.uniform(-0.17, 0.17)
        stretch = rng.uniform(0.85, 1.0, 2)
        size = rng.uniform(0.8, 1.0)
    else:
        a = rng.uniform(0, 2 * math.pi)
        stretch = rng.uniform(0.5, 1.0, 2)
        size = rng.uniform(0.6, 1.0)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = (pts @ rot.T) * stretch
    lo, hi = pts.min(0), pts.max(0)
    scale = EXTENT * size / max((hi - lo).max(), 1e-9)
    half = (hi - lo) / 2 * scale
    shift = (rng.random(2) * 2 - 1) * (EXTENT / 2 - half)
    return [(((p @ rot.T) * stretch - (lo + hi) / 2) * scale + shift, closed) for p, closed in strokes]


# --- geometry: length, sampling, feature-size check -----------------------------------------

def stroke_length(p, closed):
    seg = np.diff(np.vstack([p, p[:1]]) if closed else p, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def total_length(strokes):
    return sum(stroke_length(p, c) for p, c in strokes)


def sample_strokes(strokes, m: int) -> np.ndarray:
    """m points spread evenly by length over all strokes (each stroke gets its share)."""
    lengths = np.array([stroke_length(p, c) for p, c in strokes])
    share = m * lengths / lengths.sum()
    counts = np.floor(share).astype(int)
    for i in np.argsort(-(share - counts))[: m - counts.sum()]:   # largest remainders
        counts[i] += 1
    parts = [sample_polyline(p, n, closed=c).numpy() for (p, c), n in zip(strokes, counts) if n > 0]
    return np.concatenate(parts)


def feature_ratio(strokes, n_drones: int) -> float:
    """Smallest gap between distant parts of the shape, in drone spacings.

    Samples the strokes densely, links consecutive points (and any points that
    touch, e.g. the bar of a T or the bowl of an R), and measures distance
    ALONG the shape with shortest paths. Among point pairs more than GAP_ARC
    drone spacings apart along the shape (or on unconnected strokes), returns
    the smallest straight-line distance divided by the drone spacing s.
    """
    s = total_length(strokes) / n_drones
    h = min(s / 4, 0.01)        # sample finely; fixed so 'touching' means really touching
    pts, edges, start = [], [], 0
    for p, closed in strokes:
        k = max(2, int(math.ceil(stroke_length(p, closed) / h)) + 1)
        q = sample_polyline(p, k, closed=closed).numpy()
        idx = np.arange(start, start + len(q))
        edges += list(zip(idx[:-1], idx[1:]))
        if closed:
            edges.append((idx[-1], idx[0]))
        pts.append(q)
        start += len(q)
    P = np.concatenate(pts)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    touch = np.argwhere(D < 1.5 * h)      # strokes that touch: T-junctions, the R bowl meeting its stem
    edges += [tuple(e) for e in touch]
    e = np.array(edges)
    w = D[e[:, 0], e[:, 1]] + 1e-12
    G = shortest_path(coo_matrix((w, (e[:, 0], e[:, 1])), shape=D.shape).tocsr(), method="D", directed=False)
    far = G > GAP_ARC * s
    return float(D[far].min() / s) if far.any() else math.inf


def make_scene(family: str, n: int, m: int, rng: np.random.Generator, tries: int = 50):
    """One feasible scene, or None if this family can't be drawn by n drones."""
    for _ in range(tries):
        strokes, upright = base_strokes(family, rng)
        strokes = transform(strokes, rng, upright)
        if feature_ratio(strokes, n) >= GAP_MIN:
            L = total_length(strokes)
            return {"family": family, "goals": sample_strokes(strokes, m), "length": L,
                    "d_min": D_MIN_FRAC * L / n, "start": rng.random((n, 2)) - 0.5}
    return None


def feasible(family: str, n: int, rng=None, tries: int = 50) -> bool:
    return make_scene(family, n, n, rng or np.random.default_rng(0), tries) is not None


def make_batch(batch: int, n: int, m: int, rng: np.random.Generator, families=TRAIN_FAMILIES, device="cpu"):
    """Training batch: pos0 [B,N,2], vel0, Q [B,M,2], d_min [B]. Families that can't fit n are skipped."""
    scenes = []
    while len(scenes) < batch:
        sc = make_scene(families[int(rng.integers(len(families)))], n, m, rng, tries=5)
        if sc is not None:
            scenes.append(sc)
    t = lambda k: torch.tensor(np.stack([s[k] for s in scenes]), dtype=torch.float32, device=device)  # noqa: E731
    pos0 = t("start")
    return pos0, torch.zeros_like(pos0), t("goals"), t("d_min")


# --- the fixed benchmark --------------------------------------------------------------------

def cell_name(n, m):
    return f"n{n}_m{m}"


def build_cell(n: int, m: int, families, per_family: int, seed: int):
    """Fixed scenes for one grid cell: per_family scenes of every family that fits n drones."""
    rng = np.random.default_rng(seed)
    scenes, skipped = [], []
    for fam in families:
        got = [sc for sc in (make_scene(fam, n, m, rng) for _ in range(per_family)) if sc is not None]
        if len(got) < per_family:
            skipped.append(fam)          # can't be drawn with this few drones
            continue
        scenes += got
    stack = lambda k: torch.tensor(np.stack([s[k] for s in scenes]), dtype=torch.float32)  # noqa: E731
    return {"n": n, "m": m, "family": [s["family"] for s in scenes], "skipped": skipped,
            "start": stack("start"), "goals": stack("goals"), "d_min": stack("d_min"),
            "length": stack("length")}


def oracle(cell: dict, cfg, iters: int = 300, lr: float = 0.05) -> dict:
    """Per-scene optimum (Phase 1 method): optimize every scene's own accelerations directly.

    All scenes of a cell are optimized together; their losses don't interact
    and Adam scales each parameter separately, so this equals S separate runs.
    """
    from swarm.losses import chamfer, close_pairs
    from swarm.sim import rollout

    p0, Q = cell["start"], cell["goals"]
    S, N, _ = p0.shape
    v0 = torch.zeros_like(p0)
    raw = torch.zeros(S, cfg.steps, N, 2, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        loss = chamfer(rollout(p0, v0, cfg.a_max * torch.tanh(raw), cfg)[:, -1], Q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        traj = rollout(p0, v0, cfg.a_max * torch.tanh(raw), cfg)
        final = traj[:, -1]
        return {"oracle_chamfer": chamfer(final, Q, per_scene=True), "oracle_final": final,
                "oracle_close_pairs": close_pairs(final, cell["d_min"]),
                "oracle_traj_head": traj[:4]}          # first 4 trajectories, for GIFs


def build_benchmark(cfg, out_dir="data/benchmark", per_family: int = 4, seed: int = 2026, log=print):
    """Build and save every grid cell for both splits ('train' shapes and 'heldout' shapes)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for ci, (n, m) in enumerate(GRID):
        for si, (split, fams) in enumerate([("train", TRAIN_FAMILIES), ("heldout", HELDOUT_FAMILIES)]):
            cell = build_cell(n, m, fams, per_family, seed + 100 * ci + si)
            cell.update(oracle(cell, cfg))
            cell["split"] = split
            torch.save(cell, out / f"{cell_name(n, m)}_{split}.pt")
            row = {"n": n, "m": m, "split": split, "scenes": len(cell["family"]), "skipped": cell["skipped"],
                   "oracle_chamfer": float(cell["oracle_chamfer"].mean()),
                   "oracle_close_pairs": float(cell["oracle_close_pairs"].mean()),
                   "d_min_mean": float(cell["d_min"].mean())}
            index.append(row)
            log(row)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def load_cell(n: int, m: int, split: str = "train", root="data/benchmark") -> dict:
    return torch.load(Path(root) / f"{cell_name(n, m)}_{split}.pt", weights_only=False)
