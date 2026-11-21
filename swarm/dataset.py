"""Scenes with one goal per drone (v3): the training generator and the fixed benchmark.

A scene = N drones at random start positions + a target shape laid out as
exactly N goal slots by swarm/layout.py (a slot at every stroke end, corner and
junction, the rest spread evenly) + a collision distance scaled to the scene.

Every scene must pass the feature-size rule (the swarm has to be able to draw
the shape at all): two parts of the shape that are more than GAP_ARC drone
spacings apart along the strokes must be at least GAP_MIN spacings apart in
space. This rejects double outlines, letters too close together, and details
finer than the drones. Text that fails it is redrawn ("readable" strings only).

    s     = outline length / N          (drone spacing if spread evenly)
    d_min = D_MIN_FRAC * s              (collision distance)

Seeds (fixed so every run and every baseline sees the same scenes):
    BENCH_SEED = 2026           benchmark cell for N = BENCH_N[i], split j uses BENCH_SEED + 100 i + j
    HELDOUT_STRING_SEED = 838   drew HELDOUT_STRINGS (test_dataset.py checks they still match)
"""

import json
import math
import string
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from swarm.layout import layout, text_strokes
from swarm.shapes import sample_polyline

EXTENT = 0.8                   # larger side of a shape at full size (box is [-0.5, 0.5])
D_MIN_FRAC = 0.4               # collision distance = 0.4 x drone spacing
GAP_ARC, GAP_MIN = 6.0, 1.0    # feature-size rule, in drone spacings (1.0: readable, not just drawable).
                               # 6 along-the-shape spacings ignores corners down to ~20 deg (the layout
                               # handles those) but still catches parallel strokes, close letters, spiral turns.

N_RANGE = (8, 96)                         # training: drones (= goals) drawn uniformly per batch
BENCH_N = [8, 16, 24, 32, 48, 64, 96]     # benchmark cells
BENCH_SEED = 2026
BENCH_DIR = "data/benchmark_v3"

TRAIN_FAMILIES = ["circle", "square", "line", "triangle", "pentagon", "random_polygon", "random_polyline",
                  "letters", "random_string"]
HELDOUT_FAMILIES = ["star", "heart", "spiral", "word", "heldout_string"]
# Loss study: geometric shapes only, no text
SHAPE_TRAIN = ["circle", "square", "line", "triangle", "pentagon", "random_polygon", "random_polyline"]
SHAPE_HELDOUT = ["star", "heart", "spiral"]

ALPHABET = string.ascii_uppercase + string.digits
# Held-out words. Fewest drones that keep them readable: HI 10, UMD 26, FIRE 38, LOVE 39, MARRY ME 92.
WORDS = ["HI", "UMD", "LOVE", "FIRE", "MARRY\nME"]
HELDOUT_STRING_SEED = 838
HELDOUT_STRINGS = ["8", "A", "63", "22", "NY4", "ODP", "PH0", "0TO", "6F2L", "G", "CS", "W"]


def draw_heldout_strings(seed: int = HELDOUT_STRING_SEED, count: int = 12) -> list[str]:
    """How HELDOUT_STRINGS was drawn: random length 1-4 from A-Z 0-9, skipping the held-out words."""
    rng = np.random.default_rng(seed)
    words = {w for line in WORDS for w in line.split("\n")}
    out = []
    while len(out) < count:
        s = "".join(rng.choice(list(ALPHABET), int(rng.integers(1, 5))))
        if s not in words and s not in out:
            out.append(s)
    return out


# Never generated for training: the held-out strings and every held-out word (MARRY ME also as MARRY, ME).
EXCLUDED = set(HELDOUT_STRINGS) | {w for line in WORDS for w in line.split("\n")}


def random_text(rng: np.random.Generator, lengths=(1, 4), alphabet=ALPHABET) -> str:
    """A random training string, never one of the held-out strings or words."""
    while True:
        s = "".join(rng.choice(list(alphabet), int(rng.integers(lengths[0], lengths[1] + 1))))
        if s not in EXCLUDED:
            return s


# --- small hand-made single-stroke font (layout.py's 'simple' font; training text uses Hershey): each letter is (width, [(points, closed), ...]) on a height-1 grid -----

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
# --- shape families --------------------------------------------------------------------------

def _poly(n, radii=(1.0,), phase=math.pi / 2):
    k = np.arange(n)
    r = np.asarray(radii)[k % len(radii)]
    a = phase + 2 * math.pi * k / n
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def base_strokes(family: str, rng: np.random.Generator):
    """Untransformed strokes for one shape of a family, whether it must stay upright, and its text (or None)."""
    if family == "circle":
        return [(_poly(96), True)], False, None
    if family == "square":
        return [(_poly(4, phase=math.pi / 4), True)], False, None
    if family == "line":
        return [(np.array([[-1.0, 0], [1, 0]]), False)], False, None
    if family == "triangle":
        return [(_poly(3), True)], False, None
    if family == "pentagon":
        return [(_poly(5), True)], False, None
    if family == "star":
        return [(_poly(10, radii=(1.0, 0.45)), True)], False, None
    if family == "heart":
        t = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return [(np.stack([x, y], 1), True)], True, None
    if family == "spiral":
        t = np.linspace(0.6 * math.pi, 4.5 * math.pi, 160)
        return [(np.stack([t * np.cos(t), t * np.sin(t)], 1), False)], False, None
    if family == "random_polygon":
        n = int(rng.integers(3, 8))
        a = np.sort(rng.random(n)) * 2 * math.pi
        r = 0.5 + 0.5 * rng.random(n)
        return [(np.stack([r * np.cos(a), r * np.sin(a)], 1), True)], False, None
    if family == "random_polyline":
        n = int(rng.integers(3, 6))
        return [(rng.random((n, 2)) * 2 - 1, False)], False, None
    if family in ("letters", "random_string", "word", "heldout_string"):
        text = {"letters": lambda: random_text(rng, (1, 2), string.ascii_uppercase),
                "random_string": lambda: random_text(rng, (1, 4)),
                "word": lambda: WORDS[int(rng.integers(len(WORDS)))],
                "heldout_string": lambda: HELDOUT_STRINGS[int(rng.integers(len(HELDOUT_STRINGS)))]}[family]()
        return text_strokes(text), True, text            # Hershey Simplex, uppercase
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


# --- geometry: length, feature-size check ----------------------------------------------------

def stroke_length(p, closed):
    seg = np.diff(np.vstack([p, p[:1]]) if closed else p, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def total_length(strokes):
    return sum(stroke_length(p, c) for p, c in strokes)


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


def make_scene(family: str, n: int, rng: np.random.Generator, tries: int = 50):
    """One feasible scene with n drones and n goal slots, or None if this family can't be drawn by n drones.

    A drawn shape is rejected (and redrawn) if it fails the feature-size rule
    or has more corners/ends than n drones (a slot is required at each).
    """
    for _ in range(tries):
        strokes, upright, text = base_strokes(family, rng)
        strokes = transform(strokes, rng, upright)
        if feature_ratio(strokes, n) < GAP_MIN:
            continue
        goals, info = layout(strokes, n=n)
        if goals is None or len(goals) != n:
            continue
        L = total_length(strokes)
        anchor = np.zeros(n, bool)
        anchor[:info["anchors"]] = True              # layout lists the anchor slots first
        return {"family": family, "text": text or "", "goals": goals, "anchor": anchor, "length": L,
                "d_min": D_MIN_FRAC * L / n, "start": rng.random((n, 2)) - 0.5}
    return None


def text_scene(text: str, n: int, seed: int = 0) -> dict | None:
    """One scene for a given text (demos, held-out words), as tensors with a batch dimension of 1.

    Same pipeline as training text: Hershey strokes, a random upright transform,
    layout with n slots. None if n drones can't draw the text readably.
    """
    rng = np.random.default_rng(seed)
    strokes = transform(text_strokes(text), rng, upright=True)
    if feature_ratio(strokes, n) < GAP_MIN:
        return None
    goals, info = layout(strokes, n=n)
    if goals is None or len(goals) != n:
        return None
    anchor = torch.zeros(1, n, dtype=torch.bool)
    anchor[0, :info["anchors"]] = True
    L = total_length(strokes)
    return {"start": torch.tensor(rng.random((1, n, 2)) - 0.5, dtype=torch.float32),
            "goals": torch.tensor(goals[None], dtype=torch.float32),
            "d_min": torch.tensor([D_MIN_FRAC * L / n], dtype=torch.float32), "anchor": anchor}


def feasible(family: str, n: int, rng=None, tries: int = 50) -> bool:
    return make_scene(family, n, rng or np.random.default_rng(0), tries) is not None


def sample_n(rng: np.random.Generator, n_range=N_RANGE) -> int:
    return int(rng.integers(n_range[0], n_range[1] + 1))


def make_batch(batch: int, n: int, rng: np.random.Generator, families=TRAIN_FAMILIES, device="cpu"):
    """Training batch with n drones and n goals per scene: pos0 [B,N,2], vel0, Q [B,N,2], d_min [B].

    Families that can't be drawn by n drones are redrawn from another family.
    """
    scenes = []
    while len(scenes) < batch:
        sc = make_scene(families[int(rng.integers(len(families)))], n, rng, tries=5)
        if sc is not None:
            scenes.append(sc)
    t = lambda k: torch.tensor(np.stack([s[k] for s in scenes]), dtype=torch.float32, device=device)  # noqa: E731
    pos0 = t("start")
    return pos0, torch.zeros_like(pos0), t("goals"), t("d_min")


# --- the fixed benchmark --------------------------------------------------------------------

def build_cell(n: int, families, per_family: int, seed: int):
    """Fixed scenes for one cell: per_family scenes of every family that n drones can draw."""
    rng = np.random.default_rng(seed)
    scenes, skipped = [], []
    for fam in families:
        got = [sc for sc in (make_scene(fam, n, rng) for _ in range(per_family)) if sc is not None]
        if len(got) < per_family:
            skipped.append(fam)          # can't be drawn with this few drones
            continue
        scenes += got
    stack = lambda k: torch.tensor(np.stack([s[k] for s in scenes]))  # noqa: E731
    return {"n": n, "m": n, "family": [s["family"] for s in scenes], "text": [s["text"] for s in scenes],
            "skipped": skipped, "start": stack("start").float(), "goals": stack("goals").float(),
            "d_min": stack("d_min").float(), "length": stack("length").float(), "anchor": stack("anchor")}


def oracle(cell: dict, cfg, iters: int = 300, lr: float = 0.05) -> dict:
    """Per-scene optimum: the Phase 1 method applied to every scene of a cell.

    Optimizes each scene's own accelerations directly, with the same collision
    penalty as policy training (weight cfg.loss.w_collision, the scene's d_min),
    so the reference plays by the same rules. All scenes are optimized together;
    their losses don't interact and Adam scales each parameter separately.
    Like the policy's loss it uses Chamfer (no assignment), so it shows what
    Chamfer can reach, and is scored with the same slot metrics.
    """
    from swarm.losses import chamfer, close_pairs
    from swarm.metrics import slot_metrics
    from swarm.plan import optimize_plan

    out = optimize_plan(cell["start"], cell["goals"], cfg, iters=iters, lr=lr,
                        d_min=cell["d_min"], w_collision=cfg.loss.w_collision)
    final = out["traj"][:, -1]
    slots = slot_metrics(final, cell["goals"], cell["d_min"], cell["anchor"])
    return {"oracle_chamfer": chamfer(final, cell["goals"], per_scene=True), "oracle_final": final,
            "oracle_close_pairs": close_pairs(final, cell["d_min"]),
            **{f"oracle_{k}": v for k, v in slots.items()},
            "oracle_traj_head": out["traj"][:4]}          # first 4 trajectories, for GIFs


def build_benchmark(cfg, out_dir=BENCH_DIR, per_family: int = 4, seed: int = BENCH_SEED, log=print):
    """Build and save every cell for both splits ('train' shapes and 'heldout' shapes)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for ci, n in enumerate(BENCH_N):
        for si, (split, fams) in enumerate([("train", TRAIN_FAMILIES), ("heldout", HELDOUT_FAMILIES)]):
            cell_seed = seed + 100 * ci + si
            cell = build_cell(n, fams, per_family, cell_seed)
            cell.update(oracle(cell, cfg))
            cell.update(split=split, seed=cell_seed)
            torch.save(cell, out / f"n{n}_{split}.pt")
            row = {"n": n, "split": split, "seed": cell_seed, "scenes": len(cell["family"]),
                   "skipped": cell["skipped"], "texts": sorted({t for t in cell["text"] if t}),
                   "oracle_chamfer": float(cell["oracle_chamfer"].mean()),
                   "oracle_slot_error": float(cell["oracle_slot_error"].mean()),
                   "oracle_filled": float(cell["oracle_filled"].mean()),
                   "oracle_close_pairs": float(cell["oracle_close_pairs"].mean()),
                   "d_min_mean": float(cell["d_min"].mean())}
            index.append(row)
            log({k: v for k, v in row.items() if k != "texts"})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def load_cell(n: int, split: str = "train", root=BENCH_DIR, families=None) -> dict:
    """One benchmark cell; families: keep only scenes of these families (e.g. SHAPE_TRAIN)."""
    cell = torch.load(Path(root) / f"n{n}_{split}.pt", weights_only=False)
    return cell if families is None else subset(cell, families)


def subset(cell: dict, families) -> dict:
    """The scenes of a cell whose family is in `families` (same seeds, same starts)."""
    keep = [i for i, f in enumerate(cell["family"]) if f in families]
    idx = torch.tensor(keep, dtype=torch.long)
    out = {}
    for k, v in cell.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == len(cell["family"]):
            out[k] = v[idx]
        elif isinstance(v, list) and len(v) == len(cell["family"]):
            out[k] = [v[i] for i in keep]
        else:
            out[k] = v
    return out
