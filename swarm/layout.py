"""Goal layout: turn strokes into evenly spaced drone slots, drone-show style.

One goal per drone, about one "pitch" apart along the strokes, with a goal
always at every stroke end, corner and junction:

  1. anchors: stroke ends, corners (the stroke turns by more than CORNER_DEG)
     and junctions (a stroke end touching another stroke, e.g. the bar of an A);
     anchors closer than half a pitch are merged, so a junction gets one goal
  2. each stroke is cut at the anchors that lie on it, into pieces
  3. pieces are filled with evenly spaced goals:
       given a pitch p: a piece of length l gets round(l / p) - 1 goals inside
       given a count n: the n - (number of anchors) goals go, one by one, to
                        the piece with the largest gap, so all gaps end up even

Works for any strokes: shapes from swarm/dataset.py or text from any
single-stroke font (fonts below: Hershey Simplex, or the small hand-made one).

    goals, info = text_goals("MARRY ME", pitch=0.03)     # drone count follows from the text
    goals, info = text_goals("MARRY\\nME", n=96)          # or fix the drones, solve the spacing
"""

import heapq
import math

import numpy as np

EXTENT = 0.8        # text is scaled so its larger side spans 0.8 of the [-0.5, 0.5] box
CORNER_DEG = 35.0   # a vertex where the stroke turns more than this is a corner
MIN_SEP = 0.6       # fill dots closer than this many pitches to another dot are moved elsewhere


# --- fonts: each returns strokes for one line of text, y up, letter height ~1 -------------------

class HersheyFont:
    """Hershey single-stroke vector font (public-domain glyph data, via the Hershey-Fonts package).

    Default 'futural' = Hershey Simplex Roman, the classic plotter/engraving font.
    """

    def __init__(self, name: str = "futural"):
        from HersheyFonts import HersheyFonts
        self.font = HersheyFonts()
        self.font.load_default_font(name)
        ys = [y for s in self.font.strokes_for_text("H") for _, y in s]
        self.height = max(ys) - min(ys)              # cap height in font units
        self.top = min(ys)

    def line(self, text: str):
        strokes = []
        for s in self.font.strokes_for_text(text):
            p = np.asarray(s, dtype=float)
            p = np.stack([p[:, 0], -(p[:, 1] - self.top)], axis=1) / self.height   # y up, cap height 1
            p[:, 1] += 1.0
            closed = len(p) > 2 and np.allclose(p[0], p[-1])
            strokes.append((p[:-1] if closed else p, closed))
        return strokes


class SimpleFont:
    """The small hand-made single-stroke A-Z font (straight segments only)."""

    def __init__(self):
        from swarm.dataset import FONT
        self.glyphs = FONT

    def line(self, text: str, letter_gap: float = 0.4):
        strokes, x = [], 0.0
        for ch in text.upper():
            if ch == " ":
                x += 0.6
                continue
            width, glyph = self.glyphs[ch]
            strokes += [(np.asarray(p, float) + [x, 0.0], closed) for p, closed in glyph]
            x += width + letter_gap
        return strokes


FONTS = {}


def get_font(name: str = "hershey"):
    if name not in FONTS:
        FONTS[name] = HersheyFont() if name == "hershey" else SimpleFont() if name == "simple" else HersheyFont(name)
    return FONTS[name]


def text_strokes(text: str, font: str = "hershey", line_gap: float = 1.6):
    """Strokes for (multi-line) text, letter height 1, lines centered. Uppercase is used."""
    f = get_font(font)
    strokes = []
    for row, line in enumerate(text.upper().split("\n")):
        line_strokes = f.line(line)
        if not line_strokes:
            continue
        xs = np.concatenate([p[:, 0] for p, _ in line_strokes])
        mid = (xs.min() + xs.max()) / 2
        strokes += [(p - [mid, row * line_gap], c) for p, c in line_strokes]
    return strokes


def fit_to_box(strokes, extent: float = EXTENT):
    """Center the strokes and scale the larger side to `extent`."""
    pts = np.concatenate([p for p, _ in strokes])
    lo, hi = pts.min(0), pts.max(0)
    scale = extent / max((hi - lo).max(), 1e-12)
    return [((p - (lo + hi) / 2) * scale, c) for p, c in strokes]


# --- geometry helpers ---------------------------------------------------------------------------

def _closed_pts(p, closed):
    return np.vstack([p, p[:1]]) if closed else p


def _arclen(p, closed):
    q = _closed_pts(p, closed)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])


def _project(point, p, closed):
    """(distance, arc position) of the closest point of the stroke to `point`."""
    q = _closed_pts(p, closed)
    cum = _arclen(p, closed)
    a, b = q[:-1], q[1:]
    ab = b - a
    t = np.clip(((point - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-18), 0, 1)
    closest = a + t[:, None] * ab
    d = np.linalg.norm(closest - point, axis=1)
    i = int(d.argmin())
    return float(d[i]), float(cum[i] + t[i] * (cum[i + 1] - cum[i]))


def _point_at(p, closed, s):
    q = _closed_pts(p, closed)
    cum = _arclen(p, closed)
    s = np.clip(s, 0, cum[-1])
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(q) - 2)
    frac = (s - cum[i]) / np.maximum(cum[i + 1] - cum[i], 1e-18)
    return q[i] + frac[:, None] * (q[i + 1] - q[i])


def total_length(strokes):
    return float(sum(_arclen(p, c)[-1] for p, c in strokes))


# --- anchors and layout -------------------------------------------------------------------------

def find_anchors(strokes, merge: float, corner_deg: float = CORNER_DEG, touch: float = 0.0):
    """Stroke ends, corners and junctions, with anchors closer than `merge` merged into one."""
    cand = []
    for p, closed in strokes:
        if not closed:
            cand += [p[0], p[-1]]
        n = len(p)
        for i in (range(n) if closed else range(1, n - 1)):
            a, b = p[i] - p[i - 1], p[(i + 1) % n] - p[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                turn = math.degrees(math.acos(np.clip((a @ b) / (na * nb), -1, 1)))
                if turn > corner_deg:
                    cand.append(p[i])
    anchors = []
    for c in cand:                                   # greedy merge: keep the first, drop near duplicates
        if all(np.linalg.norm(c - a) >= max(merge, 1e-9) for a in anchors):
            anchors.append(np.asarray(c, float))
    # closed strokes with no anchor on them (a circle) get one at their start
    for p, closed in strokes:
        if closed and not any(_project(a, p, closed)[0] <= max(touch, 1e-9) for a in anchors):
            anchors.append(p[0].astype(float))
    return np.array(anchors)


def _pieces(strokes, anchors, touch):
    """Cut every stroke at the anchors lying on it. Returns [(stroke index, arc start, arc end)]."""
    pieces = []
    for k, (p, closed) in enumerate(strokes):
        L = _arclen(p, closed)[-1]
        cuts = sorted({round(s, 12) for a in anchors for d, s in [_project(a, p, closed)] if d <= touch})
        if not closed:
            cuts = sorted(set(cuts) | {0.0, round(L, 12)})
            pairs = list(zip(cuts[:-1], cuts[1:]))
        else:
            pairs = list(zip(cuts, cuts[1:] + [cuts[0] + L]))
        pieces += [(k, a, b) for a, b in pairs if b - a > 1e-9]
    return pieces


def layout(strokes, n: int | None = None, pitch: float | None = None, corner_deg: float = CORNER_DEG,
           touch: float | None = None):
    """Evenly spaced goals with an anchor at every end, corner and junction.

    Give n (number of goals = drones) or pitch (target spacing). Returns
    (goals [n, 2], info) with info: n, anchors, pitch, min_gap (smallest
    distance between two goals, in pitches). Returns (None, info) if n is
    smaller than the number of anchors (the shape needs more drones).
    Inside very sharp angles a piece may get fewer fill dots than its length
    suggests, so no two goals end up closer than MIN_SEP pitches (anchors aside).
    """
    assert (n is None) != (pitch is None), "give exactly one of n or pitch"
    L = total_length(strokes)
    p_est = pitch if pitch is not None else L / n
    if touch is None:
        touch = 0.02 * max(np.ptp(np.concatenate([p for p, _ in strokes]), axis=0))
    anchors = find_anchors(strokes, merge=0.5 * p_est, corner_deg=corner_deg, touch=touch)
    pieces = _pieces(strokes, anchors, touch)
    lengths = np.array([b - a for _, a, b in pieces])
    A = len(anchors)

    if n is not None and n < A:
        return None, {"n": n, "anchors": A, "needs_at_least": A}
    cap = np.full(len(pieces), 10**9)                 # max fill per piece (lowered near sharp V's)

    def distribute():
        if pitch is not None:
            return np.minimum(cap, np.maximum(0, np.round(lengths / pitch).astype(int) - 1))
        counts = np.zeros(len(pieces), dtype=int)
        heap = [(-l, i) for i, l in enumerate(lengths) if cap[i] > 0]   # largest gap first
        heapq.heapify(heap)
        for _ in range(n - A):
            if not heap:
                break
            _, i = heapq.heappop(heap)
            counts[i] += 1
            if counts[i] < cap[i]:
                heapq.heappush(heap, (-lengths[i] / (counts[i] + 1), i))
        return counts

    def place(counts):
        pts, owner = [anchors], [np.full(A, -1)]
        for i, ((k, a, b), c) in enumerate(zip(pieces, counts)):
            if c:
                sp = a + (b - a) * np.arange(1, c + 1) / (c + 1)
                pk, closed = strokes[k]
                pts.append(_point_at(pk, closed, sp % _arclen(pk, closed)[-1] if closed else sp))
                owner.append(np.full(c, i))
        return np.concatenate(pts), np.concatenate(owner)

    # Fill dots closer than MIN_SEP pitches to an already kept dot (inside sharp V's like the
    # top of an M) are dropped from their piece; in count mode the drone goes to the next widest gap.
    for _ in range(20):
        counts = distribute()
        goals, owner = place(counts)
        sep = MIN_SEP * L / max(A + counts.sum(), 1)
        keep, bad = [], np.zeros(len(pieces), dtype=int)
        for i, g in enumerate(goals):
            if owner[i] >= 0 and keep and np.min(np.linalg.norm(goals[keep] - g, axis=1)) < sep:
                bad[owner[i]] += 1
            else:
                keep.append(i)
        if not bad.any():
            break
        cap = np.where(bad > 0, np.maximum(0, counts - bad), cap)
    goals = goals[keep]
    m = len(goals)
    real_pitch = L / m
    d = np.linalg.norm(goals[:, None] - goals[None], axis=-1) + np.eye(m) * 1e9
    return goals, {"n": m, "anchors": A, "pitch": real_pitch, "min_gap": float(d.min() / real_pitch)}


def text_goals(text: str, n: int | None = None, pitch: float | None = None, font: str = "hershey",
               extent: float = EXTENT):
    """Goals for a word or sentence, fitted to the box. Give n (drones) or pitch (spacing, box units).

    info["readable"] is False when the drones are too few for the text's details
    (the feature-size rule of swarm/dataset.py): the layout still exists, but
    letters would blur together.
    """
    strokes = fit_to_box(text_strokes(text, font), extent)
    goals, info = layout(strokes, n=n, pitch=pitch)
    from swarm.dataset import GAP_MIN, feature_ratio      # the feature-size rule
    info["readable"] = feature_ratio(strokes, info["n"]) >= GAP_MIN
    return goals, info
