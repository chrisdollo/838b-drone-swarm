"""Target shapes from images and text.

Both return an [m, 2] point cloud in the same convention as shapes.py
(centered, larger side scaled to 0.8), so any drawing or word can be a target.

  points_from_image("data/images/snowflake.jpg", 256, crop=(0, 0, 1, 0.85))
  points_from_text("MARRY\\nME", 256)
"""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from swarm.shapes import normalize, sample_polyline


def farthest_point_sampling(points: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """Pick m of the points so they spread evenly: each new point is the one farthest from all picked so far."""
    rng = np.random.default_rng(seed)
    m = min(m, len(points))
    chosen = [int(rng.integers(len(points)))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(m - 1):
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(points - points[i], axis=1))
    return points[chosen]


def edge_pixels(path, crop=None, max_side: int = 400, blur: float = 1.0, quantile: float = 0.9,
                min_component: int = 0) -> np.ndarray:
    """Pixel coordinates (x, y up) of the strongest edges in an image.

    crop: (left, top, right, bottom) as fractions of the image, e.g. to cut a watermark.
    quantile: keep pixels whose edge strength is above this quantile.
    min_component: drop connected edge blobs smaller than this many pixels (specks, texture).
    """
    im = Image.open(path).convert("L")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    scale = max_side / max(im.size)
    im = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))))

    g = ndimage.gaussian_filter(np.asarray(im, dtype=float) / 255.0, blur)
    strength = np.hypot(ndimage.sobel(g, axis=0), ndimage.sobel(g, axis=1))   # Sobel edge magnitude
    mask = strength > np.quantile(strength, quantile)
    if min_component:
        labels, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = np.isin(labels, 1 + np.flatnonzero(sizes >= min_component))
    ys, xs = np.nonzero(mask)
    return np.stack([xs, -ys], axis=1).astype(float)       # flip y: image rows grow downward


def points_from_image(path, m: int, seed: int = 0, **edge_kwargs) -> torch.Tensor:
    """[m, 2] target points spread evenly over the image's edges (see edge_pixels for options)."""
    pts = farthest_point_sampling(edge_pixels(path, **edge_kwargs), m, seed)
    return normalize(torch.tensor(pts, dtype=torch.float32))


def text_outlines(text: str, family: str = "DejaVu Sans", weight: str = "bold", line_gap: float = 1.25):
    """Letter outlines as a list of closed polygons. '\\n' starts a new, centered line."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(family=family, weight=weight)
    lines = text.split("\n")
    polys = []
    for row, line in enumerate(lines):
        if not line.strip():
            continue
        tp = TextPath((0, 0), line, size=1.0, prop=prop)
        width = tp.get_extents().width
        for poly in tp.to_polygons():
            poly = np.asarray(poly, dtype=float)
            poly[:, 0] -= width / 2                        # center each line
            poly[:, 1] -= row * line_gap                   # stack lines downward
            polys.append(poly)
    return polys


def points_from_text(text: str, m: int, seed: int = 0, **font_kwargs) -> torch.Tensor:
    """[m, 2] target points spread evenly along the outlines of the letters."""
    polys = text_outlines(text, **font_kwargs)
    lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in polys]
    total = sum(lengths)
    dense = [sample_polyline(p, max(8, int(20 * m * L / total)), closed=True).numpy()
             for p, L in zip(polys, lengths)]
    pts = farthest_point_sampling(np.concatenate(dense), m, seed)
    return normalize(torch.tensor(pts, dtype=torch.float32))
