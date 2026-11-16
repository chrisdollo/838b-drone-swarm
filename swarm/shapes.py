"""Target shapes as point clouds.

The world is the box [-0.5, 0.5]^dim. Every shape is centered at the origin
and scaled so its larger side spans EXTENT = 0.8, which leaves a 0.1 margin to
the walls. A shape is just a set of M points: the loss never needs to know
which shape it is, so letters or user drawings work the same way as a circle.
"""

import math

import torch

EXTENT = 0.8  # side length the shape is scaled to fit


def normalize(points: torch.Tensor) -> torch.Tensor:
    """Center the bounding box on the origin and scale its larger side to EXTENT."""
    lo, hi = points.min(dim=0).values, points.max(dim=0).values
    size = (hi - lo).max()
    if size <= 0:
        raise ValueError("shape has zero size")
    return (points - (lo + hi) / 2) * (EXTENT / size)


def sample_polyline(vertices, m: int, closed: bool = False) -> torch.Tensor:
    """Return m points spaced evenly by arc length along a polyline.

    Even spacing by arc length (not per segment) means a long edge gets more
    points than a short one, so the shape is covered uniformly.
    closed=True joins the last vertex back to the first (square, letter O).
    """
    v = torch.as_tensor(vertices, dtype=torch.float32)
    if closed:
        v = torch.cat([v, v[:1]])
    seg = v[1:] - v[:-1]                       # [S, 2] segment vectors
    seg_len = seg.norm(dim=1)
    keep = seg_len > 1e-12                     # drop repeated vertices
    starts, seg, seg_len = v[:-1][keep], seg[keep], seg_len[keep]
    if len(seg) == 0:
        raise ValueError("polyline has zero length")

    cum = torch.cat([torch.zeros(1), seg_len.cumsum(0)])   # arc length at each vertex
    total = cum[-1]
    if closed:
        s = torch.arange(m) * (total / m)      # stop one gap short: the end equals the start
    else:
        s = torch.linspace(0, float(total), m)  # include both endpoints

    idx = (torch.searchsorted(cum, s, right=True) - 1).clamp(0, len(seg) - 1)
    frac = (s - cum[idx]) / seg_len[idx]       # position along that segment, 0..1
    return starts[idx] + frac[:, None] * seg[idx]


def sample_shape(name: str, m: int, dim: int = 2, vertices=None, closed: bool = False) -> torch.Tensor:
    """Return [m, dim] target points for a named shape.

    name: "circle", "square", "line", or "polyline" (pass vertices, and
    closed=True for a closed outline). For dim > 2 the shape lies flat in the
    z = 0 plane, so 3D runs work without new shapes.
    """
    if name == "circle":
        theta = 2 * math.pi * torch.arange(m) / m
        pts = (EXTENT / 2) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    elif name == "square":
        pts = normalize(sample_polyline([[-1, -1], [1, -1], [1, 1], [-1, 1]], m, closed=True))
    elif name == "line":
        pts = normalize(sample_polyline([[-1, 0], [1, 0]], m))
    elif name == "polyline":
        if vertices is None:
            raise ValueError("polyline needs vertices")
        pts = normalize(sample_polyline(vertices, m, closed=closed))
    else:
        raise ValueError(f"unknown shape: {name!r}")

    if dim > 2:
        pts = torch.cat([pts, torch.zeros(m, dim - 2)], dim=1)
    return pts
