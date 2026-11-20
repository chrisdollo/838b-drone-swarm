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


def _regular_polygon(n: int, radii=(1.0,)) -> list:
    """Vertices of a regular polygon (or a star, when two radii alternate)."""
    return [[radii[k % len(radii)] * math.cos(math.pi / 2 + 2 * math.pi * k / n),
             radii[k % len(radii)] * math.sin(math.pi / 2 + 2 * math.pi * k / n)] for k in range(n)]


# name -> (vertices, closed)
NAMED = {
    "square": ([[-1, -1], [1, -1], [1, 1], [-1, 1]], True),
    "line": ([[-1, 0], [1, 0]], False),
    "triangle": (_regular_polygon(3), True),
    "pentagon": (_regular_polygon(5), True),
    "star": (_regular_polygon(10, radii=(1.0, 0.45)), True),
    "Z": ([[-1, 1], [1, 1], [-1, -1], [1, -1]], False),
    "M": ([[-1, -1], [-1, 1], [0, 0], [1, 1], [1, -1]], False),
}

# Phase 2 trains on these (randomly rotated, stretched, scaled, shifted) ...
TRAIN_SHAPES = ["circle", "square", "line", "triangle", "pentagon", "random_polygon", "random_polyline"]
# ... and never sees these, so they can test generalization in Phase 4.
HELDOUT_SHAPES = ["star", "Z", "M"]


def sample_shape(name: str, m: int, dim: int = 2, vertices=None, closed: bool = False) -> torch.Tensor:
    """Return [m, dim] target points for a named shape.

    name: "circle", any key of NAMED ("square", "line", "triangle", "star",
    "Z", ...), or "polyline" (pass vertices, and closed=True for a closed
    outline). For dim > 2 the shape lies flat in the z = 0 plane, so 3D runs
    work without new shapes.
    """
    if name == "circle":
        theta = 2 * math.pi * torch.arange(m) / m
        pts = (EXTENT / 2) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    elif name in NAMED:
        verts, is_closed = NAMED[name]
        pts = normalize(sample_polyline(verts, m, closed=is_closed))
    elif name == "polyline":
        if vertices is None:
            raise ValueError("polyline needs vertices")
        pts = normalize(sample_polyline(vertices, m, closed=closed))
    else:
        raise ValueError(f"unknown shape: {name!r}")

    if dim > 2:
        pts = torch.cat([pts, torch.zeros(m, dim - 2)], dim=1)
    return pts


def random_shape(m: int, dim: int = 2, names=TRAIN_SHAPES, gen: torch.Generator | None = None) -> torch.Tensor:
    """One random target shape, [m, dim].

    Picks a shape from `names`, then randomly rotates and stretches it,
    rescales it to fit (0.6 to 1.0 of EXTENT) and shifts it inside the box, so
    the policy never sees the same target twice.
    """
    def rand(*size):
        return torch.rand(*size, generator=gen)

    name = names[int(torch.randint(len(names), (1,), generator=gen))]
    if name == "circle":
        theta = 2 * math.pi * torch.arange(m) / m
        base = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    elif name == "random_polygon":            # star-convex polygon, 3 to 7 corners
        n = int(torch.randint(3, 8, (1,), generator=gen))
        ang = torch.sort(rand(n)).values * 2 * math.pi
        rad = 0.5 + 0.5 * rand(n)
        base = sample_polyline(torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], 1), m, closed=True)
    elif name == "random_polyline":           # open path through 3 to 5 random points
        n = int(torch.randint(3, 6, (1,), generator=gen))
        base = sample_polyline(rand(n, 2) * 2 - 1, m)
    else:
        verts, is_closed = NAMED[name]
        base = sample_polyline(verts, m, closed=is_closed)

    a = float(rand(1)) * 2 * math.pi
    rot = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = normalize((base @ rot.T) * (0.5 + 0.5 * rand(2)))      # rotate, stretch, fit to EXTENT
    pts = pts * (0.6 + 0.4 * rand(1))                            # random size
    half = (pts.max(0).values - pts.min(0).values) / 2
    pts = pts + (rand(2) * 2 - 1) * (EXTENT / 2 - half)          # random shift, stays inside
    if dim > 2:
        pts = torch.cat([pts, torch.zeros(m, dim - 2)], dim=1)
    return pts


def random_shapes(batch: int, m: int, dim: int = 2, names=TRAIN_SHAPES,
                  gen: torch.Generator | None = None) -> torch.Tensor:
    """A batch of independent random shapes, [batch, m, dim]."""
    return torch.stack([random_shape(m, dim, names, gen) for _ in range(batch)])
