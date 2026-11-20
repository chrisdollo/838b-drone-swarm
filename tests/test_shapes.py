import pytest
import torch

from swarm.shapes import sample_shape

SHAPES = ["circle", "square", "line"]


@pytest.mark.parametrize("name", SHAPES)
@pytest.mark.parametrize("m", [16, 64, 101])
def test_count_and_inside_box(name, m):
    q = sample_shape(name, m)
    assert q.shape == (m, 2)
    assert q.abs().max() <= 0.5


def test_circle_radius():
    q = sample_shape("circle", 64)
    assert torch.allclose(q.norm(dim=1), torch.full((64,), 0.4))


@pytest.mark.parametrize("name", ["circle", "square"])
def test_closed_shapes_evenly_spaced(name):
    q = sample_shape(name, 64)
    gaps = (q.roll(-1, dims=0) - q).norm(dim=1)   # includes the wrap-around gap
    assert torch.allclose(gaps, gaps.mean().expand_as(gaps), atol=1e-5)


def test_polyline_is_normalized():
    q = sample_shape("polyline", 50, vertices=[[0, 0], [10, 0], [10, 5]])
    assert q.shape == (50, 2)
    span = q.max(dim=0).values - q.min(dim=0).values
    assert torch.isclose(span.max(), torch.tensor(0.8))


def test_3d_lies_flat():
    q = sample_shape("circle", 32, dim=3)
    assert q.shape == (32, 3)
    assert torch.all(q[:, 2] == 0)


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        sample_shape("hexagon", 10)


def test_named_and_heldout_shapes():
    from swarm.shapes import HELDOUT_SHAPES, NAMED
    for name in list(NAMED) + HELDOUT_SHAPES:
        q = sample_shape(name, 40)
        assert q.shape == (40, 2) and q.abs().max() <= 0.5


def test_random_shapes_inside_box_and_reproducible():
    from swarm.shapes import random_shapes
    a = random_shapes(50, 64, gen=torch.Generator().manual_seed(3))
    b = random_shapes(50, 64, gen=torch.Generator().manual_seed(3))
    assert a.shape == (50, 64, 2) and torch.equal(a, b)
    assert a.abs().max() <= 0.4 + 1e-5
