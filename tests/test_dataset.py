import numpy as np
import torch

from swarm.dataset import (D_MIN_FRAC, FONT, GRID, HELDOUT_FAMILIES, TRAIN_FAMILIES, feasible,
                           feature_ratio, make_batch, make_scene, sample_strokes, text_strokes, total_length)


def test_grid_matches_plan():
    assert len(GRID) == 11 and all(m >= n for n, m in GRID)


def test_scene_fields_and_spacing_rule():
    sc = make_scene("circle", 16, 32, np.random.default_rng(0))
    assert sc["goals"].shape == (32, 2) and sc["start"].shape == (16, 2)
    assert np.isclose(sc["d_min"], D_MIN_FRAC * sc["length"] / 16)
    assert np.abs(sc["goals"]).max() <= 0.4 + 1e-5


def test_feature_rule():
    assert not feasible("spiral", 8)              # turns closer than 8 drones can draw
    assert feasible("spiral", 32)
    assert feasible("circle", 8)


def test_marry_me_needs_about_52_drones():
    strokes = text_strokes("MARRY\nME")
    assert feature_ratio(strokes, 48) < 1.0 <= feature_ratio(strokes, 56)


def test_every_letter_drawable():
    for ch in FONT:
        assert feature_ratio(text_strokes(ch), 40) >= 1.0, ch


def test_sample_strokes_count_and_spread():
    strokes = text_strokes("HI")
    pts = sample_strokes(strokes, 50)
    assert pts.shape == (50, 2)
    assert np.isclose(total_length(strokes), sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p, _ in strokes))


def test_make_batch_shapes():
    pos0, vel0, Q, d_min = make_batch(4, 8, 16, np.random.default_rng(1), TRAIN_FAMILIES)
    assert pos0.shape == (4, 8, 2) and Q.shape == (4, 16, 2) and d_min.shape == (4,)
    assert (d_min > 0).all() and (vel0 == 0).all()


def test_families_disjoint():
    assert not set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES)
