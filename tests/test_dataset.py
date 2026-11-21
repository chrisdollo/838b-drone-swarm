import numpy as np

from swarm.dataset import (BENCH_N, D_MIN_FRAC, EXCLUDED, HELDOUT_FAMILIES, HELDOUT_STRINGS, N_RANGE,
                           TRAIN_FAMILIES, WORDS, draw_heldout_strings, feasible, feature_ratio, make_batch,
                           make_scene, random_text, total_length)
from swarm.layout import fit_to_box, text_strokes


def test_sizes_match_plan():
    assert N_RANGE == (8, 96) and BENCH_N == [8, 16, 24, 32, 48, 64, 96]


def test_one_goal_per_drone_and_spacing_rule():
    for fam in ("circle", "random_polyline", "random_string"):
        sc = make_scene(fam, 24, np.random.default_rng(0))
        assert sc["goals"].shape == (24, 2) and sc["start"].shape == (24, 2)
        assert np.isclose(sc["d_min"], D_MIN_FRAC * sc["length"] / 24)
        assert np.abs(sc["goals"]).max() <= 0.4 + 1e-5
        assert sc["anchor"].sum() >= 1


def test_feature_rule():
    assert not feasible("spiral", 8)              # turns closer than 8 drones can draw
    assert feasible("spiral", 32)
    assert feasible("circle", 8)


def test_marry_me_needs_about_92_drones():
    strokes = fit_to_box(text_strokes("MARRY\nME"))       # the rule works at box scale
    assert feature_ratio(strokes, 80) < 1.0 <= feature_ratio(strokes, 96)


def test_digits_in_font():
    for ch in "0123456789":
        assert total_length(text_strokes(ch)) > 0, ch


def test_heldout_strings_fixed_and_never_trained_on():
    assert draw_heldout_strings() == HELDOUT_STRINGS          # same seed, same list
    rng = np.random.default_rng(0)
    drawn = {random_text(rng) for _ in range(20000)}
    assert not drawn & EXCLUDED
    assert {"HI", "UMD", "LOVE", "FIRE", "MARRY", "ME"} <= EXCLUDED and set(HELDOUT_STRINGS) <= EXCLUDED
    assert all(w.replace("\n", "") not in HELDOUT_STRINGS for w in WORDS)


def test_make_batch_shapes():
    pos0, vel0, Q, d_min = make_batch(4, 16, np.random.default_rng(1), TRAIN_FAMILIES)
    assert pos0.shape == (4, 16, 2) and Q.shape == (4, 16, 2) and d_min.shape == (4,)
    assert (d_min > 0).all() and (vel0 == 0).all()


def test_families_disjoint():
    assert not set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES)
