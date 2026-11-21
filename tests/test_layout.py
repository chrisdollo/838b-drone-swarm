import numpy as np

from swarm.layout import find_anchors, fit_to_box, layout, text_goals, text_strokes


def test_exact_count_and_spacing():
    for n in (40, 64, 96, 128):
        g, info = text_goals("MARRY\nME", n=n)
        assert g.shape == (n, 2) and info["n"] == n
        assert info["min_gap"] >= 0.6                      # no goals piled together
        assert np.abs(g).max() <= 0.4 + 1e-6


def test_pitch_mode_sets_the_drone_count():
    g1, i1 = text_goals("HELLO", pitch=0.04)
    g2, i2 = text_goals("HELLO", pitch=0.02)
    assert 1.6 < i2["n"] / i1["n"] < 2.4                     # half the spacing, about twice the drones


def test_every_corner_and_end_gets_a_goal():
    strokes = fit_to_box(text_strokes("A"))
    g, info = layout(strokes, n=30)
    apex = max((p for s, _ in strokes for p in s), key=lambda p: p[1])
    assert np.linalg.norm(g - apex, axis=1).min() < 1e-6     # the tip of the A
    for p, closed in strokes:                                # every stroke end
        if not closed:
            for end in (p[0], p[-1]):
                assert np.linalg.norm(g - end, axis=1).min() < 0.5 * info["pitch"]


def test_too_few_drones_is_flagged_unreadable():
    assert not text_goals("MARRY\nME", n=10)[1]["readable"]
    assert text_goals("MARRY\nME", n=128)[1]["readable"]


def test_closed_curve_without_corners():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = [(np.stack([np.cos(t), np.sin(t)], 1) * 0.4, True)]
    g, info = layout(circle, n=16)
    gaps = np.linalg.norm(g[:, None] - g[None], axis=-1) + np.eye(16) * 9
    assert g.shape == (16, 2) and np.ptp(gaps.min(1)) < 0.01  # evenly spaced around


def test_junction_gets_one_goal():
    strokes = fit_to_box(text_strokes("T"))                  # bar and stem meet at the top
    anchors = find_anchors(strokes, merge=0.05)
    d = np.linalg.norm(anchors[:, None] - anchors[None], axis=-1) + np.eye(len(anchors)) * 9
    assert d.min() >= 0.05
