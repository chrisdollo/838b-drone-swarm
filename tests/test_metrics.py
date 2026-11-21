import math

import torch

from swarm.metrics import slot_error_curve, slot_metrics, steps_to_form

Q = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
D_MIN = torch.tensor([0.4])           # spacing s = 1


def test_perfect_formation():
    m = slot_metrics(Q.clone(), Q, D_MIN)
    assert m["slot_error"].item() == 0 and m["filled"].item() == 1 and m["shared_goals"].item() == 0


def test_two_drones_on_one_goal_leave_one_empty():
    P = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.1, 0.0]]])   # goal 3 empty, goal 2 doubled
    m = slot_metrics(P, Q, D_MIN)
    assert m["filled"].item() == 0.75                       # Chamfer would only see a small error
    assert math.isclose(m["slot_error"].item(), 0.9 / 4, rel_tol=1e-5)
    assert m["shared_goals"].item() == 0.25


def test_anchor_split():
    P = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]]])
    m = slot_metrics(P, Q, D_MIN, anchor_mask=torch.tensor([[True, False, False, True]]))
    assert m["filled_anchor"].item() == 0.5 and m["filled_fill"].item() == 1.0


def test_steps_to_form_must_stay_formed():
    curve = torch.tensor([[2.0, 0.4, 0.8, 0.3, 0.2], [2.0, 1.0, 0.9, 0.8, 0.7]])
    s = steps_to_form(curve, 0.5)
    assert s[0].item() == 3 and math.isnan(s[1].item())


def test_curve_matches_final_metric():
    traj = torch.stack([Q + 1.0, Q], dim=1)                 # [1, 2, 4, 2]
    c = slot_error_curve(traj, Q, D_MIN)
    assert c.shape == (1, 2) and c[0, 1].item() == 0 and c[0, 0].item() > 0
