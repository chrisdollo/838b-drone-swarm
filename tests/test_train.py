import math
from types import SimpleNamespace

import torch

from swarm.config import load_config
from swarm.losses import collision_loss, effort_loss, shape_loss, total_loss
from swarm.plan import optimize_plan
from swarm.shapes import sample_shape
from swarm.train import rollout_length


def test_curriculum():
    sched = [[0, 40], [500, 70], [1000, 100]]
    assert [rollout_length(i, sched, 100) for i in (0, 499, 500, 1500)] == [40, 40, 70, 100]


def test_collision_penalty_per_scene():
    far = torch.tensor([[[0.0, 0.0], [0.3, 0.0]]]).expand(1, 3, 2, 2)    # [B, T+1, N, dim]
    near = torch.tensor([[[0.0, 0.0], [0.01, 0.0]]]).expand(1, 3, 2, 2)
    assert collision_loss(far, 0.02).item() == 0
    assert math.isclose(collision_loss(near, 0.02).item(), 0.01**2, rel_tol=1e-4)
    both = torch.cat([far, near])
    assert torch.allclose(collision_loss(both, torch.tensor([0.02, 0.02]), per_scene=True),
                          torch.tensor([0.0, 0.01**2]), rtol=1e-4)


def test_shape_loss_hold_and_effort():
    Q = torch.rand(2, 6, 2)
    traj = Q[:, None, :3].expand(2, 11, 3, 2).clone()
    traj[:, :-2] += 1.0                                       # far away except the last 2 frames
    assert shape_loss(traj, Q, hold_frac=0.0) < shape_loss(traj, Q, hold_frac=0.5)
    assert effort_loss(torch.full((1, 4, 3, 2), 2.0)).item() == 8.0


def test_total_loss_normalized_by_spacing():
    # The same scene scaled by 2 (spacing and d_min too) gives the same normalized loss.
    cfg = load_config()
    g = torch.Generator().manual_seed(0)
    traj, Q = torch.rand(1, 5, 4, 2, generator=g) * 0.2, torch.rand(1, 8, 2, generator=g) * 0.2
    accs = torch.zeros(1, 4, 4, 2)
    a, _ = total_loss(traj, accs, Q, torch.tensor([0.02]), cfg.loss)
    b, _ = total_loss(2 * traj, accs, 2 * Q, torch.tensor([0.04]), cfg.loss)
    assert torch.isclose(a, b, rtol=1e-4)


def test_plan_reaches_circle():
    cfg = SimpleNamespace(dt=0.05, steps=60, damping=0.5, a_max=2.0)
    torch.manual_seed(0)
    out = optimize_plan(torch.rand(1, 8, 2) - 0.5, sample_shape("circle", 16)[None], cfg, iters=150)
    assert out["history"][-1] < 0.2 * out["history"][0]
