from types import SimpleNamespace

import torch

from swarm.sim import rollout, step


def test_rollout_shape():
    cfg = SimpleNamespace(dt=0.05, steps=7, damping=0.5)
    traj = rollout(torch.zeros(2, 5, 2), torch.zeros(2, 5, 2), torch.zeros(2, 7, 5, 2), cfg)
    assert traj.shape == (2, 8, 5, 2)


def test_constant_velocity_without_force_or_damping():
    cfg = SimpleNamespace(dt=0.1, steps=20, damping=0.0)
    p0, v0 = torch.rand(1, 4, 2), torch.rand(1, 4, 2)
    traj = rollout(p0, v0, torch.zeros(1, 20, 4, 2), cfg)
    t = torch.arange(21, dtype=torch.float32)[None, :, None, None]
    assert torch.allclose(traj, p0[:, None] + t * cfg.dt * v0[:, None], atol=1e-5)


def test_damping_decays_velocity_geometrically():
    dt, c = 0.05, 0.5
    lam = 1 - c * dt
    v = torch.ones(1, 1, 2)
    p = torch.zeros(1, 1, 2)
    for _ in range(10):
        p, v = step(p, v, torch.zeros_like(v), dt, c)
    assert torch.allclose(v, torch.full_like(v, lam**10))


def test_final_position_matches_closed_form():
    # Math page section 4: p_T = p_0 + sum_s w_s a_s with w_s = (dt/c)(1 - lam^(T-s)), from rest.
    cfg = SimpleNamespace(dt=0.05, steps=100, damping=0.5)
    lam = 1 - cfg.damping * cfg.dt
    p0 = torch.rand(1, 3, 2, dtype=torch.float64)
    a = torch.randn(1, cfg.steps, 3, 2, dtype=torch.float64)
    traj = rollout(p0, torch.zeros_like(p0), a, cfg)
    s = torch.arange(cfg.steps, dtype=torch.float64)
    w = (cfg.dt / cfg.damping) * (1 - lam ** (cfg.steps - s))
    expected = p0 + (w[None, :, None, None] * a).sum(dim=1)
    assert torch.allclose(traj[:, -1], expected)


def test_step_gradcheck():
    args = [torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    assert torch.autograd.gradcheck(lambda p, v, a: step(p, v, a, 0.05, 0.5), args)
