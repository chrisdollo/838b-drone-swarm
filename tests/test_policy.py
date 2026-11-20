from types import SimpleNamespace

import torch

from swarm.policy import SharedMLP
from swarm.sim import rollout_closed_loop

CFG = SimpleNamespace(dt=0.05, steps=15, damping=0.5)


def test_actions_bounded():
    policy = SharedMLP(a_max=2.0)
    with torch.no_grad():
        for p in policy.parameters():
            p.normal_(0, 5)                                   # huge weights
    a = policy(torch.randn(3, 10, 30) * 100)
    assert a.shape == (3, 10, 2)
    assert a.abs().max() <= 2.0


def test_shared_weights_same_obs_same_action():
    policy = SharedMLP()
    obs = torch.randn(1, 1, 30).expand(2, 5, 30)             # every agent sees the same thing
    a = policy(obs)
    assert torch.allclose(a, a[:1, :1].expand_as(a))


def test_starts_near_zero_action():
    a = SharedMLP().act(torch.rand(2, 16, 2) - 0.5, torch.zeros(2, 16, 2), torch.rand(2, 32, 2) - 0.5)
    assert a.abs().max() < 0.1


def test_gradients_reach_every_weight_through_rollout():
    torch.manual_seed(0)
    policy = SharedMLP(k=3, m=4)
    pos0, Q = torch.rand(2, 8, 2) - 0.5, torch.rand(2, 16, 2) - 0.5
    traj, _ = rollout_closed_loop(pos0, torch.zeros_like(pos0), Q, policy.act, CFG)
    traj[:, -1].pow(2).sum().backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_save_load_roundtrip(tmp_path):
    policy = SharedMLP(k=4, m=5, hidden=32)
    policy.save(tmp_path / "p.pt")
    loaded = SharedMLP.load(tmp_path / "p.pt")
    obs = torch.randn(2, 3, 20)
    assert torch.allclose(policy(obs), loaded(obs))
