import math

import torch

from swarm.config import load_config
from swarm.losses import collision_loss, effort_loss, shape_loss
from swarm.train import make_batch, rollout_length, train


def tiny_config():
    cfg = load_config()
    cfg.n_agents, cfg.steps = 8, 12
    cfg.train.iters, cfg.train.batch_size = 6, 4
    cfg.train.val_every, cfg.train.val_scenes = 3, 4
    cfg.train.curriculum = [[0, 6], [3, 12]]
    return cfg


def test_training_smoke(tmp_path):
    policy, history = train(tiny_config(), run_dir=tmp_path)
    assert len(history["train"]) == 6
    assert all(math.isfinite(h["loss"]) and math.isfinite(h["grad_norm"]) for h in history["train"])
    assert [h["T"] for h in history["train"]] == [6, 6, 6, 12, 12, 12]
    assert history["val"][0]["it"] == 0 and history["val"][-1]["it"] == 6
    assert (tmp_path / "best.pt").exists() and (tmp_path / "history.json").exists()


def test_curriculum():
    sched = [[0, 40], [500, 70], [1000, 100]]
    assert [rollout_length(i, sched, 100) for i in (0, 499, 500, 1500)] == [40, 40, 70, 100]


def test_make_batch_reproducible():
    a = make_batch(3, 5, 2, torch.Generator().manual_seed(7))
    b = make_batch(3, 5, 2, torch.Generator().manual_seed(7))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_collision_penalty():
    far = torch.tensor([[[0.0, 0.0], [0.3, 0.0]]]).expand(1, 3, 2, 2)    # [B, T+1, N, dim]
    near = torch.tensor([[[0.0, 0.0], [0.01, 0.0]]]).expand(1, 3, 2, 2)
    assert collision_loss(far, 0.02).item() == 0
    assert math.isclose(collision_loss(near, 0.02).item(), 0.01**2, rel_tol=1e-4)   # each agent: (0.02-0.01)^2


def test_shape_loss_hold_and_effort():
    Q = torch.rand(2, 6, 2)
    traj = Q[:, None, :3].expand(2, 11, 3, 2).clone()
    traj[:, :-2] += 1.0                                       # far away except the last 2 frames
    assert shape_loss(traj, Q, hold_frac=0.0) < shape_loss(traj, Q, hold_frac=0.5)
    assert effort_loss(torch.full((1, 4, 3, 2), 2.0)).item() == 8.0
