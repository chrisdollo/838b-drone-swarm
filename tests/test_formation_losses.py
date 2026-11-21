import math

import torch

from swarm.formation_losses import KINDS, formation_loss, laplacian_error

Q = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
S = torch.tensor([1.0])


def as_traj(P):
    return P[:, None].expand(1, 3, -1, -1).clone()


def test_perfect_formation_is_best_for_every_loss():
    doubled = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.05, 0.0]]])   # 2 drones on slot 2, slot 3 empty
    for k in KINDS:
        good = formation_loss(k, as_traj(Q.clone()), Q, S, 0.5).item()
        bad = formation_loss(k, as_traj(doubled), Q, S, 0.5).item()
        assert good < bad, k
        assert good < 1e-3, k


def test_sinkhorn_close_to_matching_for_small_blur():
    P = Q + 0.1 * torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(0))
    m = formation_loss("matching", as_traj(P), Q, S, 0.5).item()
    from swarm.formation_losses import sinkhorn_loss
    sk = sinkhorn_loss(as_traj(P), Q, S, 0.5, blur=0.05).item()
    assert math.isclose(sk, m, rel_tol=0.05)


def test_laplacian_ignores_translation_rotation_scale():
    P = torch.tensor([[[0.0, 0.0], [1.0, 0.2], [2.0, -0.1], [3.0, 0.4]]])
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    moved = 2.5 * P @ rot.T + torch.tensor([5.0, -3.0])
    assert laplacian_error(P, moved).item() < 1e-10
    assert laplacian_error(P, Q).item() > 1e-4


def test_losses_have_gradients():
    P = (Q + 0.3).requires_grad_(True)
    for k in KINDS:
        formation_loss(k, as_traj(P), Q, S, 0.5).sum().backward()
        assert P.grad is not None and P.grad.abs().sum() > 0, k
        P.grad = None


def test_assigned_uses_the_given_order():
    from swarm.formation_losses import assigned_loss
    swapped = Q[:, [1, 0, 2, 3]]
    assert assigned_loss(as_traj(Q.clone()), Q, S, 0.5).item() == 0
    assert assigned_loss(as_traj(Q.clone()), swapped, S, 0.5).item() > 0
