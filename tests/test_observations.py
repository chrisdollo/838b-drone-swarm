import torch

from swarm.observations import knn_offsets, obs_dim, observe

K, M_T = 6, 8


def scene(B=2, N=20, M=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, 2, generator=g) - 0.5, torch.randn(B, N, 2, generator=g),
            torch.rand(B, M, 2, generator=g) - 0.5)


def test_shape_and_size_independent_of_n():
    for N in (10, 32, 70):
        pos, vel, Q = scene(N=N, M=2 * N)
        assert observe(pos, vel, Q, K, M_T).shape == (2, N, obs_dim(K, M_T))


def test_translation_invariance():
    pos, vel, Q = scene()
    shift = torch.tensor([0.3, -0.2])
    assert torch.allclose(observe(pos, vel, Q, K, M_T), observe(pos + shift, vel, Q + shift, K, M_T), atol=1e-6)


def test_permutation_equivariance():
    # Reordering the agents only reorders the observations: no agent has a fixed identity.
    pos, vel, Q = scene()
    perm = torch.randperm(pos.shape[1])
    assert torch.allclose(observe(pos, vel, Q, K, M_T)[:, perm], observe(pos[:, perm], vel[:, perm], Q, K, M_T))


def test_neighbors_exclude_self_and_are_sorted():
    pos, _, _ = scene()
    off = knn_offsets(pos, pos, K, exclude_self=True)
    dist = off.norm(dim=-1)
    assert (dist > 0).all()                                  # never itself
    assert (dist[..., 1:] >= dist[..., :-1] - 1e-7).all()    # nearest first


def test_matches_brute_force():
    pos, _, Q = scene(B=1, N=5, M=9)
    off = knn_offsets(pos, Q, 3)[0, 2]                       # agent 2's 3 nearest targets
    d = (Q[0] - pos[0, 2]).norm(dim=1)
    expected = Q[0][d.argsort()[:3]] - pos[0, 2]
    assert torch.allclose(off, expected)


def test_pads_when_fewer_than_k_agents():
    pos, vel, Q = scene(N=4, M=8)
    off = knn_offsets(pos, pos, K, exclude_self=True)        # only 3 other agents exist
    assert off.shape == (2, 4, K, 2)
    assert (off[:, :, 3:] == 0).all()


def test_gradient_flows_through_offsets():
    pos, vel, Q = scene()
    pos.requires_grad_(True)
    observe(pos, vel, Q, K, M_T).sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()
