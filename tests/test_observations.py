import torch

from swarm.observations import goal_occupancy, knn_offsets, obs_dim, observe, _knn_idx

K, M_T = 6, 8


def scene(B=2, N=20, M=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, 2, generator=g) - 0.5, torch.randn(B, N, 2, generator=g),
            torch.rand(B, M, 2, generator=g) - 0.5, torch.full((B,), 0.03))


def test_shape_and_size_independent_of_n():
    for N in (8, 32, 70):
        pos, vel, Q, d = scene(N=N, M=2 * N)
        assert observe(pos, vel, Q, d, K, M_T).shape == (2, N, obs_dim(K, M_T))


def test_translation_invariance():
    pos, vel, Q, d = scene()
    shift = torch.tensor([0.3, -0.2])
    assert torch.allclose(observe(pos, vel, Q, d, K, M_T), observe(pos + shift, vel, Q + shift, d, K, M_T), atol=1e-5)


def test_permutation_equivariance():
    # Reordering the agents only reorders the observations: no agent has a fixed identity.
    pos, vel, Q, d = scene()
    perm = torch.randperm(pos.shape[1])
    assert torch.allclose(observe(pos, vel, Q, d, K, M_T)[:, perm], observe(pos[:, perm], vel[:, perm], Q, d, K, M_T))


def test_neighbors_exclude_self_and_are_sorted():
    pos, _, _, _ = scene()
    dist = knn_offsets(pos, pos, K, exclude_self=True).norm(dim=-1)
    assert (dist > 0).all()
    assert (dist[..., 1:] >= dist[..., :-1] - 1e-7).all()


def test_matches_brute_force():
    pos, _, Q, _ = scene(B=1, N=5, M=9)
    off = knn_offsets(pos, Q, 3)[0, 2]
    d = (Q[0] - pos[0, 2]).norm(dim=1)
    assert torch.allclose(off, Q[0][d.argsort()[:3]] - pos[0, 2])


def test_occupancy_skips_myself():
    # Agent 0 sits on goal 0; agent 1 is 0.3 away. Agent 0 must see goal 0 as "0.3 from someone else",
    # agent 1 must see it as taken (0.0 away, by agent 0).
    pos = torch.tensor([[[0.0, 0.0], [0.3, 0.0]]])
    Q = torch.tensor([[[0.0, 0.0], [0.3, 0.0]]])
    occ = goal_occupancy(pos, Q, _knn_idx(pos, Q, 1, False))
    assert torch.allclose(occ[0, :, 0], torch.tensor([0.3, 0.3]), atol=1e-5)   # each sits on its own goal
    occ_other = goal_occupancy(pos, Q, torch.tensor([[[1], [0]]]))            # look at the other one's goal
    assert torch.allclose(occ_other[0, :, 0], torch.tensor([0.0, 0.0]), atol=1e-5)


def test_pads_when_too_few():
    pos, vel, Q, d = scene(N=4, M=5)
    o = observe(pos, vel, Q, d, K, M_T)
    assert o.shape == (2, 4, obs_dim(K, M_T)) and torch.isfinite(o).all()


def test_gradient_flows():
    pos, vel, Q, d = scene()
    pos.requires_grad_(True)
    observe(pos, vel, Q, d, K, M_T).sum().backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
