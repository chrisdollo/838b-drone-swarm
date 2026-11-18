import torch

from swarm.losses import chamfer


def test_zero_on_identical_sets():
    P = torch.rand(3, 10, 2)
    assert chamfer(P, P).item() == 0.0


def test_symmetric():
    P, Q = torch.rand(2, 10, 2), torch.rand(2, 25, 2)
    assert torch.allclose(chamfer(P, Q), chamfer(Q, P))


def test_worked_example():
    # Math page section 5: two agents near one point of a 4-point circle -> L = 0.01 + 0.25
    Q = torch.tensor([[0.4, 0.0], [0.0, 0.4], [-0.4, 0.0], [0.0, -0.4]])
    P = torch.tensor([[[0.4, 0.1], [0.3, 0.0]]])
    assert torch.isclose(chamfer(P, Q), torch.tensor(0.26), atol=1e-6)


def test_collapse_is_penalized():
    # All agents on one target point: term A is 0, but term B (coverage) is not.
    Q = torch.tensor([[0.4, 0.0], [0.0, 0.4], [-0.4, 0.0], [0.0, -0.4]])
    P = Q[:1].repeat(5, 1)[None]
    assert chamfer(P, Q).item() > 0.1


def test_gradcheck():
    P = torch.rand(1, 6, 2, dtype=torch.float64, requires_grad=True)
    Q = torch.rand(1, 9, 2, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda p: chamfer(p, Q), (P,))
