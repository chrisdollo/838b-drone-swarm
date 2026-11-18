"""Losses on agent positions. Phase 1: Chamfer distance. Phase 2 adds collision and effort."""

import torch


def pairwise_sq_dist(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """D[b, i, j] = ||P[b, i] - Q[b, j]||^2, shape [B, N, M].

    Computed from the coordinate differences directly instead of
    torch.cdist(P, Q) ** 2: that would take a square root and then square it,
    and the square root has an infinite slope at distance 0 (an agent sitting
    exactly on a target point). This form is exact and smooth everywhere.
    """
    return ((P[:, :, None, :] - Q[:, None, :, :]) ** 2).sum(dim=-1)


def chamfer(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between agents P [B, N, dim] and targets Q [B, M, dim] or [M, dim].

    L = mean_i min_j D_ij  +  mean_j min_i D_ij
        (A: every agent is on the shape)  (B: every part of the shape is covered)

    No agent is assigned to a target: each min picks whichever is nearest
    right now. Returns a scalar averaged over the batch.
    """
    if Q.dim() == 2:
        Q = Q.expand(P.shape[0], -1, -1)
    D = pairwise_sq_dist(P, Q)
    term_a = D.min(dim=2).values.mean(dim=1)   # [B] each agent -> nearest target
    term_b = D.min(dim=1).values.mean(dim=1)   # [B] each target -> nearest agent
    return (term_a + term_b).mean()
