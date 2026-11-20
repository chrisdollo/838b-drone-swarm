"""Local observations: what each agent can see.

This file is what makes the policy decentralized. Agent i never sees the whole
swarm or the whole shape, only:
  - its k nearest other agents, as offsets p_j - p_i
  - its m nearest target points, as offsets q_l - p_i
  - its own velocity v_i
Everything is relative to the agent itself, so the observation does not change
if the whole scene is shifted (translation invariance), and its size does not
depend on N, so one policy works for any swarm size.
"""

import torch

from swarm.losses import pairwise_sq_dist


def knn_offsets(src: torch.Tensor, dst: torch.Tensor, k: int, exclude_self: bool = False) -> torch.Tensor:
    """Offsets from each src point to its k nearest dst points, nearest first.

    src: [B, N, dim], dst: [B, M, dim] -> [B, N, k, dim] holding dst_j - src_i.
    exclude_self: src and dst are the same set, so skip each point itself.
    If fewer than k points exist, the missing slots are zero.
    """
    B, N, dim = src.shape
    M = dst.shape[1]
    k_avail = min(k, M - 1 if exclude_self else M)

    # Choosing *which* points are nearest is a discrete decision with no gradient,
    # so it is done without autograd. The offsets themselves stay differentiable.
    with torch.no_grad():
        D = pairwise_sq_dist(src, dst)
        if exclude_self:
            D.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        idx = D.topk(k_avail, dim=2, largest=False).indices          # [B, N, k_avail], sorted

    batch = torch.arange(B, device=src.device)[:, None, None]
    off = dst[batch, idx] - src[:, :, None, :]                        # [B, N, k_avail, dim]
    if k_avail < k:
        off = torch.cat([off, off.new_zeros(B, N, k - k_avail, dim)], dim=2)
    return off


def observe(pos: torch.Tensor, vel: torch.Tensor, Q: torch.Tensor, k: int, m: int) -> torch.Tensor:
    """Observation of every agent, [B, N, dim * (k + m + 1)].

    pos, vel: [B, N, dim]. Q: target points [B, M, dim].
    Layout: [k neighbor offsets | m target offsets | own velocity], flattened.
    """
    neighbors = knn_offsets(pos, pos, k, exclude_self=True)          # [B, N, k, dim]
    targets = knn_offsets(pos, Q, m)                                 # [B, N, m, dim]
    return torch.cat([neighbors.flatten(2), targets.flatten(2), vel], dim=-1)


def obs_dim(k: int, m: int, dim: int = 2) -> int:
    return dim * (k + m + 1)
