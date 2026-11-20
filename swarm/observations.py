"""Local observations: what each agent can see.

This file is what makes the policy decentralized. Agent i never sees the whole
swarm or the whole shape, only:
  - its k nearest other agents, as offsets p_j - p_i
  - its m nearest goal points, as offsets q_l - p_i
  - for each of those goals, how close the nearest OTHER agent is to it
    ("is this goal already taken?": small = taken, large = free)
  - the scene's collision distance d_min (a mission parameter every drone knows)
  - its own velocity v_i
Everything is relative to the agent itself, so the observation does not change
if the whole scene is shifted, and its size does not depend on N, so one policy
works for any swarm size.
"""

import torch

from swarm.losses import pairwise_sq_dist


def _knn_idx(src: torch.Tensor, dst: torch.Tensor, k: int, exclude_self: bool) -> torch.Tensor:
    """Indices of the k nearest dst points for every src point, nearest first, [B, N, k].

    Choosing *which* points are nearest is a discrete decision with no
    gradient, so it is done without autograd; offsets built from these
    indices stay differentiable.
    """
    with torch.no_grad():
        D = pairwise_sq_dist(src, dst)
        if exclude_self:
            D.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        return D.topk(k, dim=2, largest=False).indices


def knn_offsets(src: torch.Tensor, dst: torch.Tensor, k: int, exclude_self: bool = False) -> torch.Tensor:
    """Offsets from each src point to its k nearest dst points, nearest first, [B, N, k, dim].

    If fewer than k points exist, the missing slots are zero.
    """
    B, N, dim = src.shape
    k_avail = min(k, dst.shape[1] - (1 if exclude_self else 0))
    idx = _knn_idx(src, dst, k_avail, exclude_self)
    batch = torch.arange(B, device=src.device)[:, None, None]
    off = dst[batch, idx] - src[:, :, None, :]
    if k_avail < k:
        off = torch.cat([off, off.new_zeros(B, N, k - k_avail, dim)], dim=2)
    return off


def goal_occupancy(pos: torch.Tensor, Q: torch.Tensor, goal_idx: torch.Tensor) -> torch.Tensor:
    """For each agent i and each goal it observes: distance from that goal to the nearest agent other than i.

    pos [B, N, dim], Q [B, M, dim], goal_idx [B, N, m] -> [B, N, m].
    """
    B, N, _ = pos.shape
    if N < 2:
        return pos.new_ones(B, N, goal_idx.shape[2])
    D = pairwise_sq_dist(Q, pos)                                     # [B, M, N] goal -> agent
    vals, who = D.topk(2, dim=2, largest=False)                      # two nearest agents per goal
    batch = torch.arange(B, device=pos.device)[:, None, None]
    v, w = vals[batch, goal_idx], who[batch, goal_idx]               # [B, N, m, 2]
    me = torch.arange(N, device=pos.device)[None, :, None]
    other = torch.where(w[..., 0] == me, v[..., 1], v[..., 0])       # skip myself
    return torch.sqrt(other + 1e-12)


def observe(pos: torch.Tensor, vel: torch.Tensor, Q: torch.Tensor, d_min: torch.Tensor,
            k: int, m: int) -> torch.Tensor:
    """Observation of every agent, [B, N, obs_dim(k, m, dim)].

    pos, vel [B, N, dim], Q [B, M, dim], d_min [B].
    Layout: [k neighbor offsets | m goal offsets | m goal occupancies | d_min | own velocity].
    """
    B, N, dim = pos.shape
    m_avail = min(m, Q.shape[1])
    neighbors = knn_offsets(pos, pos, k, exclude_self=True)                   # [B, N, k, dim]
    gidx = _knn_idx(pos, Q, m_avail, exclude_self=False)                      # [B, N, m]
    batch = torch.arange(B, device=pos.device)[:, None, None]
    goals = Q[batch, gidx] - pos[:, :, None, :]                               # [B, N, m, dim]
    occ = goal_occupancy(pos, Q, gidx)                                        # [B, N, m]
    if m_avail < m:
        goals = torch.cat([goals, goals.new_zeros(B, N, m - m_avail, dim)], dim=2)
        occ = torch.cat([occ, occ.new_ones(B, N, m - m_avail)], dim=2)
    dm = torch.as_tensor(d_min, dtype=pos.dtype, device=pos.device).reshape(-1, 1, 1).expand(B, N, 1)
    return torch.cat([neighbors.flatten(2), goals.flatten(2), occ, dm, vel], dim=-1)


def obs_dim(k: int, m: int, dim: int = 2) -> int:
    return dim * (k + m) + m + 1 + dim
