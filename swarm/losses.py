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


def chamfer(P: torch.Tensor, Q: torch.Tensor, per_scene: bool = False) -> torch.Tensor:
    """Chamfer distance between agents P [B, N, dim] and targets Q [B, M, dim] or [M, dim].

    L = mean_i min_j D_ij  +  mean_j min_i D_ij
        (A: every agent is on the shape)  (B: every part of the shape is covered)

    No agent is assigned to a target: each min picks whichever is nearest
    right now. Returns a scalar averaged over the batch, or [B] if per_scene.
    """
    if Q.dim() == 2:
        Q = Q.expand(P.shape[0], -1, -1)
    D = pairwise_sq_dist(P, Q)
    term_a = D.min(dim=2).values.mean(dim=1)   # [B] each agent -> nearest target
    term_b = D.min(dim=1).values.mean(dim=1)   # [B] each target -> nearest agent
    return term_a + term_b if per_scene else (term_a + term_b).mean()


def close_pairs(P: torch.Tensor, d_min) -> torch.Tensor:
    """Number of agent pairs closer than d_min in each scene, [B]. d_min: float or [B]."""
    N = P.shape[1]
    d = torch.sqrt(pairwise_sq_dist(P, P)) + torch.eye(N, device=P.device) * 1e9
    d_min = torch.as_tensor(d_min, dtype=P.dtype, device=P.device).reshape(-1, 1, 1)
    return (d < d_min).sum(dim=(1, 2)) / 2


# --- Phase 2 ------------------------------------------------------------------

def shape_loss(traj: torch.Tensor, Q: torch.Tensor, hold_frac: float = 0.0, per_scene: bool = False):
    """Chamfer averaged over the last hold_frac of the rollout (just the final frame if 0).

    traj: [B, T+1, N, dim]. Scoring several final frames makes the policy
    arrive *and stay*, instead of passing through the shape at step T.
    """
    B, T1, N, dim = traj.shape
    n = max(1, round(hold_frac * (T1 - 1)))
    P = traj[:, -n:].reshape(B * n, N, dim)
    if Q.dim() == 2:
        Q = Q.expand(B, -1, -1)
    c = chamfer(P, Q.repeat_interleave(n, dim=0), per_scene=True).reshape(B, n).mean(dim=1)
    return c if per_scene else c.mean()


def collision_loss(traj: torch.Tensor, d_min, per_scene: bool = False):
    """Soft collision penalty, averaged over steps and agents (and batch unless per_scene).

    For every pair closer than d_min: (d_min - distance)^2, zero otherwise.
    Smooth, so it has useful gradients (a hard "collided yes/no" has none).
    d_min: a float, or [B] for one collision distance per scene (scaled to its density).
    """
    B, T1, N, dim = traj.shape
    P = traj[:, 1:].reshape(B * (T1 - 1), N, dim)
    d = torch.sqrt(pairwise_sq_dist(P, P) + 1e-12)               # eps: sqrt has no slope at 0
    off_diag = 1 - torch.eye(N, device=traj.device, dtype=traj.dtype)
    d_min = torch.as_tensor(d_min, dtype=traj.dtype, device=traj.device)
    if d_min.dim() == 1:                                          # per scene -> per frame
        d_min = d_min.repeat_interleave(T1 - 1).reshape(-1, 1, 1)
    pen = (torch.relu(d_min - d) ** 2 * off_diag).sum(dim=2).mean(dim=1).reshape(B, T1 - 1).mean(dim=1)
    return pen if per_scene else pen.mean()


def effort_loss(accs: torch.Tensor) -> torch.Tensor:
    """Mean squared acceleration: prefers gentle, energy-saving motion."""
    return (accs ** 2).sum(dim=-1).mean()


def total_loss(traj, accs, Q, d_min, lcfg, d_min_frac: float = 0.4):
    """Weighted Phase 2 loss. d_min: [B] collision distance per scene; lcfg: the `loss` config section.

    Shape and collision are divided by each scene's squared drone spacing
    s^2 = (d_min / d_min_frac)^2. Both scale with s^2, so without this the
    sparse 8-drone scenes would outweigh the dense 64-drone ones ~30x.
    Returns (loss tensor, dict of each term as a float for logging).
    """
    s2 = (d_min / d_min_frac) ** 2
    shape = (shape_loss(traj, Q, lcfg.hold_frac, per_scene=True) / s2).mean()
    coll = (collision_loss(traj, d_min, per_scene=True) / s2).mean()
    effort = effort_loss(accs)
    loss = shape + lcfg.w_collision * coll + lcfg.w_effort * effort
    return loss, {"shape": shape.item(), "collision": coll.item(), "effort": effort.item()}
