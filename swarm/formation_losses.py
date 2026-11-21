"""Formation (shape) losses for one goal slot per drone (M = N), with time weighting and robust forms.

Every loss is computed per frame and then combined over time by a weighting
scheme. Inputs: trajectory [B, T+1, N, 2], slots Q [B, N, 2], drone spacing
s [B]. Output: a per-scene loss [B]. Distances are measured in drone spacings
(squared-distance losses divided by s^2, energy by s), so scenes of any size
weigh the same.

Losses (lcfg.shape)
  chamfer    each drone to its nearest slot + each slot to its nearest drone;
             no one-to-one rule (the v2/v3 loss)
  matching   Hungarian one-to-one matching, then distance to the own slot
  sinkhorn   entropic optimal transport: a soft one-to-one matching
             (Cuturi 2013; GeomLoss)
  energy     energy distance (an MMD with the distance kernel; Szekely & Rizzo;
             GeomLoss "energy"): drones and slots compared as distributions,
             with built-in drone-drone repulsion
  laplacian  matching + the Laplacian formation-similarity error of Quan et
             al. (arXiv 2109.07682 / 2210.04048, IEEE T-RO 2023), which scores
             only the SHAPE (invariant to translation, rotation, scale)
  dcd        density-aware Chamfer (Wu et al., NeurIPS 2021): tried and
             dropped, its gradient vanishes for drones far from the shape
  assigned   distance to a GIVEN slot (Q already ordered): the centralized
             reference "Hungarian once from the start, then optimize the plan"

Robust form (lcfg.huber, in drone spacings; 0 = off)
  squared-distance losses: x^2 -> Huber, i.e. x^2 within delta, 2 delta |x| -
  delta^2 beyond: a drone far from its slot is pulled with a bounded force
  instead of one growing with distance.
  energy (lcfg.energy_delta): |x| -> sqrt(|x|^2 + delta^2): smooth near a slot
  (plain |x| pulls at full strength until the drone is exactly there).

Time weighting (lcfg.time_weighting) over steps t = 1..T
  final     the last step only (a terminal cost)
  last      uniform over the last hold_frac of the steps (default; v2/v3)
  every     uniform over every step (a running cost)
  terminal  half on the last step, half spread uniformly over all steps
            (running + terminal cost, the standard optimal-control form)
  discount  w_t proportional to gamma^(T - t) (lcfg.time_gamma): later steps
            weigh more, smoothly, as in discounted RL returns read backwards
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from swarm.losses import pairwise_sq_dist

KINDS = ["chamfer", "matching", "sinkhorn", "energy", "dcd", "laplacian"]
TIME_WEIGHTINGS = ["final", "last", "every", "terminal", "discount"]


# --- time weighting ---------------------------------------------------------------------------

def time_weights(T: int, scheme: str = "last", hold_frac: float = 0.2, gamma: float = 0.95):
    """(frame indices into traj, weights summing to 1). Frame 0 (the start) is never scored."""
    t = torch.arange(1, T + 1)
    if scheme == "final":
        idx, w = t[-1:], torch.ones(1)
    elif scheme == "last":
        n = max(1, round(hold_frac * T))
        idx, w = t[-n:], torch.full((n,), 1.0 / n)
    elif scheme == "every":
        idx, w = t, torch.full((T,), 1.0 / T)
    elif scheme == "terminal":
        idx, w = t, torch.full((T,), 0.5 / T)
        w[-1] += 0.5
    elif scheme == "discount":
        idx, w = t, gamma ** (T - t).float()
        w = w / w.sum()
    else:
        raise ValueError(f"unknown time weighting {scheme!r}")
    return idx, w


def hold_frames(traj: torch.Tensor, hold_frac: float) -> torch.Tensor:
    """The last hold_frac of the steps (at least the final frame): [B, n, N, dim]."""
    n = max(1, round(hold_frac * (traj.shape[1] - 1)))
    return traj[:, -n:]


def huber_sq(x2: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber of a distance given its square x2 (already in spacings^2): x^2 within delta, linear beyond."""
    if not delta:
        return x2
    x = torch.sqrt(x2 + 1e-12)
    return torch.where(x <= delta, x2, 2 * delta * x - delta ** 2)


@torch.no_grad()
def assign(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Hungarian matching per scene: Q reordered so row i is drone i's slot. P, Q [B, N, dim]."""
    D = pairwise_sq_dist(P, Q).cpu().double().numpy()
    cols = np.stack([linear_sum_assignment(d)[1] for d in D])
    idx = torch.as_tensor(cols, device=Q.device)
    return torch.gather(Q, 1, idx[..., None].expand(-1, -1, Q.shape[-1]))


# --- per-frame losses: P [B, n, N, dim] frames, return [B, n] ----------------------------------

def _flat(P, Q):
    B, n, N, dim = P.shape
    return P.reshape(B * n, N, dim), Q.repeat_interleave(n, 0), B, n


def chamfer_frames(P, Q, s, huber=0.0):
    Pf, Qf, B, n = _flat(P, Q)
    D = huber_sq(pairwise_sq_dist(Pf, Qf) / (s ** 2).repeat_interleave(n)[:, None, None], huber)
    return (D.min(2).values.mean(1) + D.min(1).values.mean(1)).reshape(B, n)


def to_slot_frames(P, Qm, s, huber=0.0):
    """Each drone to its own slot (Qm row i = drone i's slot): [B, n]."""
    d2 = ((P - Qm[:, None]) ** 2).sum(-1) / (s ** 2)[:, None, None]
    return huber_sq(d2, huber).mean(-1)


@torch.no_grad()
def _sinkhorn_plan(C: torch.Tensor, eps: torch.Tensor, iters: int = 60) -> torch.Tensor:
    """Entropic OT plan with uniform marginals 1/N, log domain, eps annealed down to `eps`. C [K, N, N], eps [K]."""
    K, N, _ = C.shape
    log_mu = torch.full((K, N), -np.log(N), dtype=C.dtype, device=C.device)
    f = torch.zeros(K, N, dtype=C.dtype, device=C.device)
    g = torch.zeros_like(f)
    big = C.amax((1, 2)).clamp(min=1e-12)
    for k in range(iters):
        e = torch.maximum(eps, big * 0.5 ** k)[:, None, None]           # anneal: coarse to fine
        f = -e[..., 0] * torch.logsumexp((g[:, None, :] - C) / e + log_mu[:, None, :], dim=2)
        g = -e[..., 0] * torch.logsumexp((f[:, :, None] - C) / e + log_mu[:, :, None], dim=1)
    e = eps[:, None, None]
    return torch.exp((f[:, :, None] + g[:, None, :] - C) / e + log_mu[:, :, None] + log_mu[:, None, :])


def sinkhorn_frames(P, Q, s, pi, huber=0.0):
    """sum_ij pi_ij cost(p_i, q_j) with a fixed transport plan pi [B, N, M] (pi sums to 1: a mean over drones)."""
    C = ((P[:, :, :, None, :] - Q[:, None, None, :, :]) ** 2).sum(-1) / (s ** 2)[:, None, None, None]
    return (pi[:, None] * huber_sq(C, huber)).sum((2, 3))


def energy_frames(P, Q, s, delta=0.0):
    Pf, Qf, B, n = _flat(P, Q)
    h2 = ((delta * s) ** 2).repeat_interleave(n)[:, None, None] + 1e-12
    dist = lambda X, Y: torch.sqrt(pairwise_sq_dist(X, Y) + h2)                  # noqa: E731
    e = 2 * dist(Pf, Qf).mean((1, 2)) - dist(Pf, Pf).mean((1, 2)) - dist(Qf, Qf).mean((1, 2))
    return e.reshape(B, n) / s[:, None]


def dcd_frames(P, Q, s, scale=2.0):
    Pf, Qf, B, n = _flat(P, Q)
    d = torch.sqrt(pairwise_sq_dist(Pf, Qf) + 1e-12)                            # [K, N drones, M slots]
    a = (1.0 / (scale * s)).repeat_interleave(n)[:, None]
    d1, j = d.min(2)
    d2, i = d.min(1)
    with torch.no_grad():
        n1 = torch.zeros_like(d1).scatter_add_(1, j, torch.ones_like(d1)).gather(1, j)
        n2 = torch.zeros_like(d2).scatter_add_(1, i, torch.ones_like(d2)).gather(1, i)
    return (0.5 * ((1 - torch.exp(-a * d1) / n1).mean(1) + (1 - torch.exp(-a * d2) / n2).mean(1))).reshape(B, n)


def normalized_laplacian(P: torch.Tensor) -> torch.Tensor:
    """L = I - D^-1/2 W D^-1/2 of the complete graph with weights W_ij = |p_i - p_j|^2. P [K, N, dim]."""
    W = pairwise_sq_dist(P, P)
    d = W.sum(2).clamp(min=1e-12)
    return torch.eye(P.shape[1], dtype=P.dtype, device=P.device) - W / torch.sqrt(d[:, :, None] * d[:, None, :])


def laplacian_error(P: torch.Tensor, Qm: torch.Tensor) -> torch.Tensor:
    """Formation similarity error of Quan et al.: ||L(P) - L(Qm)||_F^2, drone i <-> slot i. [K]"""
    return ((normalized_laplacian(P) - normalized_laplacian(Qm)) ** 2).sum((1, 2))


# --- dispatch ---------------------------------------------------------------------------------

def formation_loss(kind: str, traj, Q, s, hold_frac: float = 0.2, lcfg=None):
    """Per-scene formation loss [B] for traj [B, T+1, N, dim].

    Settings come from the `loss` config section (lcfg) with these defaults:
    time_weighting "last" (with hold_frac), huber 0, energy_delta 0,
    sinkhorn_blur 0.25, w_laplacian 1, dcd_scale 2, time_gamma 0.95.
    Matching, laplacian and sinkhorn solve their pairing / transport plan once,
    on the final frame (no gradient), and hold it over all scored frames.
    """
    g = lambda k, d: getattr(lcfg, k, d) if lcfg is not None else d  # noqa: E731
    idx, w = time_weights(traj.shape[1] - 1, g("time_weighting", "last"), hold_frac, g("time_gamma", 0.95))
    P = traj[:, idx.to(traj.device)]
    w = w.to(traj.device, traj.dtype)
    huber = g("huber", 0.0)
    if kind == "chamfer":
        per = chamfer_frames(P, Q, s, huber)
    elif kind == "assigned":
        per = to_slot_frames(P, Q, s, huber)
    elif kind == "matching":
        per = to_slot_frames(P, assign(traj[:, -1], Q), s, huber)
    elif kind == "sinkhorn":
        pi = _sinkhorn_plan(pairwise_sq_dist(traj[:, -1], Q).detach(), (g("sinkhorn_blur", 0.25) * s) ** 2)
        per = sinkhorn_frames(P, Q, s, pi, huber)
    elif kind == "energy":
        per = energy_frames(P, Q, s, g("energy_delta", 0.0))
    elif kind == "dcd":
        per = dcd_frames(P, Q, s, g("dcd_scale", 2.0))
    elif kind == "laplacian":
        Qm = assign(traj[:, -1], Q)
        B, n, N, dim = P.shape
        lap = laplacian_error(P.reshape(B * n, N, dim), Qm.repeat_interleave(n, 0)).reshape(B, n)
        per = to_slot_frames(P, Qm, s, huber) + g("w_laplacian", 1.0) * lap
    else:
        raise ValueError(f"unknown formation loss {kind!r}")
    return (per * w).sum(1)


# Thin wrappers (tests, notebooks): the plain losses with "last" weighting.
def matching_loss(traj, Q, s, hold_frac):
    return formation_loss("matching", traj, Q, s, hold_frac)


def sinkhorn_loss(traj, Q, s, hold_frac, blur: float = 0.25):
    from types import SimpleNamespace
    return formation_loss("sinkhorn", traj, Q, s, hold_frac, SimpleNamespace(sinkhorn_blur=blur))


def assigned_loss(traj, Q, s, hold_frac):
    return formation_loss("assigned", traj, Q, s, hold_frac)
