"""Slot metrics for drones = goals (M = N): one drone per goal, checked by one-to-one matching.

Chamfer only asks "is every goal near some drone and every drone near some
goal"; two drones can share one goal while another stays empty. These metrics
match drones to goals one-to-one (Hungarian algorithm, minimizing the total
distance) and measure each drone against its own goal. All distances are in
drone spacings s = d_min / D_MIN_FRAC, so scenes of any size compare.

    slot error   mean matched distance / s                          (0 = every goal hit exactly)
    slots filled share of goals whose matched drone is within d_min  (1 = all filled)
    steps to form first step after which the slot error stays below FORM_THRESHOLD
                 for the rest of the run (nan = never formed)
    shared goals share of goals that are the nearest goal of 2+ drones (pile-ups;
                 not matching-based, it shows *why* slots are empty)
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

D_MIN_FRAC = 0.4       # d_min = 0.4 x drone spacing (as in swarm/dataset.py)
FORM_THRESHOLD = 0.5   # "formed" = slot error below half a drone spacing


def matched_distances(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """[S, N, 2] drones, [S, M, 2] goals -> [S, min(N, M)] distances of the optimal one-to-one matching."""
    P, Q = P.detach().cpu().double(), Q.detach().cpu().double()
    D = torch.cdist(P, Q).numpy()
    out = []
    for d in D:
        r, c = linear_sum_assignment(d)
        out.append(d[r, c])
    return torch.tensor(np.stack(out), dtype=torch.float32)


def _spacing(d_min) -> torch.Tensor:
    return torch.as_tensor(d_min, dtype=torch.float32).reshape(-1) / D_MIN_FRAC


def slot_metrics(final: torch.Tensor, Q: torch.Tensor, d_min, anchor_mask=None) -> dict:
    """Per-scene slot error and slots filled at one frame (the last step). Tensors [S].

    anchor_mask [S, M] (optional, True = goal at a stroke end/corner/junction):
    adds the filled share at anchors and at the even fill dots separately.
    """
    S, N, _ = final.shape
    s = _spacing(d_min)
    P, Q = final.detach().cpu().double(), Q.detach().cpu().double()
    D = torch.cdist(P, Q).numpy()
    dist, filled = np.zeros((S, Q.shape[1])), np.zeros((S, Q.shape[1]), bool)
    for k, d in enumerate(D):
        r, c = linear_sum_assignment(d)
        dist[k, c] = d[r, c]
        filled[k, c] = d[r, c] <= float(D_MIN_FRAC * s[k])
    dist, filled = torch.tensor(dist, dtype=torch.float32), torch.tensor(filled)
    nearest = D.argmin(axis=2)                                  # each drone's nearest goal
    shared = torch.tensor([(np.bincount(g, minlength=Q.shape[1]) >= 2).mean() for g in nearest], dtype=torch.float32)
    out = {"slot_error": (dist / s[:, None]).mean(1), "filled": filled.float().mean(1), "shared_goals": shared}
    if anchor_mask is not None:
        a = torch.as_tensor(anchor_mask, dtype=torch.bool)
        f = filled.float()
        out["filled_anchor"] = (f * a).sum(1) / a.sum(1).clamp(min=1)
        out["filled_fill"] = (f * ~a).sum(1) / (~a).sum(1).clamp(min=1)
    return out


def slot_error_curve(traj: torch.Tensor, Q: torch.Tensor, d_min) -> torch.Tensor:
    """[S, T+1, N, 2] trajectories -> [S, T+1] slot error at every step (in spacings)."""
    S, T1 = traj.shape[:2]
    flat = matched_distances(traj.reshape(S * T1, *traj.shape[2:]),
                             Q[:, None].expand(S, T1, *Q.shape[1:]).reshape(S * T1, *Q.shape[1:]))
    return flat.reshape(S, T1, -1).mean(-1) / _spacing(d_min)[:, None]


def steps_to_form(curve: torch.Tensor, threshold: float = FORM_THRESHOLD) -> torch.Tensor:
    """First step from which the slot error stays below `threshold` until the end. [S] float, nan = never."""
    below = curve < threshold
    stays = torch.flip(torch.cumprod(torch.flip(below, [1]).int(), 1), [1]).bool()   # below from t to the end
    first = stays.float().argmax(1).float()
    first[~stays[:, -1]] = float("nan")
    return first
