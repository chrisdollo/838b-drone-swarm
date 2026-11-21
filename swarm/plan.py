"""Phase 1 method: optimize a fixed plan of accelerations for known scenes.

The plan is one raw number per drone, step and axis, squashed by tanh:
a = a_max * tanh(raw). Used by the Phase 1 notebook (toy optimization) and
as the per-scene optimum ("oracle") of the benchmark, which every method is
compared against.
"""

import torch

from swarm.losses import chamfer, collision_loss, shape_loss
from swarm.sim import rollout


def optimize_plan(pos0, Q, cfg, iters: int = 300, lr: float = 0.05, cosine: bool = True,
                  d_min=None, w_collision: float = 0.0, hold_frac: float = 0.0, log_every: int = 0):
    """Optimize accelerations for each scene of a batch. Scenes don't interact.

    pos0 [S, N, dim], Q [S, M, dim]. Loss: Chamfer, plus w_collision *
    collision penalty over all steps if d_min is given.
    hold_frac: share of the steps the Chamfer is averaged over, counted from
    the end. 0 = final frame only (the drones arrive just in time: nothing
    rewards being early). 1 = every step: being away from the shape costs at
    every moment, so the drones go as fast as the physics allows and then hold.
    cosine: decay the learning rate to 0 (Phase 1 lesson: a constant lr makes
    Adam oscillate and diverge near the optimum).
    Returns dict: accs [S, T, N, dim], traj [S, T+1, N, dim], history (loss per iteration).
    """
    S, N, dim = pos0.shape
    vel0 = torch.zeros_like(pos0)
    raw = torch.zeros(S, cfg.steps, N, dim, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters) if cosine else None
    history = []
    for k in range(iters):
        traj = rollout(pos0, vel0, cfg.a_max * torch.tanh(raw), cfg)
        loss = shape_loss(traj, Q, hold_frac) if hold_frac else chamfer(traj[:, -1], Q)
        if d_min is not None and w_collision:
            loss = loss + w_collision * collision_loss(traj, d_min)
        opt.zero_grad()
        loss.backward()                       # BPTT through all T steps
        opt.step()
        if sched:
            sched.step()
        history.append(loss.item())
        if log_every and (k % log_every == 0 or k == iters - 1):
            print(f"iter {k:4d}   loss {history[-1]:.6f}")
    accs = (cfg.a_max * torch.tanh(raw)).detach()
    with torch.no_grad():
        traj = rollout(pos0, vel0, accs, cfg)
    return {"accs": accs, "traj": traj, "history": history}
