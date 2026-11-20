"""Differentiable point-mass simulator.

All state tensors have shape [B, N, dim] (batch, agents, spatial dimension).
Every line uses PyTorch ops only, so autograd can backpropagate from a loss on
the final positions through every step to the accelerations (BPTT).
"""

import torch


def step(pos: torch.Tensor, vel: torch.Tensor, acc: torch.Tensor,
         dt: float, damping: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One semi-implicit Euler step with linear damping.

    v_{t+1} = (1 - c*dt) * v_t + a_t * dt
    p_{t+1} = p_t + v_{t+1} * dt

    Using the *new* velocity for the position update (semi-implicit Euler) is
    more stable than plain explicit Euler at the same dt.
    """
    vel = (1 - damping * dt) * vel + acc * dt   # new tensors, no in-place ops
    pos = pos + vel * dt
    return pos, vel


def rollout(pos0: torch.Tensor, vel0: torch.Tensor, accs: torch.Tensor,
            cfg) -> torch.Tensor:
    """Run T steps with a given acceleration sequence.

    pos0, vel0: [B, N, dim]
    accs:       [B, T, N, dim]  (in Phase 2 this becomes the policy's output)
    returns:    [B, T+1, N, dim] trajectory, including the start positions
    """
    pos, vel = pos0, vel0
    traj = [pos]
    for t in range(cfg.steps):
        pos, vel = step(pos, vel, accs[:, t], cfg.dt, cfg.damping)
        traj.append(pos)
    return torch.stack(traj, dim=1)


def rollout_closed_loop(pos0: torch.Tensor, vel0: torch.Tensor, Q: torch.Tensor, controller,
                        cfg, steps: int | None = None, tbptt: int = 0, return_state: bool = False):
    """Run T steps where the actions come from a controller that looks at the current state.

    controller(pos, vel, Q) -> acc, all [B, N, dim]. The trained policy, and
    later the Phase 3 baselines, all use this one signature.
    tbptt: if K > 0, cut the gradient chain every K steps (truncated BPTT).
    Returns traj [B, T+1, N, dim] and accs [B, T, N, dim]; with return_state,
    also the final velocity, so a run can be continued (e.g. after removing agents).
    """
    steps = cfg.steps if steps is None else steps
    pos, vel = pos0, vel0
    traj, accs = [pos], []
    for t in range(steps):
        if tbptt and t > 0 and t % tbptt == 0:
            pos, vel = pos.detach(), vel.detach()
        acc = controller(pos, vel, Q)
        pos, vel = step(pos, vel, acc, cfg.dt, cfg.damping)
        traj.append(pos)
        accs.append(acc)
    out = (torch.stack(traj, dim=1), torch.stack(accs, dim=1))
    return (*out, vel) if return_state else out


@torch.no_grad()
def rollout_with_removal(pos0, vel0, Q, controller, cfg, steps: int, remove_at: int, keep: torch.Tensor):
    """Self-healing test: run, delete some agents at step remove_at, keep running.

    keep: bool [N], which agents survive. B must be 1. Returns a list of
    steps+1 position frames ([N_t, dim]; N_t drops at remove_at) for viz.save_gif.
    """
    traj1, _, vel = rollout_closed_loop(pos0, vel0, Q, controller, cfg, steps=remove_at, return_state=True)
    pos = traj1[:, -1][:, keep]
    traj2, _ = rollout_closed_loop(pos, vel[:, keep], Q, controller, cfg, steps=steps - remove_at)
    return [f for f in traj1[0]] + [f for f in traj2[0, 1:]]
