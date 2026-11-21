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


def rollout_closed_loop(pos0: torch.Tensor, vel0: torch.Tensor, Q: torch.Tensor, d_min, controller,
                        cfg, steps: int | None = None, tbptt: int = 0, grad_decay: float = 0.0,
                        return_state: bool = False):
    """Run T steps where the actions come from a controller that looks at the current state.

    controller(pos, vel, Q, d_min) -> acc [B, N, dim]; d_min [B] is each scene's
    collision distance. The trained policy, and later the Phase 3 baselines,
    all use this one signature.
    tbptt: if K > 0, cut the gradient chain every K steps (truncated BPTT).
    grad_decay: if 0 < a < 1, temporal gradient decay (Zhang et al. 2024, "Back
    to Newton's Laws"): the state carried to the next step is
    a * x + (1 - a) * x.detach(), same value, but the gradient flowing back
    through it shrinks by a per step. Losses k steps later reach an action
    with weight a^k: no hard cut, and the product of step Jacobians can't explode.
    Returns traj [B, T+1, N, dim] and accs [B, T, N, dim]; with return_state,
    also the final velocity, so a run can be continued (e.g. after removing agents).
    """
    steps = cfg.steps if steps is None else steps
    pos, vel = pos0, vel0
    traj, accs = [pos], []
    for t in range(steps):
        if tbptt and t > 0 and t % tbptt == 0:
            pos, vel = pos.detach(), vel.detach()
        if grad_decay and t > 0:
            pos = grad_decay * pos + (1 - grad_decay) * pos.detach()
            vel = grad_decay * vel + (1 - grad_decay) * vel.detach()
        acc = controller(pos, vel, Q, d_min)
        pos, vel = step(pos, vel, acc, cfg.dt, cfg.damping)
        traj.append(pos)
        accs.append(acc)
    out = (torch.stack(traj, dim=1), torch.stack(accs, dim=1))
    return (*out, vel) if return_state else out

