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
