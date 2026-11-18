"""Phase 1 checkpoint: optimize the acceleration sequence directly (no network).

If the agents form the target shape, the simulator, the loss and autograd are
all correct.

Run from the repo root:
    uv run python -m scripts.toy_optimize
    uv run python -m scripts.toy_optimize --shape square --iters 500
"""

import argparse
import math
import time

import torch

from swarm.config import load_config
from swarm.losses import chamfer
from swarm.shapes import sample_shape
from swarm.sim import rollout
from swarm.viz import plot_loss, save_gif


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    N, T, D = cfg.n_agents, cfg.steps, cfg.dim

    # 1. Random start positions in the box [-0.5, 0.5], at rest. B = 1.
    pos0 = torch.rand(1, N, D) - 0.5
    vel0 = torch.zeros(1, N, D)
    Q = sample_shape(args.shape, 2 * N, D)               # [M, D], M = 2N

    # 2. The parameters: one raw number per agent, step and axis (math page section 3).
    raw = torch.zeros(1, T, N, D, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=args.lr)
    # Decay the learning rate to 0. Adam steps every parameter by about lr no matter
    # how small its gradient is, and one aligned step of 0.05 can move a final
    # position by up to a_max * lr * sum(w_s) ~ 0.64 units, far more than the 0.04
    # target spacing. With a constant lr the loss reaches ~6e-4, then oscillates
    # and climbs back up to ~3e-2 by iteration 300.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    history = []
    start = time.time()
    for k in range(args.iters):
        accs = cfg.a_max * torch.tanh(raw)                # a = a_max tanh(r), always in [-a_max, a_max]
        traj = rollout(pos0, vel0, accs, cfg)             # [1, T+1, N, D]
        loss = chamfer(traj[:, -1], Q)                    # loss on the final positions only

        opt.zero_grad()
        loss.backward()                                   # BPTT through all T steps
        opt.step()
        sched.step()

        history.append(loss.item())                       # .item() is fine here: outside the sim loop
        if k % 10 == 0 or k == args.iters - 1:
            print(f"iter {k:4d}   loss {history[-1]:.6f}")

    # Final rollout with the optimized actions (no gradients needed).
    with torch.no_grad():
        traj = rollout(pos0, vel0, cfg.a_max * torch.tanh(raw), cfg)
        final = chamfer(traj[:, -1], Q).item()

    # Best possible loss for M = 2N circle points (math page section 5): each agent
    # covers two neighboring points s apart and sits a quarter of the way between
    # them, giving A = s^2/16 and B = 5 s^2/16, so L = 3/8 s^2 (curvature ignored).
    floor = None
    if args.shape == "circle":
        spacing = 2 * 0.4 * math.sin(math.pi / (2 * N))
        floor = 3 / 8 * spacing**2

    print(f"\nfinal loss {final:.6f}  (start {history[0]:.4f}, {time.time() - start:.1f} s)")
    if floor is not None:
        print(f"best possible ≈ {floor:.6f}")

    gif = save_gif(traj[0], Q, f"results/gifs/toy_{args.shape}.gif",
                   title=f"Toy optimization: {N} agents → {args.shape}")
    fig = plot_loss(history, f"results/figures/toy_{args.shape}_loss.png",
                    title=f"Toy optimization ({args.shape})", floor=floor)
    print(f"saved {gif} and {fig}")


if __name__ == "__main__":
    main()
