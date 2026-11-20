"""Run a trained policy on any scenario and save a GIF.

    uv run python -m scripts.play --ckpt runs/phase2/best.pt --shape star
    uv run python -m scripts.play --ckpt runs/phase2/best.pt --shape circle --n 64
    uv run python -m scripts.play --ckpt runs/phase2/best.pt --shape square --remove 8 --remove-at 60 --steps 160
    uv run python -m scripts.play --ckpt runs/phase2/best.pt --text "MARRY\\nME" --n 256 --steps 150
    uv run python -m scripts.play --ckpt runs/phase2/best.pt --image data/images/snowflake.jpg --crop 0,0,1,0.88 --n 256
--shape takes circle, square, line, triangle, pentagon, star, Z, M, or "random".
"""

import argparse

import torch

from swarm.config import load_config
from swarm.image_shapes import points_from_image, points_from_text
from swarm.losses import chamfer
from swarm.policy import SharedMLP
from swarm.shapes import random_shape, sample_shape
from swarm.sim import rollout_closed_loop, rollout_with_removal
from swarm.viz import save_gif


def main():
    ap = argparse.ArgumentParser(description="Run a trained policy and save a GIF.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--n", type=int, help="number of agents (default: config n_agents)")
    ap.add_argument("--steps", type=int, help="rollout length (default: config steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remove", type=int, default=0, help="agents to delete mid-run")
    ap.add_argument("--remove-at", type=int, default=60)
    ap.add_argument("--text", help='target is this text, e.g. "MARRY\\nME" (\\n = new line)')
    ap.add_argument("--image", help="target is the edges of this image file")
    ap.add_argument("--crop", help="image crop as fractions left,top,right,bottom, e.g. 0,0,1,0.88")
    ap.add_argument("--out", help="GIF path (default: results/gifs/play_<shape>_n<N>.gif)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    policy = SharedMLP.load(args.ckpt).eval()
    n = args.n or cfg.n_agents
    steps = args.steps or cfg.steps
    gen = torch.Generator().manual_seed(args.seed)

    pos0 = (torch.rand(1, n, cfg.dim, generator=gen) - 0.5)
    vel0 = torch.zeros_like(pos0)
    if args.text:
        Q, args.shape = points_from_text(args.text.replace("\\n", "\n"), 2 * n), "text"
    elif args.image:
        crop = tuple(float(x) for x in args.crop.split(",")) if args.crop else None
        Q, args.shape = points_from_image(args.image, 2 * n, crop=crop), "image"
    elif args.shape == "random":
        Q = random_shape(2 * n, cfg.dim, gen=gen)
    else:
        Q = sample_shape(args.shape, 2 * n, cfg.dim)
    Qb = Q[None]

    with torch.no_grad():
        if args.remove:
            keep = torch.ones(n, dtype=torch.bool)
            keep[torch.randperm(n, generator=gen)[:args.remove]] = False
            frames = rollout_with_removal(pos0, vel0, Qb, policy.act, cfg, steps, args.remove_at, keep)
            final = frames[-1][None]
        else:
            traj, _ = rollout_closed_loop(pos0, vel0, Qb, policy.act, cfg, steps=steps)
            frames, final = traj[0], traj[:, -1]
        err = chamfer(final, Qb).item()

    tag = f"_remove{args.remove}" if args.remove else ""
    out = args.out or f"results/gifs/play_{args.shape}_n{n}{tag}.gif"
    title = f"{args.shape}, N={n}" + (f", {args.remove} removed at step {args.remove_at}" if args.remove else "")
    save_gif(frames, Q, out, title=title)
    print(f"final Chamfer {err:.5f}  ->  {out}")


if __name__ == "__main__":
    main()
