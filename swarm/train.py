"""Phase 2 training loop: train the shared policy by backpropagating through the simulator.

Each iteration:
  1. sample a batch of scenes (random starts, random shapes)
  2. closed-loop rollout: observe -> policy -> sim.step, T times
  3. loss = shape (last frames) + collision + effort
  4. backward through every step and every policy call, clip, Adam step
Called from scripts/train.py or the Phase 2 notebook.
"""

import json
import math
import time
from pathlib import Path

import torch

from swarm.config import to_dict
from swarm.losses import chamfer, pairwise_sq_dist, total_loss
from swarm.policy import SharedMLP, policy_from_config
from swarm.shapes import TRAIN_SHAPES, random_shapes
from swarm.sim import rollout_closed_loop


def make_batch(batch: int, n_agents: int, dim: int, gen: torch.Generator, names=TRAIN_SHAPES,
               device="cpu"):
    """Random scenes: start positions uniform in the box, at rest, one random shape each (M = 2N)."""
    pos0 = torch.rand(batch, n_agents, dim, generator=gen) - 0.5
    Q = random_shapes(batch, 2 * n_agents, dim, names, gen)
    return pos0.to(device), torch.zeros(batch, n_agents, dim, device=device), Q.to(device)


def rollout_length(it: int, curriculum, T_max: int) -> int:
    """Rollout length T for iteration `it` from a [[from_iter, T], ...] schedule."""
    T = T_max
    for start, length in curriculum:
        if it >= start:
            T = length
    return min(T, T_max)


@torch.no_grad()
def evaluate(policy, pos0, vel0, Q, cfg, steps: int | None = None) -> dict:
    """Score a policy on fixed scenes at full length. All numbers are means over scenes."""
    traj, accs = rollout_closed_loop(pos0, vel0, Q, policy.act, cfg, steps=steps)
    final = traj[:, -1]
    N = final.shape[1]
    d = torch.sqrt(pairwise_sq_dist(final, final))
    d = d + torch.eye(N, device=d.device) * 1e9
    close_pairs = (d < cfg.loss.d_min).float().sum(dim=(1, 2)) / 2
    return {
        "chamfer": chamfer(final, Q).item(),                    # final-frame shape error
        "min_dist": d.amin(dim=(1, 2)).mean().item(),           # closest pair at the end
        "collisions": close_pairs.mean().item(),                # pairs closer than d_min at the end
        "speed": traj[:, -1].sub(traj[:, -2]).norm(dim=-1).mean().item() / cfg.dt,  # ~0 if settled
    }


def train(cfg, run_dir=None, callback=None, callback_every: int = 50, policy: SharedMLP | None = None):
    """Train a policy. Returns (policy, history).

    run_dir: where to save best.pt, last.pt and history.json (None: don't save).
    callback(history): called every callback_every iterations, e.g. to redraw
    live plots in a notebook.
    """
    tc = cfg.train
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    policy = (policy or policy_from_config(cfg)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=tc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.iters, eta_min=tc.lr_min)

    val_gen = torch.Generator().manual_seed(tc.val_seed)
    val_batch = make_batch(tc.val_scenes, cfg.n_agents, cfg.dim, val_gen, device=device)

    history = {"train": [], "val": []}
    best = math.inf
    run_dir = Path(run_dir) if run_dir else None
    start = time.time()

    def validate(it):
        nonlocal best
        val = {"it": it, **evaluate(policy, *val_batch, cfg)}
        history["val"].append(val)
        if run_dir and val["chamfer"] < best:
            best = val["chamfer"]
            policy.save(run_dir / "best.pt", config=to_dict(cfg), it=it, val=val)

    for it in range(tc.iters):
        if it % tc.val_every == 0:                    # before the update: it 0 = untrained policy
            validate(it)
        T = rollout_length(it, tc.curriculum, cfg.steps)
        pos0, vel0, Q = make_batch(tc.batch_size, cfg.n_agents, cfg.dim, gen, device=device)

        traj, accs = rollout_closed_loop(pos0, vel0, Q, policy.act, cfg, steps=T, tbptt=tc.tbptt)
        loss, parts = total_loss(traj, accs, Q, cfg.loss)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), tc.grad_clip).item()
        if math.isfinite(grad_norm):                  # skip the update if gradients blew up
            opt.step()
        sched.step()

        history["train"].append({"it": it, "T": T, "loss": loss.item(), **parts,
                                 "grad_norm": grad_norm, "lr": sched.get_last_lr()[0],
                                 "time": time.time() - start})

        if it == tc.iters - 1:
            validate(tc.iters)                        # after the last update
        if callback and (it % callback_every == 0 or it == tc.iters - 1):
            callback(history)

    if run_dir:
        policy.save(run_dir / "last.pt", config=to_dict(cfg), it=tc.iters - 1)
        (run_dir / "history.json").write_text(json.dumps(history))
    return policy, history
