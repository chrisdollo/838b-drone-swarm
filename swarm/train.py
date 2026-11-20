"""Phase 2 training loop: train the shared policy by backpropagating through the simulator.

Each iteration:
  1. pick a drones x goals cell (N, M) from the grid
  2. sample a batch of scenes for it (random starts, random training shapes, d_min per scene)
  3. closed-loop rollout: observe -> policy -> sim.step, T times
  4. loss = shape (last frames) + collision + effort, each scene normalized by its spacing^2
  5. backward through every step and every policy call, clip, Adam step
Validation scores the fixed benchmark against its per-scene optimum.
Called from notebooks/phase2.ipynb.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from swarm.config import to_dict
from swarm.dataset import D_MIN_FRAC, GRID, TRAIN_FAMILIES, load_cell, make_batch
from swarm.losses import chamfer, close_pairs, total_loss
from swarm.policy import SharedMLP, policy_from_config
from swarm.sim import rollout_closed_loop


def rollout_length(it: int, curriculum, T_max: int) -> int:
    """Rollout length T for iteration `it` from a [[from_iter, T], ...] schedule."""
    T = T_max
    for start, length in curriculum:
        if it >= start:
            T = length
    return min(T, T_max)


@torch.no_grad()
def evaluate_cell(policy, cell: dict, cfg, steps: int | None = None) -> dict:
    """Run a controller on every scene of a benchmark cell. Per-scene tensors.

    ratio = (Chamfer + eps) / (per-scene optimum + eps), with eps = 1% of the
    scene's spacing^2. 1.0 = as good as optimizing that scene directly. The
    floor keeps the ratio meaningful when the optimum is ~0 (e.g. 8 drones
    sitting exactly on 8 goals): errors far below a drone spacing don't count.
    """
    controller = policy.act if hasattr(policy, "act") else policy
    p0, Q, d_min = cell["start"], cell["goals"], cell["d_min"]
    traj, _ = rollout_closed_loop(p0, torch.zeros_like(p0), Q, d_min, controller, cfg, steps=steps)
    final = traj[:, -1]
    ch = chamfer(final, Q, per_scene=True)
    eps = 0.01 * (d_min / D_MIN_FRAC) ** 2
    return {"chamfer": ch, "ratio": (ch + eps) / (cell["oracle_chamfer"] + eps), "close_pairs": close_pairs(final, d_min),
            "speed": (traj[:, -1] - traj[:, -2]).norm(dim=-1).mean(dim=1) / cfg.dt, "traj": traj}


def evaluate_benchmark(policy, cfg, split: str = "train", cells=GRID) -> list[dict]:
    """Score a controller on every cell of the benchmark. One summary row per cell."""
    rows = []
    for n, m in cells:
        cell = load_cell(n, m, split)
        r = evaluate_cell(policy, cell, cfg)
        rows.append({"n": n, "m": m, "scenes": len(cell["family"]),
                     "ratio": float(torch.exp(torch.log(r["ratio"]).mean())),      # geometric mean
                     "chamfer": float(r["chamfer"].mean()), "oracle": float(cell["oracle_chamfer"].mean()),
                     "close_pairs": float(r["close_pairs"].mean()),
                     "oracle_close_pairs": float(cell["oracle_close_pairs"].mean())})
    return rows


def overall_ratio(rows) -> float:
    """Geometric mean of the per-cell ratios: one number for the whole grid."""
    return float(np.exp(np.mean([np.log(r["ratio"]) for r in rows])))


def train(cfg, run_dir=None, callback=None, callback_every: int = 50, policy: SharedMLP | None = None):
    """Train a policy on the whole drones x goals grid. Returns (policy, history).

    run_dir: where to save best.pt, last.pt and history.json (None: don't save).
    callback(history): called every callback_every iterations (live plots in the notebook).
    """
    tc = cfg.train
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    policy = policy or policy_from_config(cfg)
    opt = torch.optim.Adam(policy.parameters(), lr=tc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.iters, eta_min=tc.lr_min)

    history = {"train": [], "val": []}
    best = math.inf
    run_dir = Path(run_dir) if run_dir else None
    start = time.time()

    def validate(it):
        nonlocal best
        rows = evaluate_benchmark(policy, cfg, "train")
        val = {"it": it, "ratio": overall_ratio(rows), "cells": rows,
               "close_pairs": float(np.mean([r["close_pairs"] for r in rows]))}
        history["val"].append(val)
        if run_dir and val["ratio"] < best:
            best = val["ratio"]
            policy.save(run_dir / "best.pt", config=to_dict(cfg), it=it, val_ratio=val["ratio"])

    for it in range(tc.iters):
        if it % tc.val_every == 0:                    # before the update: it 0 = untrained policy
            validate(it)
        n, m = GRID[int(rng.integers(len(GRID)))]
        T = rollout_length(it, tc.curriculum, cfg.steps)
        pos0, vel0, Q, d_min = make_batch(tc.batch_size, n, m, rng, TRAIN_FAMILIES)

        traj, accs = rollout_closed_loop(pos0, vel0, Q, d_min, policy.act, cfg, steps=T, tbptt=tc.tbptt)
        loss, parts = total_loss(traj, accs, Q, d_min, cfg.loss)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), tc.grad_clip).item()
        if math.isfinite(grad_norm):                  # skip the update if gradients blew up
            opt.step()
        sched.step()

        history["train"].append({"it": it, "n": n, "m": m, "T": T, "loss": loss.item(), **parts,
                                 "grad_norm": grad_norm, "lr": sched.get_last_lr()[0],
                                 "time": time.time() - start})
        if it == tc.iters - 1:
            validate(tc.iters)                        # after the last update
        if callback and (it % callback_every == 0 or it == tc.iters - 1):
            callback(history)

    if run_dir:
        policy.save(run_dir / "last.pt", config=to_dict(cfg), it=tc.iters)
        (run_dir / "history.json").write_text(json.dumps(history))
    return policy, history
