"""Phase 2 training loop: train the shared policy by backpropagating through the simulator.

Each iteration:
  1. draw the swarm size N uniformly from N_RANGE (8-96); every scene has N goal slots (M = N)
  2. sample a batch of scenes (random starts, random training shapes, d_min per scene)
  3. closed-loop rollout: observe -> policy -> sim.step, T times
  4. loss = shape (last frames) + collision + effort, each scene normalized by its spacing^2
  5. backward through every step and every policy call, clip, Adam step
Validation scores the fixed benchmark (data/benchmark_v3); the best checkpoint
and early stopping use the mean slot error (swarm/metrics.py).
Called from notebooks/phase2.ipynb.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from swarm.config import to_dict
from swarm.dataset import BENCH_DIR, BENCH_N, D_MIN_FRAC, N_RANGE, TRAIN_FAMILIES, load_cell, make_batch, sample_n
from swarm.losses import chamfer, close_pairs, spacing_cv, total_loss
from swarm.metrics import slot_error_curve, slot_metrics, steps_to_form
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
def evaluate_cell(policy, cell: dict, cfg, steps: int | None = None, form: bool = False) -> dict:
    """Run a controller on every scene of a benchmark cell. Per-scene tensors.

    slot_error / filled / shared_goals: one-to-one matching at the last step (swarm/metrics.py).
    ratio = (Chamfer + eps) / (per-scene optimum + eps), eps = 1% of spacing^2
    (secondary with M = N: the optimum is ~0, so this is mostly Chamfer / eps).
    form=True also computes the slot error at every step and steps_to_form (slower).
    """
    controller = policy.act if hasattr(policy, "act") else policy
    dev = next(policy.parameters()).device if hasattr(policy, "parameters") else cell["start"].device
    p0, Q, d_min = (cell[k].to(dev) for k in ("start", "goals", "d_min"))
    traj, _ = rollout_closed_loop(p0, torch.zeros_like(p0), Q, d_min, controller, cfg, steps=steps)
    traj, p0, Q, d_min = traj.cpu(), p0.cpu(), Q.cpu(), d_min.cpu()         # scoring on the CPU
    final = traj[:, -1]
    ch = chamfer(final, Q, per_scene=True)
    eps = 0.01 * (d_min / D_MIN_FRAC) ** 2
    out = {"chamfer": ch, "close_pairs": close_pairs(final, d_min),
           "uniformity": spacing_cv(final), **slot_metrics(final, Q, d_min, cell.get("anchor")),
           "speed": (traj[:, -1] - traj[:, -2]).norm(dim=-1).mean(dim=1) / cfg.dt, "traj": traj}
    if "oracle_chamfer" in cell:                     # benchmark cells; one-off scenes have no optimum
        out["ratio"] = (ch + eps) / (cell["oracle_chamfer"] + eps)
    if form:
        out["curve"] = slot_error_curve(traj, Q, d_min)
        out["steps_to_form"] = steps_to_form(out["curve"])
    return out


def _nanmedian(x):
    x = x[~torch.isnan(x)]
    return float(x.median()) if len(x) else float("nan")


def evaluate_benchmark(policy, cfg, split: str = "train", ns=BENCH_N, root=BENCH_DIR, form: bool = False,
                       families=None) -> list[dict]:
    """Score a controller on every cell of the benchmark. One summary row per cell (means over its scenes).

    families: score only scenes of these families (e.g. dataset.SHAPE_TRAIN)."""
    rows = []
    for n in ns:
        cell = load_cell(n, split, root, families)
        if not cell["family"]:
            continue
        r = evaluate_cell(policy, cell, cfg, form=form)
        row = {"n": n, "scenes": len(cell["family"]),
               "slot_error": float(r["slot_error"].mean()), "filled": float(r["filled"].mean()),
               "filled_anchor": float(r["filled_anchor"].mean()), "filled_fill": float(r["filled_fill"].mean()),
               "shared_goals": float(r["shared_goals"].mean()),
               "ratio": float(torch.exp(torch.log(r["ratio"]).mean())),      # geometric mean
               "chamfer": float(r["chamfer"].mean()), "close_pairs": float(r["close_pairs"].mean()),
               "uniformity": float(r["uniformity"].mean())}
        for k in ("chamfer", "close_pairs", "slot_error", "filled", "filled_anchor", "filled_fill", "shared_goals"):
            row[f"oracle_{k}"] = float(cell[f"oracle_{k}"].float().mean())
        row["oracle_uniformity"] = float(spacing_cv(cell["oracle_final"]).mean())
        if form:
            row["formed"] = float((~torch.isnan(r["steps_to_form"])).float().mean())   # share that formed at all
            row["steps_to_form"] = _nanmedian(r["steps_to_form"])                       # median over those
        rows.append(row)
    return rows


def overall(rows, key: str = "slot_error") -> float:
    """One number for the whole benchmark: mean over cells (geometric mean for the ratio)."""
    vals = [r[key] for r in rows]
    return float(np.exp(np.mean(np.log(vals)))) if key == "ratio" else float(np.nanmean(vals))


def overall_ratio(rows) -> float:
    return overall(rows, "ratio")


def train(cfg, run_dir=None, callback=None, callback_every: int = 50, policy: SharedMLP | None = None,
          families=None):
    """Train a policy on the whole drones x goals grid. Returns (policy, history).

    run_dir: where to save best.pt, last.pt and history.json (None: don't save).
    callback(history): called every callback_every iterations (live plots in the notebook).
    families: training shape families (default dataset.TRAIN_FAMILIES); the
    benchmark validation uses the same families.
    Validation score = cfg.train.select over the benchmark cells (training
    shapes): "slot_error" (mean, lower is better) or "filled" (scored as
    1 - slots filled).
    Early stopping: stop when it has not improved by at least train.min_delta
    (relative) for train.patience validations in a row.
    """
    tc = cfg.train
    families = families or TRAIN_FAMILIES
    select = getattr(tc, "select", "slot_error")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(getattr(cfg, "device", "cpu"))
    policy = (policy or policy_from_config(cfg)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=tc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.iters, eta_min=tc.lr_min)

    history = {"train": [], "val": [], "stopped_at": None}
    best = math.inf
    patience, min_delta = getattr(tc, "patience", 0), getattr(tc, "min_delta", 0.0)
    since_best = 0
    run_dir = Path(run_dir) if run_dir else None
    start = time.time()

    def validate(it):
        """Score the benchmark; save the best weights. Returns True if training should stop early."""
        nonlocal best, since_best
        rows = evaluate_benchmark(policy, cfg, "train", families=families)
        score = 1 - overall(rows, "filled") if select == "filled" else overall(rows, "slot_error")
        val = {"it": it, "score": score, "slot_error": overall(rows, "slot_error"),
               "filled": overall(rows, "filled"), "ratio": overall_ratio(rows), "cells": rows,
               "close_pairs": overall(rows, "close_pairs"), "uniformity": overall(rows, "uniformity")}
        history["val"].append(val)
        if val["score"] < best * (1 - min_delta):
            since_best = 0
        else:
            since_best += 1
        if val["score"] < best:
            best = val["score"]
            if run_dir:
                policy.save(run_dir / "best.pt", config=to_dict(cfg), it=it, val_score=val["score"],
                            val_filled=val["filled"])
        return bool(patience) and since_best >= patience

    for it in range(tc.iters):
        if it % tc.val_every == 0 and validate(it):   # before the update: it 0 = untrained policy
            history["stopped_at"] = it                # early stop: no improvement for `patience` checks
            if callback:
                callback(history)
            break
        n = sample_n(rng, N_RANGE)
        T = rollout_length(it, tc.curriculum, cfg.steps)
        pos0, vel0, Q, d_min = make_batch(tc.batch_size, n, rng, families, device=device)

        traj, accs = rollout_closed_loop(pos0, vel0, Q, d_min, policy.act, cfg, steps=T, tbptt=tc.tbptt,
                                           grad_decay=getattr(tc, "grad_decay", 0.0))
        loss, parts = total_loss(traj, accs, Q, d_min, cfg.loss)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), tc.grad_clip).item()
        if math.isfinite(grad_norm):                  # skip the update if gradients blew up
            opt.step()
        sched.step()

        history["train"].append({"it": it, "n": n, "T": T, "loss": loss.item(), **parts,
                                 "grad_norm": grad_norm, "lr": sched.get_last_lr()[0],
                                 "time": time.time() - start})
        if it == tc.iters - 1:
            validate(tc.iters)                        # after the last update
        if callback and (it % callback_every == 0 or it == tc.iters - 1):
            callback(history)

    if run_dir:
        policy.save(run_dir / "last.pt", config=to_dict(cfg), it=history["train"][-1]["it"] + 1)
        (run_dir / "history.json").write_text(json.dumps(history))
    return policy, history
