"""Run the experiment grid (swarm/experiments.py GRID) several runs at a time, resumably.

Made for a Colab GPU (notebooks/colab_grid.ipynb) but works anywhere. Each run
is its own process; a run is finished when <out>/<name>/history.json exists,
so restarting the command skips finished runs (a run cut off midway restarts).

    python -m swarm.grid list      --out OUT                       # progress
    python -m swarm.grid bench     --device cuda --jobs 1 4 8      # timing test, ~10 min
    python -m swarm.grid run       --out OUT --device cuda --jobs 8 [--only energy]
    python -m swarm.grid summarize --out OUT --device cuda          # score finished runs -> OUT/grid_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _done(out: Path, name: str) -> bool:
    return (out / name / "history.json").exists()


def _names(only=None):
    from swarm.experiments import GRID
    return [n for n in GRID if not only or all(o in n.split("_") for o in only)]


def train_one(name: str, out: str, device: str, threads: int, extra: dict | None = None):
    """Train one grid configuration (called in a subprocess)."""
    import torch
    torch.set_num_threads(threads)
    from swarm.config import load_config, set_overrides
    from swarm.dataset import SHAPE_TRAIN
    from swarm.experiments import GRID
    from swarm.train import train
    cfg = load_config(ROOT / "configs/default.yaml")
    set_overrides(cfg, GRID[name] | {"device": device} | (extra or {}))
    train(cfg, run_dir=Path(out) / name, families=SHAPE_TRAIN)


def _launch(name, out, device, threads, log_dir, extra=None):
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); from swarm.grid import train_one; "
            f"train_one({name!r}, {str(out)!r}, {device!r}, {threads}, {extra!r})")
    log = open(Path(log_dir) / f"{name}.log", "w")
    env = os.environ | {"OMP_NUM_THREADS": str(threads)}
    return subprocess.Popen([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)


def run(out, device, jobs, threads, only=None, extra=None, log_dir=None, quiet=False):
    """Run every unfinished configuration, `jobs` at a time."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir or out / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in _names(only) if not _done(out, n)]
    total, running, t0 = len(todo), [], time.time()
    if not quiet:
        print(f"{total} runs to do, {jobs} at a time on {device}", flush=True)
    while todo or running:
        while todo and len(running) < jobs:
            n = todo.pop(0)
            running.append((n, _launch(n, out, device, threads, log_dir, extra), time.time()))
        time.sleep(10)
        for item in list(running):
            n, p, start = item
            if p.poll() is not None:
                running.remove(item)
                finished = total - len(todo) - len(running)
                if not quiet:
                    eta = (time.time() - t0) / finished * (total - finished) / 3600
                    status = "ok" if p.returncode == 0 and _done(out, n) else f"FAILED (exit {p.returncode}, see {log_dir / n}.log)"
                    print(f"{time.strftime('%H:%M')}  {finished}/{total}  {n}: {status}, "
                          f"{(time.time() - start) / 60:.0f} min   (about {eta:.1f} h left)", flush=True)
    return time.time() - t0


def bench(device, jobs_list, threads, iters=40):
    """Time `iters` full-length iterations with 1, 4, 8, ... runs at once; estimate the whole grid."""
    from swarm.experiments import GRID
    names = list(GRID)
    extra = {"train.iters": iters, "train.val_every": 10 ** 9, "train.curriculum": [[0, 100]]}
    print(f"timing {iters} iterations at full length (100 steps) on {device}\n")
    best = None
    for jobs in jobs_list:
        with tempfile.TemporaryDirectory() as tmp:
            picks = [names[(i * 7) % len(names)] for i in range(jobs)]     # a mix of losses and settings
            t0 = time.time()
            procs = [_launch(n, tmp, device, threads, tmp, extra) for n in picks]
            for p in procs:
                p.wait()
            per_it = []
            for n in picks:
                h = Path(tmp, n, "history.json")
                if h.exists():
                    tr = json.loads(h.read_text())["train"]
                    per_it.append((tr[-1]["time"] - tr[0]["time"]) / (len(tr) - 1))
        if not per_it:
            print(f"{jobs} at once: all runs failed (check device / logs)")
            continue
        s = sum(per_it) / len(per_it)
        grid_h = len(names) * 3000 * s / jobs / 3600
        print(f"{jobs:2d} at once: {s:.2f} s per iteration per run -> whole grid ({len(names)} x 3,000) about {grid_h:.1f} h")
        best = min(best or (grid_h, jobs), (grid_h, jobs))
    if best:
        print(f"\nfastest: --jobs {best[1]}, about {best[0]:.1f} h (training only; full-length timing, so a slight overestimate)")


def summarize(out, device):
    """Score every finished run (best checkpoint) on the benchmark; write OUT/grid_results.json."""
    import numpy as np
    import torch
    from swarm.config import load_config
    from swarm.dataset import SHAPE_HELDOUT, SHAPE_TRAIN
    from swarm.experiments import GRID
    from swarm.policy import SharedMLP
    from swarm.train import evaluate_benchmark, overall
    out = Path(out)
    cfg = load_config(ROOT / "configs/default.yaml")
    results = {}
    path = out / "grid_results.json"
    if path.exists():
        results = json.loads(path.read_text())
    for n in GRID:
        if not _done(out, n) or n in results:
            continue
        h = json.loads((out / n / "history.json").read_text())
        late = [t for t in h["train"] if t["T"] == 100]
        g = np.array([t["grad_norm"] for t in late]) if late else np.array([np.nan])
        val = [v["filled"] for v in h["val"]]
        p = SharedMLP.load(out / n / "best.pt").to(device).eval()
        tr = evaluate_benchmark(p, cfg, "train", form=True, families=SHAPE_TRAIN)
        he = evaluate_benchmark(p, cfg, "heldout", form=True, families=SHAPE_HELDOUT)
        rows = tr + he
        results[n] = {
            "filled_train": overall(tr, "filled"), "filled_heldout": overall(he, "filled"),
            "slot_error": overall(rows, "slot_error"), "formed": overall(rows, "formed"),
            "steps_to_form": overall(rows, "steps_to_form"), "shared_goals": overall(rows, "shared_goals"),
            "close_pairs": overall(rows, "close_pairs"),
            "filled_small": float(np.mean([r["filled"] for r in rows if r["n"] <= 24])),
            "filled_large": float(np.mean([r["filled"] for r in rows if r["n"] >= 64])),
            "grad_median": float(np.nanmedian(g)), "grad_max": float(np.nanmax(g)),
            "effort": float(np.median([t["effort"] for t in late])) if late else None,
            "best_val": max(val), "final_val": val[-1], "collapsed": bool(max(val) - val[-1] > 0.15),
            "minutes": h["train"][-1]["time"] / 60,
        }
        path.write_text(json.dumps(results, indent=1))            # saved after every run: safe to interrupt
        r = results[n]
        print(f"{n:36s} filled {100 * r['filled_train']:3.0f}% / held-out {100 * r['filled_heldout']:3.0f}%"
              f"{'   COLLAPSED' if r['collapsed'] else ''}", flush=True)
    print(f"\n{len(results)} runs scored -> {path}")
    return results


def show(out):
    """Progress: finished / total, and the top runs so far if summarized."""
    out = Path(out)
    names = _names()
    done = [n for n in names if _done(out, n)]
    print(f"{len(done)}/{len(names)} runs finished in {out}")
    path = out / "grid_results.json"
    if path.exists():
        res = json.loads(path.read_text())
        top = sorted(res, key=lambda n: -(res[n]["filled_train"] + res[n]["filled_heldout"]))[:10]
        print("\ntop runs so far (mean of training and held-out slots filled):")
        for n in top:
            r = res[n]
            print(f"  {n:36s} {100 * (r['filled_train'] + r['filled_heldout']) / 2:4.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "bench", "run", "summarize"])
    ap.add_argument("--out", default="runs/grid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, nargs="+", default=[8])
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per run")
    ap.add_argument("--only", nargs="*", help="run only names containing all these parts, e.g. energy robust")
    a = ap.parse_args()
    if a.command == "list":
        show(a.out)
    elif a.command == "bench":
        bench(a.device, a.jobs, a.threads)
    elif a.command == "run":
        run(a.out, a.device, a.jobs[0], a.threads, a.only)
    else:
        summarize(a.out, a.device)


if __name__ == "__main__":
    main()
