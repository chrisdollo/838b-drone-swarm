"""Build the fixed benchmark (every drones x goals cell, train and held-out shapes) and review figures.

    uv run python -m scripts.build_benchmark

Writes data/benchmark/n<N>_m<M>_<split>.pt + index.json, and
results/figures/benchmark_overview.png, results/gifs/benchmark_oracle.gif.
Every scene also gets its per-scene optimum ("oracle"): the Phase 1 method,
optimizing that scene's accelerations directly. Methods are later reported
as a ratio to it.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from swarm.config import load_config  # noqa: E402
from swarm.dataset import (GRID, HELDOUT_FAMILIES, TRAIN_FAMILIES, build_benchmark,  # noqa: E402
                           load_cell, make_scene, oracle)
from swarm.viz import AGENT, TARGET, save_gif_grid  # noqa: E402


def overview_figure(path="results/figures/benchmark_overview.png"):
    fams = TRAIN_FAMILIES + HELDOUT_FAMILIES
    fig, axs = plt.subplots(len(GRID), len(fams), figsize=(1.35 * len(fams) + 1.2, 1.35 * len(GRID)), dpi=110)
    for r, (n, m) in enumerate(GRID):
        cells = {s: load_cell(n, m, s) for s in ("train", "heldout")}
        for c, fam in enumerate(fams):
            ax = axs[r, c]
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.85")
            cell = cells["train" if fam in TRAIN_FAMILIES else "heldout"]
            if r == 0:
                ax.set_title(fam.replace("random_", "rand. ") + ("*" if fam in HELDOUT_FAMILIES else ""), fontsize=8)
            if fam in cell["skipped"]:
                ax.text(0, 0, "too few\ndrones", ha="center", va="center", fontsize=7, color="0.55")
                continue
            i = cell["family"].index(fam)
            Q, P = cell["goals"][i], cell["oracle_final"][i]
            ax.scatter(Q[:, 0], Q[:, 1], s=3, facecolors="none", edgecolors=TARGET, lw=0.5)
            ax.scatter(P[:, 0], P[:, 1], s=4 if n >= 32 else 7, color=AGENT, zorder=3)
        axs[r, 0].set_ylabel(f"N={n}\nM={m}", rotation=0, ha="right", va="center", fontsize=8, labelpad=12)
    fig.suptitle("Benchmark: one example per cell and family. Green = goals, blue = per-scene optimum "
                 "(Phase 1 method). * = held out from training", fontsize=10, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path)
    plt.close(fig)
    return path


def oracle_gif(cfg, path="results/gifs/benchmark_oracle.gif"):
    """Per-scene optimum for a few cells, to show what 'best possible' looks like at each density."""
    picks = [("circle", 8, 8), ("letters", 16, 32), ("heart", 32, 32), ("star", 32, 128 // 2),
             ("word", 64, 64), ("spiral", 64, 128)]
    rng = np.random.default_rng(7)
    trajs, goals, titles = [], [], []
    for fam, n, m in picks:
        sc = make_scene(fam, n, m, rng)
        cell = {"start": torch.tensor(sc["start"][None], dtype=torch.float32),
                "goals": torch.tensor(sc["goals"][None], dtype=torch.float32),
                "d_min": torch.tensor([sc["d_min"]], dtype=torch.float32)}
        o = oracle(cell, cfg)
        trajs.append(o["oracle_traj_head"][0])
        goals.append(cell["goals"][0])
        titles.append(f"{fam}, N={n}, M={m}")
    return save_gif_grid(trajs, goals, path, titles=titles, cols=3, panel=3.0)


def main():
    cfg = load_config()
    t = time.time()
    index = build_benchmark(cfg, log=lambda row: print(
        f"N={row['n']:3d} M={row['m']:4d} {row['split']:8s} scenes {row['scenes']:3d}  "
        f"optimum Chamfer {row['oracle_chamfer']:.6f}  close pairs {row['oracle_close_pairs']:.2f}  "
        f"d_min {row['d_min_mean']:.3f}  skipped {row['skipped']}", flush=True))
    print(f"built {sum(r['scenes'] for r in index)} scenes in {time.time() - t:.0f} s")
    print("saved", overview_figure())
    print("saved", oracle_gif(cfg))


if __name__ == "__main__":
    main()
