"""Plots and GIFs. Nothing here needs gradients, so tensors are detached first."""

from pathlib import Path

import imageio.v3 as iio
import matplotlib

if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")   # scripts write files; notebooks keep their inline backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AGENT = "#2F6DB5"
TARGET = "#13867A"
WARN = "#C8402F"


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _frames(traj):
    """[T+1, N, dim] tensor or list of [N_t, dim] -> list of [N_t, 2] arrays."""
    return [_np(f)[..., :2] for f in traj]


def _setup(ax, Q, title=""):
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ls="--", lw=0.8, color="0.6"))
    Q = _np(Q)[..., :2]
    ax.scatter(Q[:, 0], Q[:, 1], s=12, facecolors="none", edgecolors=TARGET, lw=0.8)
    if title:
        ax.set_title(title, fontsize=9)


def save_gif_grid(trajs, Qs, path, titles=None, cols: int = 4, every: int = 2, fps: int = 20,
                  hold_s: float = 1.0, panel: float = 2.6) -> Path:
    """Animate several rollouts side by side, all on the same clock.

    trajs: list of trajectories ([T+1, N, dim] or lists of [N_t, dim] frames,
    e.g. with agents removed mid-run); Qs: list of [M, dim] targets.
    """
    trajs = [_frames(t) for t in trajs]
    S = len(trajs)
    cols = min(cols, S)
    rows = -(-S // cols)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(rows, cols, figsize=(panel * cols, panel * rows), dpi=100, squeeze=False)
    dots = []
    for i, ax in enumerate(axs.flat):
        if i >= S:
            ax.axis("off")
            continue
        _setup(ax, Qs[i], titles[i] if titles else "")
        dots.append(ax.scatter(*trajs[i][0].T, s=14, color=AGENT, zorder=3))
    clock = fig.text(0.01, 0.99, "", va="top", fontsize=8, color="0.35")
    fig.tight_layout()

    T1 = max(len(t) for t in trajs)
    idx = list(range(0, T1, every))
    if idx[-1] != T1 - 1:
        idx.append(T1 - 1)
    frames = []
    for t in idx:
        for d, tr in zip(dots, trajs):
            d.set_offsets(tr[min(t, len(tr) - 1)])
        clock.set_text(f"step {t}/{T1 - 1}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    frames += [frames[-1]] * int(hold_s * fps)
    iio.imwrite(path, np.stack(frames), duration=1000 / fps, loop=0)   # duration in ms
    return path


def save_gif(traj, Q, path, every: int = 2, fps: int = 20, hold_s: float = 1.0, title: str = "") -> Path:
    """Animate one rollout. traj: [T+1, N, dim] (or a list of frames); Q: [M, dim]."""
    return save_gif_grid([traj], [Q], path, titles=[title] if title else None, cols=1,
                         every=every, fps=fps, hold_s=hold_s, panel=4.0)


def plot_training(history, path=None):
    """Phase 2 training curves. Returns the figure.

    Left: normalized training shape loss and the validation score (v3: mean
    slot error in drone spacings; v2 histories: ratio to the per-scene optimum).
    Middle: penalties. Right: gradient norm and curriculum.
    """
    tr, val = history["train"], history["val"]
    it = [h["it"] for h in tr]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), dpi=100)

    ax = axs[0]
    ax.semilogy(it, [h["shape"] for h in tr], color=AGENT, lw=0.6, alpha=0.6, label="shape loss / s² (train)")
    if val and "slot_error" in val[0]:
        ax.semilogy([v["it"] for v in val], [v["slot_error"] for v in val], "o-", color=TARGET, ms=4, lw=1.6,
                    label="benchmark: slot error (spacings)")
        ax.axhline(0.5, color="0.5", ls=":", lw=1, label="formed (0.5 spacings)")
    elif val:
        ax.semilogy([v["it"] for v in val], [v["ratio"] for v in val], "o-", color=TARGET, ms=4, lw=1.6,
                    label="benchmark: × per-scene optimum")
        ax.axhline(1, color="0.5", ls=":", lw=1)
    ax.set_title("Shape error", fontsize=10)
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    ax.semilogy(it, [max(h["collision"], 1e-9) for h in tr], color=WARN, lw=0.8, label="collision / s²")
    ax.semilogy(it, [h["effort"] for h in tr], color="0.4", lw=0.8, label="effort ||a||²")
    ax.set_title("Penalties (unweighted)", fontsize=10)
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axs[2]
    ax.semilogy(it, [h["grad_norm"] for h in tr], color="0.3", lw=0.6, label="gradient norm")
    ax2 = ax.twinx()
    ax2.plot(it, [h["T"] for h in tr], color=AGENT, lw=1.4, label="rollout length T")
    ax2.set_ylim(0, max(h["T"] for h in tr) * 1.15)
    ax.set_title("Gradient norm and curriculum", fontsize=10)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3, which="both")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_benchmark_overview(path=None, finals=None, title=None):
    """Every benchmark size N x every shape family: one example scene each. Returns the figure.

    Green = goal slots (filled squares = anchors: stroke ends, corners, junctions).
    Blue = final positions: the per-scene optimum by default, or
    finals[(n, split)] = [S, N, 2] from any method (e.g. the trained policy).
    """
    from swarm.dataset import BENCH_N, HELDOUT_FAMILIES, TRAIN_FAMILIES, load_cell

    fams = TRAIN_FAMILIES + HELDOUT_FAMILIES
    fig, axs = plt.subplots(len(BENCH_N), len(fams), figsize=(1.3 * len(fams) + 1.0, 1.3 * len(BENCH_N)), dpi=100)
    for r, n in enumerate(BENCH_N):
        cells = {sp: load_cell(n, sp) for sp in ("train", "heldout")}
        for c, fam in enumerate(fams):
            ax = axs[r, c]
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.85")
            split = "train" if fam in TRAIN_FAMILIES else "heldout"
            cell = cells[split]
            if r == 0:
                name = {"random_polygon": "rand. polygon", "random_polyline": "rand. polyline",
                        "random_string": "rand. string", "heldout_string": "held-out str."}.get(fam, fam)
                ax.set_title(name + ("*" if split == "heldout" else ""), fontsize=8)
            if fam in cell["skipped"]:
                ax.text(0, 0, "too few\ndrones", ha="center", va="center", fontsize=7, color="0.55")
                continue
            i = cell["family"].index(fam)
            Q, a = cell["goals"][i], cell["anchor"][i]
            P = (finals[(n, split)] if finals else cell["oracle_final"])[i]
            ax.scatter(Q[~a, 0], Q[~a, 1], s=4, facecolors="none", edgecolors=TARGET, lw=0.5)
            ax.scatter(Q[a, 0], Q[a, 1], s=5, marker="s", color=TARGET, lw=0)
            ax.scatter(P[:, 0], P[:, 1], s=3 if n >= 48 else 6, color=AGENT, zorder=3)
        axs[r, 0].set_ylabel(f"N={n}", rotation=0, ha="right", va="center", fontsize=8, labelpad=10)
    fig.suptitle(title or "Benchmark: one example per size and family. Green = goal slots (squares = anchors), "
                 "blue = per-scene optimum. * = held out", fontsize=10, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_scores_by_n(rows_by_label: dict, path=None, title="", oracle_rows=None, colors=None):
    """Slot metrics against swarm size. rows_by_label: {label: evaluate_benchmark rows}. Returns the figure.

    Panels: slots filled, slot error, steps to form (if the rows have it). Dashed
    black = per-scene optimum (from oracle_rows, if given).
    """
    keys = [("filled", "slots filled (share of goals)", False), ("slot_error", "slot error (drone spacings)", True)]
    if all("steps_to_form" in rows[0] for rows in rows_by_label.values()):
        keys.append(("steps_to_form", "steps to form (median)", False))
    fig, axs = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.4), dpi=100)
    for ax, (key, label, log) in zip(np.atleast_1d(axs), keys):
        for j, (name, rows) in enumerate(rows_by_label.items()):
            col = (colors or {}).get(name, plt.cm.tab10(j))
            ls = "--" if "held" in name else "-"
            ax.plot([r["n"] for r in rows], [r[key] for r in rows], "o" + ls, ms=4, lw=1.5, color=col, label=name)
        if oracle_rows is not None and f"oracle_{key}" in oracle_rows[0]:
            ax.plot([r["n"] for r in oracle_rows], [r[f"oracle_{key}"] for r in oracle_rows], "k:", lw=1.2,
                    label="per-scene optimum")
        if key == "slot_error":
            ax.axhline(0.5, color=WARN, lw=0.8, ls=":")
        if key == "filled":
            ax.set_ylim(0, 1.02)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("drones N (= goals)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_loss(history, path, title: str = "", floor: float | None = None) -> Path:
    """Phase 1: loss per iteration on a log scale, with an optional best-possible line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
    ax.semilogy(history, color=AGENT, lw=1.6)
    if floor is not None:
        ax.axhline(floor, color=TARGET, ls="--", lw=1, label=f"best possible ≈ {floor:.1e}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("Chamfer loss")
    ax.grid(alpha=0.3, which="both")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
