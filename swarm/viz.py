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


def plot_final_grid(finals, Qs, titles=None, cols: int = 4, path=None, panel: float = 2.6):
    """Static grid of final positions vs targets. Returns the figure."""
    S = len(finals)
    cols = min(cols, S)
    rows = -(-S // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(panel * cols, panel * rows), dpi=100, squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i >= S:
            ax.axis("off")
            continue
        _setup(ax, Qs[i], titles[i] if titles else "")
        P = _np(finals[i])[..., :2]
        ax.scatter(P[:, 0], P[:, 1], s=14, color=AGENT, zorder=3)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_training(history, path=None):
    """Training curves: loss terms, validation metrics, gradient norm. Returns the figure."""
    tr, val = history["train"], history["val"]
    it = [h["it"] for h in tr]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), dpi=100)

    ax = axs[0]
    ax.semilogy(it, [h["shape"] for h in tr], color=AGENT, lw=1, label="shape (train)")
    if val:
        ax.semilogy([v["it"] for v in val], [v["chamfer"] for v in val], "o-", color=TARGET, ms=3,
                    lw=1.4, label="Chamfer (validation)")
    ax.set_title("Shape error", fontsize=10)
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axs[1]
    ax.semilogy(it, [max(h["collision"], 1e-9) for h in tr], color=WARN, lw=1, label="collision penalty")
    ax.semilogy(it, [h["effort"] for h in tr], color="0.4", lw=1, label="effort ||a||²")
    ax.set_title("Penalties (unweighted)", fontsize=10)
    ax.set_xlabel("iteration")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axs[2]
    ax.semilogy(it, [h["grad_norm"] for h in tr], color="0.3", lw=0.8, label="gradient norm")
    ax2 = ax.twinx()
    ax2.plot(it, [h["T"] for h in tr], color=AGENT, lw=1.4, label="rollout length T")
    ax2.set_ylim(0, max(h["T"] for h in tr) * 1.15)
    ax.set_title("Gradient norm and curriculum", fontsize=10)
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.3, which="both")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, frameon=False, fontsize=8, loc="upper right")

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
