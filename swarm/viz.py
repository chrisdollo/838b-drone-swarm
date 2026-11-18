"""Plots and GIFs. Nothing here needs gradients, so tensors are detached first."""

from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AGENT = "#2F6DB5"
TARGET = "#13867A"


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def save_gif(traj, Q, path, every: int = 2, fps: int = 20, hold_s: float = 1.0, title: str = "") -> Path:
    """Animate one rollout.

    traj: [T+1, N, dim] positions for one batch element; Q: [M, dim] targets.
    every: keep every k-th frame. hold_s: repeat the last frame so the final
    shape stays on screen before the GIF loops.
    """
    traj, Q = _np(traj)[..., :2], _np(Q)[..., :2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ls="--", lw=0.8, color="0.6"))
    ax.scatter(Q[:, 0], Q[:, 1], s=14, facecolors="none", edgecolors=TARGET, lw=0.8)
    agents = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=22, color=AGENT, zorder=3)
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, color="0.3")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    frames = []
    idx = list(range(0, len(traj), every))
    if idx[-1] != len(traj) - 1:
        idx.append(len(traj) - 1)
    for t in idx:
        agents.set_offsets(traj[t])
        label.set_text(f"step {t}/{len(traj) - 1}")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    frames += [frames[-1]] * int(hold_s * fps)
    iio.imwrite(path, np.stack(frames), duration=1000 / fps, loop=0)   # duration in ms
    return path


def plot_loss(history, path, title: str = "", floor: float | None = None) -> Path:
    """Loss per iteration on a log scale. floor: optional reference line (e.g. the best possible loss)."""
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
