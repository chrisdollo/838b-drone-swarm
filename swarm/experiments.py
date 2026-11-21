"""Named experiment settings, shared by the notebooks and command-line runs.

Each experiment is a set of overrides of configs/default.yaml (swarm.config.set_overrides).

Loss study (geometric shapes only, one slot per drone, 8-96 drones):
the same training setup with five formation losses (swarm/formation_losses.py).
Weights were chosen with the per-scene optimizer so each loss reaches its
formation (notebooks/phase2_losses.ipynb, section 2): energy needs w_shape 10
because it is in spacings, not spacings^2.
"""

LOSS_STUDY = {
    "chamfer":   {"loss.shape": "chamfer"},
    "matching":  {"loss.shape": "matching"},
    "sinkhorn":  {"loss.shape": "sinkhorn"},
    "energy":    {"loss.shape": "energy", "loss.w_shape": 10.0},
    "laplacian": {"loss.shape": "laplacian"},
}
# Stabilized energy runs: the first loss study diverged once rollouts reached 100
# steps (gradient norms 1e5-1e13). Truncated BPTT (cut the gradient every 25
# steps) + Huber-like energy (smooth within 0.2 spacings of a slot;
# 0.5 cost accuracy: the per-scene optimizer filled 77% instead of 95% at N=64).
LOSS_STUDY_STABLE = {
    "energy_stable":     {"loss.shape": "energy", "loss.w_shape": 10.0, "loss.energy_delta": 0.2, "train.tbptt": 25},
    "energy_stable_all": {"loss.shape": "energy", "loss.w_shape": 10.0, "loss.energy_delta": 0.2, "train.tbptt": 25,
                          "loss.hold_frac": 1.0},          # variant: shape scored at every step
}
LOSS_TAG = "phase2_loss_{}"          # runs/<tag>/, results/models/<tag>.pt

# Stabilizer screen: every loss x every stabilizer, short runs (1,000 iterations).
# The curriculum is compressed so ~70% of each run uses full 100-step rollouts,
# the regime where the first study diverged. No early stopping, so all runs
# are compared at the same budget. Energy always uses the Huber-like form.
SCREEN_COMMON = {"train.iters": 1000, "train.curriculum": [[0, 40], [150, 70], [300, 100]],
                 "train.val_every": 200, "train.patience": 0}
SCREEN_LOSSES = {
    "chamfer":   {"loss.shape": "chamfer"},
    "matching":  {"loss.shape": "matching"},
    "sinkhorn":  {"loss.shape": "sinkhorn"},
    "energy":    {"loss.shape": "energy", "loss.w_shape": 10.0, "loss.energy_delta": 0.2},
    "laplacian": {"loss.shape": "laplacian"},
}
SCREEN_STABILIZERS = {
    "none":  {},                              # gradient clipping only (always on)
    "tbptt": {"train.tbptt": 25},             # cut the gradient every 25 steps (SHAC-style short horizon)
    "decay": {"train.grad_decay": 0.92},      # temporal gradient decay (Zhang et al. 2024)
}
STABILIZER_SCREEN = {f"{l}_{s}": SCREEN_COMMON | lo | so
                     for l, lo in SCREEN_LOSSES.items() for s, so in SCREEN_STABILIZERS.items()}
SCREEN_TAG = "screen_{}"             # runs/screen_<loss>_<stabilizer>/

# Full grid (Colab runner, swarm/grid.py): loss x form x time weighting x stabilizer,
# 3,000 iterations each, no early stopping (best checkpoint by slots filled is kept).
GRID_COMMON = {"train.iters": 3000, "train.patience": 0, "train.val_every": 250}
GRID_LOSSES = {
    "chamfer":   {"loss.shape": "chamfer"},
    "matching":  {"loss.shape": "matching"},
    "sinkhorn":  {"loss.shape": "sinkhorn"},
    "energy":    {"loss.shape": "energy", "loss.w_shape": 10.0},
    "laplacian": {"loss.shape": "laplacian"},
}
GRID_FORMS = {
    "plain":  {},
    "robust": {"loss.huber": 1.0, "loss.energy_delta": 0.2},   # Huber for squared losses; smooth |x| for energy
}
GRID_TIMES = {t: {"loss.time_weighting": t} for t in ["final", "last", "every", "terminal", "discount"]}
GRID_STABILIZERS = {
    "tbptt": {"train.tbptt": 25},
    "decay": {"train.grad_decay": 0.92},
}
GRID = {f"{l}_{f}_{t}_{s}": GRID_COMMON | lo | fo | to | so
        for l, lo in GRID_LOSSES.items() for f, fo in GRID_FORMS.items()
        for t, to in GRID_TIMES.items() for s, so in GRID_STABILIZERS.items()}
