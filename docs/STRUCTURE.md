# Project Structure

How the repo is organized. Files marked *(planned)* don't exist yet; the phase in brackets says when they are added (see [plan.txt](plan.txt)).

```
project-drone-swarm/
├── README.md               # project summary and how to run
├── pyproject.toml          # dependencies (managed with uv)
├── uv.lock                 # locked dependency versions
├── main.py                 # uv placeholder; safe to delete
├── configs/                # YAML settings for runs
├── swarm/                  # the Python package: all core code
│   └── baselines/          # comparison methods
├── scripts/                # entry points you run from the command line
├── tests/                  # pytest tests
├── demo/                   # browser demo (static site)
│   ├── js/
│   └── models/             # exported ONNX policy
├── results/                # outputs kept in git
│   ├── figures/
│   └── gifs/
├── runs/                   # checkpoints and logs (git-ignored)
├── report/
│   ├── progress/           # Nov 17 progress report
│   └── final/              # final report
└── docs/                   # planning and background documents
```

## Folders

### `configs/`

One YAML file per setup, so every run is reproducible from its config.

- `default.yaml` *(planned, Phase 1)*: `dim`, `n_agents`, `dt`, `steps`, `damping`, `a_max`, `seed`. Later phases add policy, loss-weight and training settings.

### `swarm/`

The importable package. Everything here is written in PyTorch so the whole rollout is differentiable. State tensors have shape `[B, N, dim]` (batch, agents, dimension).

| File | Purpose | Phase |
| --- | --- | --- |
| `sim.py` | Point-mass dynamics step and rollout | 1 |
| `shapes.py` | Target point clouds: circle, square, line, polyline | 1 |
| `losses.py` | Chamfer distance; later collision and effort penalties | 1, 2 |
| `viz.py` | Plots and GIFs | 1 |
| `observations.py` | Local observations: nearest neighbors and target points | 2 |
| `policy.py` | Shared MLP policy | 2 |
| `train.py` | Training loop | 2 |
| `metrics.py` | Evaluation metrics shared by all methods | 4 |

All of these are *(planned)*.

### `swarm/baselines/`

Methods we compare against, all using the same simulator and metrics.

| File | Method | Phase |
| --- | --- | --- |
| `hungarian_pd.py` | Hungarian assignment + PD control (centralized) | 3 |
| `potential_field.py` | Boids / potential field (hand-designed) | 3 |
| `ppo.py` | PPO environment wrapper and training (RL) | 3 |

### `scripts/`

Thin command-line entry points that load a config and call into `swarm/`. Run them from the repo root, e.g. `uv run python -m scripts.toy_optimize`.

| Script | Purpose | Phase |
| --- | --- | --- |
| `toy_optimize.py` | Optimize accelerations directly to check gradients flow | 1 |
| `train.py` | Train the shared policy | 2 |
| `evaluate.py` | Run experiments for all methods | 4 |
| `export_onnx.py` | Export the trained policy to `demo/models/` | 5 |

### `tests/`

pytest tests for the simulator, shapes, losses and gradients (e.g. `gradcheck` on `sim.step`). Run with `uv run pytest`.

### `demo/`

Static browser demo, hosted on GitHub Pages: `index.html`, JS simulator and Canvas drawing in `js/`, and the exported policy in `models/`. *(Phase 5)*

### `results/` vs `runs/`

- `results/` holds final figures, GIFs and tables that go into the reports. Kept in git.
- `runs/` holds checkpoints and training logs. It is in `.gitignore`.

### `report/`

LaTeX sources (or Overleaf exports) for the progress report and the final report.

### `docs/`

| File | Contents |
| --- | --- |
| [plan.txt](plan.txt) | High-level phases and steps |
| [STRUCTURE.md](STRUCTURE.md) | This file |
| [Decentralized Swarm Shape Formation via Differentiable Simulation.md](Decentralized%20Swarm%20Shape%20Formation%20via%20Differentiable%20Simulation.md) | Main project doc: claim, approach, baselines, timeline, literature |
| [Project Plan Phase 1.md](Project%20Plan%20Phase%201.md) | Detailed Phase 1 concepts and implementation steps |
| [Lecture Drone Swarms and Differentiable Programming.md](Lecture%20Drone%20Swarms%20and%20Differentiable%20Programming.md) | 25-minute course lecture plan |
