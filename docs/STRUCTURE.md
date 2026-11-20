# Project Structure

How the repo is organized. Each phase runs from one notebook; all logic lives in the `swarm/` package so notebooks, tests and later phases share the same code.

```
project-drone-swarm/
├── README.md
├── pyproject.toml, uv.lock   # dependencies (managed with uv)
├── configs/default.yaml      # all settings
├── notebooks/                # one runnable notebook per phase
│   ├── phase1.ipynb
│   └── phase2.ipynb
├── swarm/                    # the Python package: all core code
│   └── baselines/            # Phase 3 comparison methods (to come)
├── data/benchmark/           # fixed test scenes + per-scene optimum
├── results/                  # outputs kept in git
│   ├── figures/
│   ├── gifs/
│   └── models/               # trained policy (phase2_policy.pt)
├── tests/                    # pytest tests
├── docs/                     # walkthroughs, diagrams, plans
└── runs/                     # checkpoints and logs (git-ignored)
```

## `swarm/`

Everything is written in PyTorch so the whole rollout is differentiable. State tensors have shape `[B, N, dim]` (batch of scenes, drones, dimension).

| File | Purpose | Phase |
| --- | --- | --- |
| `config.py` | Load `configs/default.yaml` with attribute access (`cfg.train.lr`) | 1 |
| `sim.py` | Point-mass step; open-loop rollout (Phase 1); closed-loop rollout and removal test (Phase 2) | 1, 2 |
| `shapes.py` | Basic target shapes and even sampling along polylines | 1 |
| `losses.py` | Chamfer distance; shape, collision and effort terms, normalized by drone spacing | 1, 2 |
| `plan.py` | Optimize a fixed acceleration plan for known scenes (toy optimization and per-scene optimum) | 1, 2 |
| `dataset.py` | Drones × goals grid, shape families, single-stroke font, feature-size rule, training batches, benchmark | 2 |
| `observations.py` | What each drone sees: nearest drones, nearest goals and whether they are taken, d_min, own velocity | 2 |
| `policy.py` | The shared MLP; save and load | 2 |
| `train.py` | Training loop and benchmark scoring (ratio to the per-scene optimum) | 2 |
| `viz.py` | GIFs and figures | 1, 2 |

Controllers share one signature, `controller(pos, vel, Q, d_min) -> acc`, so the Phase 3 baselines run in the same simulator and are scored on the same benchmark.

## `notebooks/`

| Notebook | Runs | Writes |
| --- | --- | --- |
| `phase1.ipynb` | Simulator gradient check, shapes, Chamfer example, toy optimization, learning-rate comparison, fixed plan on new starts | `results/gifs/phase1_circle.gif`, `results/figures/phase1_*.png` |
| `phase2.ipynb` | Benchmark (build or load), observation demo, training, scoring every cell, GIFs | `runs/phase2/`, `results/models/phase2_policy.pt`, `results/figures/phase2_*.png`, `results/gifs/phase2_*.gif` |

## `data/benchmark/`

One file per grid cell and split, `n<N>_m<M>_<train|heldout>.pt`, plus `index.json`. Each holds the scenes (start positions, goals, d_min, shape family) and their per-scene optimum. Rebuilt by setting `REBUILD_BENCHMARK = True` in the Phase 2 notebook.

## `docs/`

| File | Contents |
| --- | --- |
| `walkthrough/phase1.html`, `walkthrough/phase2.html` | The math of each phase, with results (open in a browser); `walkthrough/media/` holds their images |
| `phases_explained.txt` | Plain-text notes: what each phase does and what every file is for |
| `architecture/` | Architecture diagrams per phase; `make_figures.py` regenerates them |
| `plan.txt` | High-level phases and steps |
| `Decentralized Swarm Shape Formation via Differentiable Simulation.md` | Main project doc: claim, approach, baselines, literature |
| `Project Plan Phase 1.md` | The original Phase 1 plan and concepts |
