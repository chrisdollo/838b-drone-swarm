# Project Structure

How the repo is organized. Each phase runs from one notebook; all logic lives in the `swarm/` package so notebooks, tests and later phases share the same code.

```
project-drone-swarm/
├── README.md
├── pyproject.toml, uv.lock   # dependencies (managed with uv)
├── configs/default.yaml      # all settings
├── notebooks/                # one runnable notebook per phase
│   ├── phase1.ipynb
│   ├── phase2.ipynb          # one named experiment per run (v3)
│   ├── phase2_compare.ipynb  # v2 vs v3 runs side by side
│   ├── v3/                   # executed copy of phase2.ipynb for the v3 baseline run
│   └── v2/                   # archived, executed v2 notebooks (read-only)
├── swarm/                    # the Python package: all core code
│   └── baselines/            # Phase 3 comparison methods (to come)
├── data/benchmark_v3/        # fixed test scenes + per-scene optimum (v3, current)
├── data/benchmark/           # the v2 benchmark (kept for reference)
├── results/                  # outputs kept in git
│   ├── figures/
│   ├── gifs/
│   └── models/               # trained policies: phase2_<exp>.pt (v2), phase2_v3_<exp>.pt (v3)
├── tests/                    # pytest tests
├── docs/                     # walkthroughs, reports, diagrams, plans
└── runs/                     # checkpoints and logs (git-ignored)
```

## `swarm/`

Everything is written in PyTorch so the whole rollout is differentiable. State tensors have shape `[B, N, dim]` (batch of scenes, drones, dimension).

| File | Purpose | Phase |
| --- | --- | --- |
| `config.py` | Load `configs/default.yaml` with attribute access (`cfg.train.lr`) | 1 |
| `sim.py` | Point-mass step; open-loop rollout (Phase 1); closed-loop rollout (Phase 2) | 1, 2 |
| `shapes.py` | Basic target shapes and even sampling along polylines | 1 |
| `metrics.py` | Slot metrics (one goal per drone): Hungarian matching, slots filled, slot error, steps to form, shared goals | 2 |
| `losses.py` | Chamfer distance; shape, collision, effort and spacing (repulsion) terms, normalized by drone spacing; evenness metric | 1, 2 |
| `plan.py` | Optimize a fixed acceleration plan for known scenes (toy optimization and per-scene optimum) | 1, 2 |
| `dataset.py` | Shape families (incl. random letter/digit strings), fixed held-out strings and seeds, feature-size rule, training batches with N = M drawn from 8-96, the v3 benchmark | 2 |
| `layout.py` | Goal layout for any strokes: an anchor at every end, corner and junction, even fill, one goal per drone; text via the Hershey single-stroke font, A-Z and 0-9 (`text_goals`, `fewest_readable`) | 2 |
| `observations.py` | What each drone sees: nearest drones, nearest goals and whether they are taken, d_min, own velocity | 2 |
| `policy.py` | The shared MLP; save and load | 2 |
| `train.py` | Training loop (model selection by slot error) and benchmark scoring | 2 |
| `viz.py` | GIFs and figures | 1, 2 |

Controllers share one signature, `controller(pos, vel, Q, d_min) -> acc`, so the Phase 3 baselines run in the same simulator and are scored on the same benchmark.

## `notebooks/`

| Notebook | Runs | Writes |
| --- | --- | --- |
| `phase1.ipynb` | Simulator gradient check, shapes, Chamfer example, toy optimization, learning-rate comparison, fixed plan on new starts, text goals with one goal per drone | `results/gifs/phase1_circle.gif`, `results/gifs/phase1_text_<string>.gif`, `results/figures/phase1_*.png` |
| `phase2.ipynb` | One named experiment (`EXPERIMENT`, or the env variable): benchmark, observation demo, training with early stopping, slot metrics per swarm size, GIFs incl. held-out strings, words and 128-drone demos | `runs/phase2_v3_<name>/`, `results/models/phase2_v3_<name>.pt`, `results/phase2_v3_<name>_scores.json`, `results/figures/phase2_v3_<name>_*.png`, `results/gifs/phase2_v3_<name>_*.gif` |
| `phase2_compare.ipynb` | v2 and v3 runs on identical v3 benchmark scenes | `results/figures/phase2_v3_compare_*.png`, `results/gifs/phase2_v3_compare_*.gif`, `results/phase2_v3_compare.json` |
| `v2/` | The executed v2 notebooks, kept as a record (they need the v2 code, commit `35826a0`) | |

Output names: v2 files are `phase2_<exp>_*` and `phase2_compare_*`; v3 files are `phase2_v3_<exp>_*` and `phase2_v3_compare_*`.

## `data/`

`benchmark_v3/` (current): one file per swarm size and split, `n<N>_<train|heldout>.pt`, plus `index.json`. N ∈ {8, 16, 24, 32, 48, 64, 96}, one goal slot per drone, 4 scenes per family. Each file holds the scenes (start positions, goals, anchor flags, d_min, family, text), the seed it was built from, and the per-scene optimum with its slot metrics. Seeds: cell i (in N order), split j (0 = train, 1 = held-out) uses `2026 + 100 i + j`; the held-out strings were drawn with seed 838. Rebuilt by setting `REBUILD_BENCHMARK = True` in the Phase 2 notebook.

`benchmark/`: the v2 benchmark (drones × goals grid), kept for reference.

## `docs/`

| File | Contents |
| --- | --- |
| `walkthrough/phase1.html`, `walkthrough/phase2.html` | The math of each phase, with results (open in a browser); `walkthrough/media/` holds their images |
| `reports/phase2_spacing.html` | Experiment report: baseline vs spacing (repulsion) loss, with side-by-side GIFs |
| `phases_explained.txt` | Plain-text notes: what each phase does and what every file is for |
| `architecture/` | Architecture diagrams per phase; `make_figures.py` regenerates them |
| `plan.txt` | High-level phases and steps |
| `Decentralized Swarm Shape Formation via Differentiable Simulation.md` | Main project doc: claim, approach, baselines, literature |
| `Project Plan Phase 1.md` | The original Phase 1 plan and concepts |
