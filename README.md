# Decentralized Swarm Shape Formation via Differentiable Simulation

CMSC 838B (Differentiable Programming) final project, UMD.

We train one small neural policy, shared by every drone, that makes a 2D swarm self-organize into a target shape using only local information. Instead of reinforcement learning or hand-tuned rules, training backpropagates a shape loss through a differentiable simulator.

**Main claim:** training through the differentiable simulator matches or beats RL on shape quality and trains much faster. The learned policy stays decentralized, scales to more drones, re-forms after drones are lost, and works on shapes it never saw in training.

## How it works

- **Simulator:** 2D point-mass drones in PyTorch; every step is differentiable.
- **Policy:** one shared MLP. Each drone sees only its nearest drones, its nearest goal points and whether those goals are taken, all relative to itself, and outputs an acceleration.
- **Loss:** Chamfer distance between drones and the target shape (no drone is assigned a fixed spot), plus collision and effort penalties.
- **Training:** roll out the swarm, score the formation, backpropagate through every step.
- **Data:** a grid of 8–64 drones × 8–128 goals, shapes that pass a feature-size check, and a fixed benchmark scored against each scene's optimum.
- **Baselines (Phase 3):** Hungarian assignment + PD control, potential fields, and PPO.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # create .venv and install dependencies (including Jupyter's kernel)
uv run pytest    # run the tests
```

Each phase is one notebook that runs end to end and can be re-run on its own. Open it in VS Code (kernel: the project's `.venv`) or Jupyter:

| Notebook | What it does | Time |
| --- | --- | --- |
| [notebooks/phase1.ipynb](notebooks/phase1.ipynb) | Simulator checks, Chamfer loss, toy optimization, learning-rate lesson | ~1 min |
| [notebooks/phase2.ipynb](notebooks/phase2.ipynb) | Benchmark data, training the shared policy, results on every cell, GIFs | ~25 min |

The walkthroughs explain the math behind each phase: [docs/walkthrough/phase1.html](docs/walkthrough/phase1.html) and [docs/walkthrough/phase2.html](docs/walkthrough/phase2.html) (open in a browser).

## Phase 1: gradients flow through the simulator

Before adding any neural network, we optimize the 100 × 32 accelerations of one scene directly by backpropagating the Chamfer loss through the whole rollout. The drones form the circle, which shows that the simulator, the loss and autograd work together.

<p align="center">
  <img src="results/gifs/phase1_circle.gif" alt="32 drones moving from random positions onto a circle" width="320">
  <img src="results/figures/phase1_loss.png" alt="Chamfer loss falling from 0.033 to 6.5e-4 over 300 iterations" width="420">
</p>

- **A constant learning rate diverges.** Adam moves every parameter by about the learning rate however small its gradient is, so near the optimum the loss climbs back up. We decay the learning rate with a cosine schedule.
- **The best loss is not zero.** With 64 goals and 32 drones, each drone's best spot is a quarter of the way between two goals: L = (3/8)s².
- **A fixed plan only works for the scene it was made for:** from new starts it scores 0.16, worse than not moving. That motivates Phase 2.

## Phase 2: one shared policy

Every drone runs the same 6,850-weight network on what it sees: its 6 nearest drones, its 8 nearest goals and whether each is already taken, the scene's collision distance and its own velocity. It is trained on a grid of 8–64 drones × 8–128 goals (16 minutes on a laptop CPU) and scored on a fixed benchmark against each scene's optimum, i.e. the Phase 1 method applied to that exact scene.

<p align="center">
  <img src="results/gifs/phase2_heldout.gif" alt="The policy forming a star, heart, spiral, LOVE and MARRY ME, none of which were in training" width="760">
</p>

| Result | |
| --- | --- |
| Benchmark, training shapes | **2.23×** the per-scene optimum (untrained: 109×) |
| Benchmark, held-out shapes (star, heart, spiral, words) | **2.25×**: no penalty for unseen shapes |
| New starts: Phase 1 fixed plan vs policy | 0.167 vs **0.00049** Chamfer |
| Self-healing (8 of 32 drones removed) | repairs 64–87% of the damage |

It is strongest with two or more goals per drone (1.3–2.4×) and weakest with one goal per drone (3–6.5×), and dense 64-drone swarms still crowd. Details: [docs/walkthrough/phase2.html](docs/walkthrough/phase2.html).

## Repo layout

| Path | Contents |
| --- | --- |
| `swarm/` | Core package: simulator, shapes, data, losses, policy, training, plots |
| `swarm/baselines/` | Phase 3 baselines (to come) |
| `configs/default.yaml` | All settings |
| `notebooks/` | One runnable notebook per phase |
| `data/benchmark/` | Fixed test scenes with their per-scene optimum |
| `results/` | Figures, GIFs and the trained policy (`results/models/`) |
| `tests/` | pytest tests |
| `docs/` | Walkthroughs, architecture diagrams, plans |
| `runs/` | Training checkpoints and logs (git-ignored) |

See [docs/STRUCTURE.md](docs/STRUCTURE.md) for details.

## Plan

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Foundations, simulator, toy optimization | Done |
| 2 | Shared policy and training | Done |
| 3 | Baselines + progress report | Next |
| 4 | Experiments | |
| 5 | Browser demo + final write-up | |

The phases and their steps are in [docs/plan.txt](docs/plan.txt). The full project doc is [docs/Decentralized Swarm Shape Formation via Differentiable Simulation.md](docs/Decentralized%20Swarm%20Shape%20Formation%20via%20Differentiable%20Simulation.md).
