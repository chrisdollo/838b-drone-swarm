# Decentralized Swarm Shape Formation via Differentiable Simulation

CMSC 838B (Differentiable Programming) final project, UMD.

We train one small neural policy, shared by every agent, that makes a 2D swarm self-organize into a target shape using only local information. Instead of reinforcement learning or hand-tuned rules, training backpropagates a shape loss through a differentiable simulator.

**Main claim:** training through the differentiable simulator matches or beats RL on shape quality and trains much faster. The learned policy stays decentralized, scales to more agents, re-forms after agents are lost, and works on shapes it never saw in training.

## How it works

- **Simulator:** 2D point-mass agents in PyTorch; every step is differentiable.
- **Policy:** one shared MLP. Each agent sees only its nearest neighbors and nearby target points, as offsets relative to itself, and outputs an acceleration.
- **Loss:** Chamfer distance between agents and the target shape (no agent is assigned a fixed spot), plus collision and control-effort penalties.
- **Training:** roll out the swarm, compute the loss on the final state, backpropagate through every step.
- **Baselines:** Hungarian assignment + PD control, Boids / potential fields, and PPO.
- **Demo:** the trained policy runs live in the browser (ONNX + HTML Canvas).

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # create .venv and install dependencies
uv run pytest    # run the tests
```

Scripts are run from the repo root, for example:

```bash
uv run python -m scripts.toy_optimize
```

## Repo layout

| Path | Contents |
| --- | --- |
| `swarm/` | Core package: simulator, shapes, losses, policy, training |
| `swarm/baselines/` | Hungarian + PD, potential field, PPO |
| `configs/` | YAML run settings |
| `scripts/` | Command-line entry points |
| `tests/` | pytest tests |
| `demo/` | Browser demo |
| `results/` | Figures and GIFs for the reports |
| `report/` | Progress and final reports |
| `docs/` | Plans and background |

See [docs/STRUCTURE.md](docs/STRUCTURE.md) for details.

## Plan

| Phase | Dates | Focus |
| --- | --- | --- |
| 1 | Oct 15 – Oct 21 | Foundations, simulator, toy optimization |
| 2 | Oct 22 – Nov 9 | Shared policy and training |
| 3 | Nov 10 – Nov 17 | Baselines + progress report |
| 4 | Nov 18 – Nov 24 | Experiments |
| 5 | Nov 25 – Dec 1 | Browser demo + final write-up |

The phases and their steps are in [docs/plan.txt](docs/plan.txt). The full project doc is [docs/Decentralized Swarm Shape Formation via Differentiable Simulation.md](docs/Decentralized%20Swarm%20Shape%20Formation%20via%20Differentiable%20Simulation.md).
