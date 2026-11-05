# Decentralized Swarm Shape Formation via Differentiable Simulation

CMSC 838B Final Project · Oct 11, 2026 · @Chris

## Overview

Train one small neural controller, shared by every agent, that makes a 2D swarm self-organize into any target shape using only local information. Training backpropagates through a differentiable simulator instead of using reinforcement learning or hand-tuned rules.

**Main claim:** differentiable-simulation training matches or beats RL on shape quality while training much faster, and the learned policy stays decentralized, scales to more agents, self-heals after losses, and generalizes to unseen shapes.

**Scope:** simulation plus a live browser demo. No real-drone hardware.

## Success criteria

The minimum bar is criteria 1 and 2 plus a basic demo (criterion 4). Criterion 3 is what makes the project paper-worthy.

1. **It trains.** The shared decentralized policy, trained through the differentiable sim, reliably forms target shapes with stable training and low Chamfer error.
2. **It beats or matches the baselines.** Shape quality is comparable to PPO or better, and training is clearly faster than PPO.
3. **It shows decentralized properties** (at least two):
   - Scaling: trained on N agents, works on 2N
   - Self-healing: re-forms after agents are removed
   - Generalization: forms shapes never seen in training
4. **The demo works.** The trained policy runs live in the browser with shape selection, an agent-count slider, and click-to-remove.

## Approach

Each agent runs the same small network on what it can sense locally; the loss is computed on the final swarm, and gradients flow back through every simulation step.

- **Simulator:** 2D point-mass agents in PyTorch, written so every step is differentiable.
- **Policy:** one shared MLP. Inputs are local only: nearby agents' relative positions and nearby samples of the target shape. Output is an acceleration.
- **Loss:** assignment-free Chamfer distance between agents and target points, plus collision and control-effort penalties.
- **Training:** roll out the swarm, compute the loss, backpropagate through the rollout. Main risk is exploding or noisy gradients over long rollouts; mitigate with truncated backprop and gradient clipping.

## Baselines and experiments

Three baselines cover the main alternatives: central assignment, hand-designed rules, and RL.

| Baseline | Type | Why it is included |
| --- | --- | --- |
| Hungarian + PD | Centralized assignment | Upper bound on shape quality |
| Potential field / Boids | Hand-designed, decentralized | Classic local-rule swarm |
| PPO | Reinforcement learning | Tests the main claim: speed and quality vs RL |

| Experiment | Measures |
| --- | --- |
| Shape error | Chamfer distance on training shapes |
| Scaling | Error when agent count changes from N to 2N |
| Robustness | Recovery after removing agents |
| Generalization | Error on unseen shapes |
| Training efficiency | Wall-clock time and samples to reach a target error, vs PPO |

## Timeline and milestones

&#91;embedded content: project timeline · 5 phases, Oct 15 to Dec 1\]

Work runs Oct 15 to Dec 1, about 102 hours total, or roughly 15 hours a week. The Nov 17 progress report and Dec 17 course deadline stay fixed; Dec 1 leaves two weeks of buffer. Phase 1 is broken down step by step in *Project Plan: Phase 1*.

| Phase | Dates | Component | Hours |
| --- | --- | --- | --- |
| 1. Setup and foundations | Oct 15 – Oct 21 | Read the 4 essential papers | 4 |
|  |  | Learn the key concepts | 4 |
|  |  | Repo setup, basic simulator, Chamfer loss, toy optimization, tests | 4 |
|  |  | **Phase total** | **12** |
| 2. Core sim and training | Oct 22 – Nov 9 | Differentiable point-mass simulator | 8 |
|  |  | Target-shape loader | 3 |
|  |  | Loss: Chamfer, collision, effort | 5 |
|  |  | Shared MLP policy + training loop | 10 |
|  |  | Debug gradient stability, tune | 9 |
|  |  | **Phase total** | **35** |
| 3. Baselines + progress report | Nov 10 – Nov 17 | Hungarian + PD baseline | 4 |
|  |  | Boids / potential-field baseline | 4 |
|  |  | PPO baseline setup | 6 |
|  |  | Progress report | 4 |
|  |  | **Phase total** | **18** |
| 4. Experiments | Nov 18 – Nov 24 | Finish and tune PPO | 6 |
|  |  | Shape error + scaling | 5 |
|  |  | Robustness + generalization | 5 |
|  |  | Training-efficiency comparison | 4 |
|  |  | **Phase total** | **20** |
| 5. Demo + write-up | Nov 25 – Dec 1 | Export policy (ONNX or JS port) | 3 |
|  |  | Browser demo | 7 |
|  |  | Final write-up | 7 |
|  |  | **Phase total** | **17** |
| **All phases** | Oct 15 – Dec 1 |  | **102** |

The Nov 17 report needs Phase 2 done plus at least one baseline running.

## Tools and software

Everything runs in Python and PyTorch except the browser demo.

| Purpose | Tool |
| --- | --- |
| Simulator + training | Python, PyTorch |
| Shape loss (Chamfer / optimal transport) | PyTorch (hand-written), or geomloss |
| Hungarian baseline | SciPy linear\_sum\_assignment |
| PPO baseline | Stable-Baselines3 or CleanRL |
| Logging + plots | Weights & Biases or TensorBoard, Matplotlib |
| Browser demo | JavaScript + HTML Canvas; ONNX export + onnxruntime-web, or hand-port MLP weights to JS |
| Hosting | GitHub + GitHub Pages or Netlify |
| Report | LaTeX / Overleaf |

## Concepts to know

Listed in the order the project needs them; the first two are covered in the 838B lectures. The Phase 1 plan explains the foundational ones in depth.

| Concept | What it is | Where it's used |
| --- | --- | --- |
| Gradients, chain rule, backprop | How a loss's sensitivity flows back to every parameter | Training the policy |
| Automatic differentiation (reverse mode, PyTorch autograd) | Computing gradients of any program automatically | The whole pipeline |
| PyTorch tensors, broadcasting, batching | Swarm state as `[B, N, dim]` tensors | The simulator |
| Point-mass dynamics + Euler integration | Each agent is a point: velocity += accel·dt, position += velocity·dt | The simulator |
| Differentiable simulation / backprop through time | Unroll the sim for T steps and backprop through all of them | Core training method |
| Exploding gradients, truncated backprop, gradient clipping | Why long rollouts break training, and the standard fixes | Main technical risk |
| MLPs and parameter sharing | One small network copied onto every agent | The policy |
| Decentralized control, local observations | Each agent acts only on what it senses nearby | Policy inputs, scaling |
| Permutation invariance (sum/mean pooling) | Output doesn't depend on neighbor order or count | Handling variable neighbors |
| Point clouds + Chamfer distance | Shape as a set of points; distance between two sets | The shape loss |
| Optimal transport (Sinkhorn), optional | Moving one distribution onto another at minimum cost | Alternative shape loss |
| Soft penalties | Adding collision and effort costs to the loss | Collision avoidance |
| Hungarian algorithm + PD control | Optimal agent-to-target assignment, then steer to it | Baseline 1 |
| Boids / potential fields | Hand-made attract and repel rules | Baseline 2 |
| RL basics + PPO | Learning from rewards by trial and error | Baseline 3 |
| Experiment design | Metrics, scaling, robustness and generalization tests, fair comparisons | Experiments |
| ONNX export, JS Canvas | Running a trained model in the browser | The demo |

## Literature review

Minimum reading set: the four Essential papers below.

| Paper | Category | Priority | Role in the project |
| --- | --- | --- | --- |
| [Zhang et al., Back to Newton's Laws (2024)](https://arxiv.org/abs/2407.10648v1) | Differentiable agents | Essential | Technical template for the method |
| [DiffeoMorph (2025)](https://arxiv.org/pdf/2512.17129) | Differentiable agents | Essential | Closest competitor |
| [Batra et al., Decentralized Control of Quadrotor Swarms with End-to-end Deep RL (CoRL 2021)](https://arxiv.org/pdf/2109.07735) | Drone swarms | Essential | RL reference; shapes the PPO baseline |
| [Sun et al., Mean-shift exploration in shape assembly of robot swarms (Nature Comms 2023)](https://eprints.whiterose.ac.uk/id/eprint/200472/) | Drone swarms | Essential | Strongest non-learned method |
| Rubenstein et al., Programmable self-assembly in a thousand-robot swarm (Science 2014) | Drone swarms | Additional | Motivation and history |
| Xu et al., SHAC (ICLR 2022); Suh et al., Do Differentiable Simulators Give Better Policy Gradients? (ICML 2022) | Differentiable agents | Additional | Main technical risk: gradient stability |
| [Bandyopadhyay et al., Probabilistic swarm guidance using optimal transport (2014)](https://experts.illinois.edu/en/publications/probabilistic-swarm-guidance-using-optimal-transport/) | Other | Additional | Theory for the assignment-free loss |
| [CrazyFlow, differentiable drone simulator in JAX (2026)](https://www.dynsyslab.org/wp-content/papercite-data/pdf/schuck-arxiv26.pdf) | Other | Additional | Path to realistic dynamics; future work |

### Paper notes

**Back to Newton's Laws (Zhang et al., 2024)** — Essential

- **About:** Trains a drone controller by backpropagating through a point-mass simulator. Works on real drones; multiple drones coordinate without communicating.
- **Helps:** Almost the same method, applied to obstacle navigation instead of shapes. Use it as the technical template.
- **Important:** Proves simple differentiable sims can train real decentralized swarm behavior.

**DiffeoMorph (2025)** — Essential

- **About:** Learns to guide agents into 3D shapes with a differentiable agent simulation, inspired by how cells form organs.
- **Helps:** Shows a working shape-matching loss and the results reviewers expect.
- **Important:** Closest competitor. The write-up must state what this project adds: drone framing, RL comparison, scaling, self-healing and generalization tests.

**Batra et al., Decentralized Quadrotor Swarms with Deep RL (2021)** — Essential

- **About:** Decentralized drone-swarm controllers trained with large-scale multi-agent RL and transferred to real quadrotors.
- **Helps:** Shows how a strong RL swarm setup is built, so the PPO baseline is fair.
- **Important:** The standard RL reference; the "faster than RL" claim is measured against this kind of work.

**Sun et al., Mean-Shift Shape Assembly (Nature Comms, 2023)** — Essential

- **About:** Hand-designed rule: each robot moves toward the densest empty spots in the target shape. Tested on 50 real robots; rebuilds after damage.
- **Helps:** Strongest non-learned method for this exact task; comparison point for self-healing.
- **Important:** Shows what is possible without learning; the learned policy must match or beat it.

**Rubenstein et al., Kilobots (Science, 2014)** — Additional

- **About:** 1,000 tiny robots assembling shapes with simple local rules.
- **Helps:** Motivation and history citation for the introduction.
- **Important:** The landmark paper that made swarm shape formation famous.

**SHAC (Xu et al., 2022) and Suh et al. (2022)** — Additional

- **About:** Why gradients through long simulations explode or mislead, and fixes (shorter rollouts, smoothing).
- **Helps:** Guide for the main technical risk, unstable training.
- **Important:** Explains when differentiable sims beat RL and when they don't; useful for framing results honestly.

**Bandyopadhyay et al., Swarm Guidance via Optimal Transport (2014)** — Additional

- **About:** Treats the swarm as a mass distribution and moves it efficiently toward a target distribution.
- **Helps:** Theory behind assignment-free losses like Chamfer and optimal transport.
- **Important:** Justifies not assigning each agent to a specific spot.

**CrazyFlow (2026)** — Additional

- **About:** Fast differentiable drone simulator in JAX with realistic quadrotor physics.
- **Helps:** Option for real drone dynamics instead of point masses.
- **Important:** Bridge from the sim to real hardware; a good future-work line.

## Application framing: firefighting

A fire front or containment line can be one of the target shapes, which gives the project a real application without changing the method.

| Paper | Priority | Key idea |
| --- | --- | --- |
| [Pan et al., Using Drone Swarm to Stop Wildfire: A Predict-then-optimize Approach (2024)](https://arxiv.org/abs/2411.16144v1) | Essential | Neural fire-spread prediction from real data, then swarm task planning with mixed-integer and dynamic programming |
| [Ausonio et al., Drone Swarms in Fire Suppression Activities (2021)](https://arxiv.org/pdf/2007.00883) | Essential | Hundreds of UAVs deliver a continuous flow of extinguishing liquid on the fire front; fire modeled with cellular automata |
| [Innocente & Grasso, Swarms of autonomous drones self-organised to fight the spread of wildfires](https://pureportal.coventry.ac.uk/en/publications/swarms-of-autonomous-drones-self-organised-to-fight-the-spread-of/) | Additional | Wildfire spread model coupled with self-organizing drones using a particle swarm algorithm |
| [Optimized Area Coverage in Disaster Response Utilizing Autonomous UAV Swarm Formations (2025)](https://arxiv.org/pdf/2512.08028) | Additional | Formation-keeping UAV swarm to support first responders in wildfires |

### Paper notes

**Pan et al., Predict-then-Optimize Wildfire Swarm (2024)** — Essential

- **Helps:** Application story: target shapes as containment lines.
- **Important:** The most ML-oriented firefighting swarm paper.

**Ausonio et al., Drone Swarms in Fire Suppression (2021)** — Essential

- **Helps:** Shows why swarm shape matters: it must match the fire front.
- **Important:** The most-cited firefighting swarm paper.

**Innocente & Grasso; UAV Swarm Formations for Disaster Response (2025)** — Additional

- **Helps:** Extra citations for the application section.
- **Important:** Show the field values decentralized, robust swarms, which is what this project builds.

**Open question:** confirm with Prof. Lin which firefighting papers she meant.
