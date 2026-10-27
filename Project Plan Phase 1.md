# Project Plan: Phase 1

Decentralized Swarm Shape Formation via Differentiable Simulation · Oct 11, 2026 · @Chris

Phase 1 runs Oct 15 to Oct 21, 2026: learn the foundations, read the key papers, and build a working differentiable simulator that a toy optimization can steer into a shape.

## Goal and checkpoint

Phase 1 is done when you can explain every design choice below and a toy optimization moves 32 agents into a circle by backpropagating through your own simulator.

| Part | Hours |
| --- | --- |
| Read the 4 essential papers | 4 |
| Learn the concepts | 4 |
| Setup, simulator, loss, toy optimization, tests | 4 |
| **Total** | **12** |

**Deliverables:**

- [ ] One page of notes per paper (what to extract is listed below)
- [ ] Repo on GitHub with the simulator, shapes, Chamfer loss and tests
- [ ] A GIF of agents forming a circle from the toy optimization

Phase 2 then adds the neural policy, local observations, and collision and effort penalties on top of this code.

## Papers

Read in this order: the first gives you the RL vocabulary, the second the method you will copy, the last two the task and the competition.

| Order | Paper | Time | Focus on | Extract into your notes |
| --- | --- | --- | --- | --- |
| 1 | [Batra et al., Decentralized Control of Quadrotor Swarms with End-to-end Deep RL (2021)](https://arxiv.org/pdf/2109.07735) | 1 h | Problem setup, observation design, network architecture | What each drone observes, why the policy is shared, how neighbors are encoded, how long training takes |
| 2 | [Zhang et al., Back to Newton's Laws (2024)](https://arxiv.org/abs/2407.10648v1) | 1.5 h | Method: point-mass model, loss, training through the sim; multi-agent section | Their dynamics equations, loss terms, rollout length, tricks for stable gradients |
| 3 | [Sun et al., Mean-shift Shape Assembly (2023)](https://eprints.whiterose.ac.uk/id/eprint/200472/) | 0.5 h | Task definition, metrics, results figures | How they define a shape, how they measure success, what self-healing looks like |
| 4 | [DiffeoMorph (2025)](https://arxiv.org/pdf/2512.17129) | 1 h | Shape loss, agent model, experiments | Their loss vs Chamfer, what they test, what they do not test (RL comparison, scaling) |

Skip proofs and appendices on first read. If time is left, skim Suh et al., Do Differentiable Simulators Give Better Policy Gradients? (ICML 2022), which backs concepts 8 and 9 below.

## How it fits

&#91;embedded content: training loop · forward pass and gradient path\]

The forward pass runs the loop for T steps; the loss is computed once at the end, and its gradient travels back through every step to the policy's weights.

## Concepts

Ten concepts, in dependency order. Each has an intuition, a formal version for your notes, and the project choice it explains.

### 1. Agent, state, observation, action, policy

**Intuition:** an agent is one drone. Its state is everything true about it (position, velocity). Its observation is the part it can actually sense. Its action is what it decides to do. The policy is the brain that turns observations into actions, like a driver who looks around and then steers.

**Formal:** agent i has state s\_i = (p\_i, v\_i), observation o\_i, and action a\_i. A policy with weights θ is a function:

```latex
a_i = \pi_\theta(o_i)
```

A rollout (or episode) is one full run of the simulation for T steps.

**Project choice:** the action is an acceleration (a 2D vector), not a target position. Acceleration is what a real drone's motors control, and it keeps motion smooth.

### 2. Centralized vs decentralized control, shared policy

**Intuition:** centralized means one boss sees everything and tells each drone where to go. Decentralized means each drone decides on its own from what it sees. A shared policy means every drone has an identical copy of the same brain, like ants all following the same instincts.

**Formal:** centralized: (a\_1, …, a\_N) = F(s\_1, …, s\_N). Decentralized with sharing: a\_i = π\_θ(o\_i) for every i, with one θ.

**Project choice:** decentralized and shared because it (1) works for any N without retraining, (2) survives losing agents, since no one is in charge, and (3) needs one small network, which is easy to run in a browser. These three properties are exactly what the experiments test.

### 3. Local observations

**Intuition:** each drone only sees its closest neighbors and the nearby part of the target shape, and it sees them relative to itself ("a neighbor 2 m to my left"), not as world coordinates.

**Formal:** for agent i, take its k nearest neighbors and m nearest target points and stack their offsets:

```latex
o_i = [\, p_{j_1} - p_i, \dots, p_{j_k} - p_i,\; q_{l_1} - p_i, \dots, q_{l_m} - p_i,\; v_i \,]
```

Sorting by distance gives a fixed-size input regardless of N.

**Project choice:** relative offsets make the policy work anywhere in space (translation invariance). A fixed k makes the input size independent of swarm size, which is what lets one policy scale from N to 2N. Using offsets also means the code works unchanged in 3D.

### 4. Point-mass dynamics and Euler integration

**Intuition:** treat each drone as a dot with mass. Each time step, acceleration changes velocity and velocity changes position, like updating a ball's motion frame by frame in a game.

**Formal:** with time step Δt and damping c (air drag):

```latex
v_{t+1} = (1 - c\,\Delta t)\, v_t + a_t\, \Delta t, \qquad p_{t+1} = p_t + v_{t+1}\, \Delta t
```

Actions are bounded with a\_t = a\_max · tanh(network output).

**Project choice:** point masses are simple, smooth and fully differentiable, and Back to Newton's Laws shows they are enough to train real drones. Damping keeps agents from drifting forever. tanh caps acceleration like real motor limits while staying differentiable.

### 5. The rollout as one big computation graph

**Intuition:** running the simulation for T steps is just a long chain of math operations. Autograd records every one, so the final loss is connected to the policy weights through every step, like a very deep network where each layer is one time step.

**Formal:** p\_T depends on θ through every action. Backpropagation applies the chain rule backward through time:

```latex
\frac{\partial L}{\partial \theta} = \sum_{t=0}^{T-1} \frac{\partial L}{\partial a_t}\, \frac{\partial a_t}{\partial \theta}
```

where each ∂L/∂a\_t itself flows back through all later steps. This is backpropagation through time (BPTT).

**Project choice:** this is the core idea of the project and the link to your autograd lectures. Every simulator line must use PyTorch operations, with no NumPy or .item() inside the loop, or the chain breaks.

### 6. Loss vs reward

**Intuition:** RL uses a reward, a score the world hands back with no explanation of how to improve. Here we use a loss, a smooth formula of the agents' positions, so the math itself tells each agent which way to move.

**Formal:** a reward r is treated as a black box. A loss L(p\_1, …, p\_N) is differentiable, so ∇L exists.

**Project choice:** the same quantity (distance to shape) could be a reward or a loss; making it a differentiable loss is what allows gradients through the sim instead of trial and error.

### 7. Chamfer distance (assignment-free shape loss)

**Intuition:** two checks. Every agent should be near some target point (no stragglers), and every target point should be near some agent (no gaps). Nobody is assigned a specific spot; agents just need to cover the shape.

**Formal:** for agent positions P and target points Q:

```latex
L_{CD}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \lVert p - q \rVert^2 + \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \lVert p - q \rVert^2
```

**Project choice:** an assignment (agent 7 goes to spot 7) needs a central planner and breaks when agents are lost. Chamfer has no assignment, so it fits decentralized control, works for any N, and self-healing falls out naturally. Both terms matter: the first alone lets all agents pile onto one point.

### 8. Why not RL, and the gradient trade-off

**Intuition:** RL learns by trying random variations and keeping what scored better, like tuning a recipe by taste-testing many batches. Differentiable sim gets the exact direction to improve in one pass, like reading the recipe's chemistry. The catch: if the world has sudden jumps (collisions) or very long chains, exact gradients can explode or point the wrong way.

**Formal:** RL estimates the gradient from samples (zeroth-order, the log-likelihood trick):

```latex
\nabla_\theta J \approx \frac{1}{K} \sum_{k=1}^{K} R(\tau_k)\, \nabla_\theta \log \pi_\theta(\tau_k)
```

This estimate is noisy, so K must be large. Differentiable sim computes ∇\_θ L directly (first-order). PPO is a stabilized version of this RL estimator and is the baseline you will compare against later.

**Project choice:** the project's main claim is that the first-order path trains faster. Smooth point-mass dynamics and soft (smooth) collision penalties are chosen specifically to avoid the cases where Suh et al. show first-order gradients fail.

### 9. Gradient stability over long rollouts

**Intuition:** backprop multiplies many per-step Jacobians together. If each slightly grows a signal, after 100 steps it explodes; if each shrinks it, it vanishes.

**Formal:** ∂p\_T/∂p\_0 = J\_{T-1} ··· J\_1 J\_0; its norm can grow or decay exponentially in T.

**Project choice:** damping (c > 0) makes each step slightly shrinking, which tames explosions. Later you add gradient clipping (cap the gradient norm) and truncated BPTT (backprop only through the last chunk of steps), and you can start with short rollouts and lengthen them (curriculum).

### 10. Optimizer and batching

**Intuition:** each training step, run several simulations at once from random start positions, average the loss, and nudge the weights. Adam is gradient descent that adapts its step size per weight.

**Formal:** with batch size B, the loss is (1/B) Σ\_b L\_b and the update is θ ← Adam(θ, ∇\_θ L).

**Project choice:** random starts and random shapes per batch force the policy to learn a general skill rather than memorize one trajectory, which is what generalization to unseen shapes needs. State tensors carry a batch dimension: \[B, N, dim\].

## Implementation steps

Nine steps, about 4 hours. The end result is a toy optimization that proves gradients flow through your simulator before any neural network is added.

### Step 1. Environment (20 min)

1. Create a Python 3.11 environment (conda or venv).
2. Install: `pip install torch numpy scipy matplotlib pyyaml pytest imageio`. Add `wandb` if you want online logging.
3. CPU is enough for Phase 1 (32 agents, 100 steps). Check `torch.cuda.is_available()` only for later phases.

### Step 2. Repo structure (15 min)

```
swarm-diffsim/
  configs/default.yaml   # all settings in one place
  swarm/
    sim.py               # dynamics step + rollout
    shapes.py            # target point clouds
    losses.py            # Chamfer (later: collision, effort)
    viz.py               # plots and GIFs
  scripts/toy_optimize.py
  tests/
  README.md
```

Create the GitHub repo and make a first commit.

### Step 3. Config (10 min)

Start with these values in `configs/default.yaml`:

```
dim: 2          # set to 3 later for 3D
n_agents: 32
dt: 0.05        # seconds per step
steps: 100      # rollout length T (5 s of motion)
damping: 0.5    # c in the dynamics
a_max: 2.0      # max acceleration
seed: 0
```

Why: dim is a setting from day one so 3D is a one-line change. dt × steps = 5 s gives agents time to cross the unit box.

### Step 4. Simulator step and rollout (45 min)

In `swarm/sim.py`, positions and velocities are tensors of shape `[B, N, dim]` (batch, agents, dimension).

```python
def step(pos, vel, acc, dt, damping):
    vel = (1 - damping * dt) * vel + acc * dt
    pos = pos + vel * dt
    return pos, vel

def rollout(pos0, vel0, accs, cfg):
    pos, vel = pos0, vel0
    traj = [pos]
    for t in range(cfg.steps):
        pos, vel = step(pos, vel, accs[:, t], cfg.dt, cfg.damping)
        traj.append(pos)
    return torch.stack(traj, dim=1)  # [B, T+1, N, dim]
```

Rules: no in-place ops (`+=` on tensors that need gradients), no `.numpy()` or `.item()` inside the loop. Either breaks the gradient chain.

In Phase 2, `accs[:, t]` is replaced by the policy's output at step t.

### Step 5. Target shapes (30 min)

In `swarm/shapes.py`, write `sample_shape(name, m, dim)` that returns `[m, dim]` points, centered and scaled to fit a box of side 1.

- **circle:** angles evenly spaced, points (cos, sin) × 0.4
- **square:** points evenly spaced along the 4 edges
- **line:** points evenly spaced on a segment
- **polyline:** any list of vertices, sampled evenly by arc length (this later covers letters and user-drawn shapes)

Use m = 2 × N target points so the shape is densely described.

### Step 6. Chamfer loss (20 min)

In `swarm/losses.py`:

```python
def chamfer(P, Q):            # P: [B, N, dim], Q: [B, M, dim]
    d = torch.cdist(P, Q) ** 2 # [B, N, M] squared distances
    return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()
```

The first term pulls every agent onto the shape; the second fills every part of it.

### Step 7. Toy optimization: the key Phase 1 test (60 min)

In `scripts/toy_optimize.py`, skip the neural network and directly optimize the acceleration sequence:

1. Random start positions in the box, zero velocity, B = 1.
2. Parameters: `raw = torch.zeros(1, steps, N, dim, requires_grad=True)`; accelerations `a_max * torch.tanh(raw)`.
3. Loop 300 iterations: rollout, loss = chamfer(final positions, circle), `loss.backward()`, Adam step (lr 0.05), zero gradients.
4. Log the loss every 10 iterations. It should drop steadily toward near zero.
5. Save a GIF of the final trajectory with `viz.py` (matplotlib scatter per step, written with imageio).

Why this test: if agents form the circle, the simulator, loss and autograd are all correct, and any Phase 2 bug must be in the policy or observations.

### Step 8. Tests (30 min)

In `tests/`, with pytest:

- `chamfer(P, P) == 0`, and chamfer is symmetric
- `sample_shape` returns the right count and stays inside the box
- with zero acceleration and zero damping, velocity stays constant and position moves linearly
- `torch.autograd.gradcheck` on `step` in float64 (compares autograd to finite differences)

### Step 9. Wrap up (10 min)

- [ ] Commit, push, and add a README with how to run the toy script and the GIF
- [ ] Write down one surprise or issue (for example, how the loss changed when you varied damping or steps); these become talking points for the Nov 17 report
