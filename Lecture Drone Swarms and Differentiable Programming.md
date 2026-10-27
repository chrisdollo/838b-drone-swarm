# Lecture: Drone Swarms and Differentiable Programming

CMSC 838B · Oct 11, 2026 · @Chris

A 25-minute lecture on how swarms are controlled, why RL is slow, and how training through a differentiable simulator offers a faster path.

## Run of show

Five sections over 25 minutes, building from local rules to RL to differentiable simulation.

| # | Section | Paper | Minutes |
| --- | --- | --- | --- |
| 1 | Why swarms, and local rules | Boids | 3 |
| 2 | Learning swarms with RL | Batra et al. | 6 |
| 3 | Gradients through simulators | Suh et al. | 7 |
| 4 | Differentiable swarms in practice | Back to Newton's Laws | 7 |
| 5 | Wrap-up and open problems | DiffeoMorph | 2 |
|  | **Total** |  | **25** |

## Papers

Four core papers carry the lecture; the optional ones are swap-ins or a closing slide.

| Paper | Role | Priority |
| --- | --- | --- |
| Reynolds, Flocks, Herds, and Schools: A Distributed Behavioral Model (Boids, SIGGRAPH 1987) | Intro: local rules create swarm behavior | Core |
| [Batra et al., Decentralized Control of Quadrotor Swarms with End-to-end Deep RL (CoRL 2021)](https://arxiv.org/pdf/2109.07735) | How swarms are learned today, and why RL is costly | Core |
| Suh et al., Do Differentiable Simulators Give Better Policy Gradients? (ICML 2022) | When gradients through a simulator help or fail | Core |
| [Zhang et al., Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics (2024)](https://arxiv.org/abs/2407.10648v1) | A real drone swarm trained through a differentiable sim | Core |
| Xu et al., SHAC: Accelerated Policy Learning with Parallel Differentiable Simulation (ICLR 2022) | Concrete training algorithm; swap for Suh et al. | Optional |
| [DiffeoMorph (2025)](https://arxiv.org/pdf/2512.17129) | Shape formation; bridge to the project | Optional |

Boids, Suh et al. and SHAC are cited from memory; confirm links before the slides.

## Content

15 slides, under 2 minutes each. One idea per slide; each section ends by setting up the next.

### 1. Why swarms, and local rules (3 min)

1. **Title + hook:** a drone light show or firefighting swarm video; question: how do hundreds of drones coordinate?
2. **Boids:** three local rules (separation, alignment, cohesion) produce flocking with no leader.
   - Takeaway: decentralized, local-only control scales and survives losses.
   - Transition: hand-written rules are hard to design for complex goals, so can we learn them?

### 2. Learning swarms with RL (6 min)

3. **RL in one slide:** agent acts, gets a reward, improves by trial and error; no gradient through the world.
4. **Batra et al. setup:** one shared policy per drone, local observations, trained with large-scale multi-agent RL in a quadrotor simulator.
5. **Results:** flocking, tight formations, collision avoidance, and zero-shot transfer to real small quadrotors.
6. **The cost:** huge numbers of samples and long training, because the reward signal is noisy.
   - Transition: the simulator is just math. What if we differentiate through it?

### 3. Gradients through simulators (7 min)

7. **Differentiable simulation:** unroll the sim, compute a loss on the final state, backpropagate through every step (backprop through time).
8. **Two gradient types:** first-order (exact gradients through the sim) vs zeroth-order (RL-style, estimated from samples).
9. **Suh et al. findings:** first-order gradients are faster and lower-variance when dynamics are smooth; they can be biased or explode with contacts, discontinuities, or long chaotic rollouts.
10. **Fixes:** shorter or truncated rollouts, smoothing, gradient clipping, mixing both estimators (SHAC is one example).
    - Transition: does this actually work for real drones?

### 4. Differentiable swarms in practice (7 min)

11. **Back to Newton's Laws method:** each drone is a simple point mass; a depth renderer gives the camera view; a recurrent policy is trained by backprop through the sim with a physics-driven loss.
12. **Results:** about 90% success vs 60% for the prior state of the art in cluttered single-drone flight; up to 20 m/s in real forests; zero-shot sim-to-real.
13. **Swarm result:** multiple drones navigate together with no communication or central planner; coordination emerges from the loss.

### 5. Wrap-up (2 min)

14. **Open problems:** shape formation (DiffeoMorph), realistic drone dynamics, scaling to hundreds of agents, fair comparisons with RL. Optional one-line pointer to your project.
15. **Questions.**

## Prep plan

About 15 hours of prep; start at least two weeks before the lecture date.

- [ ] Read Boids and Batra et al. (2 h)
- [ ] Read Suh et al., skim SHAC (3 h)
- [ ] Read Back to Newton's Laws in depth (2 h)
- [ ] Skim DiffeoMorph for the wrap-up (0.5 h)
- [ ] Collect figures and videos from the papers (1 h)
- [ ] Build slides (4 h)
- [ ] Write speaker notes (1 h)
- [ ] Two timed rehearsals, cut to 25 min (1.5 h)

## Open questions

- [ ] What is the lecture date?
- [ ] Must the lecture center on one main paper? If so, make Back to Newton's Laws the main paper and the rest background.
- [ ] Does the 25 minutes include Q&A?
- [ ] Is a short mention of the final project allowed or expected?
- [ ] Required slide format or template (Google Slides link for the course sheet)?
