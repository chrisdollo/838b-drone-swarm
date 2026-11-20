"""Architecture diagrams for every phase of the project.

Run from the repo root:
    uv run python docs/architecture/make_figures.py

Writes <name>.png (slides, web) and <name>.pdf (LaTeX) next to this file.
Edit the box text here when the design changes, then re-run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).parent

INK = "#1F2633"
MUTED = "#5B6577"
LINE = "#3B4252"
GRAD = "#C8402F"      # gradient (backward) arrows
DONE = "#2E8B57"      # "built" marker
NEUTRAL_FC = "#F4F6F9"
NEUTRAL_EC = "#8A94A6"
PHASE = {1: "#2F6DB5", 2: "#13867A", 3: "#B7791F", 4: "#6B4FBB", 5: "#B5476A"}
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"


def tint(hex_color: str, amount: float) -> str:
    """Mix a color with white; amount=1 gives white."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02X}" for c in mixed)


def new_fig(w, h, phase, title, subtitle):
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.add_patch(Rectangle((0, h - 0.1), w, 0.1, color=PHASE.get(phase, INK), lw=0))
    ax.text(0.4, h - 0.52, title, fontsize=18, weight="bold", color=INK, va="center", family=SANS)
    ax.text(0.4, h - 0.95, subtitle, fontsize=11.5, color=MUTED, va="center", family=SANS)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), color=None, mono=True, note=None, done=False,
        title_size=11.5, size=10.2):
    fc = tint(color, 0.88) if color else NEUTRAL_FC
    ec = color or NEUTRAL_EC
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + 0.18, y + h - 0.3, title, fontsize=title_size, weight="bold", color=INK,
            va="center", family=MONO if mono else SANS, zorder=3)
    if done:
        ax.text(x + w - 0.15, y + h - 0.3, "✓ built", fontsize=8.5, weight="bold", color=DONE,
                ha="right", va="center", family=SANS, zorder=3)
    rows = list(lines) + ([note] if note else [])
    for i, line in enumerate(rows):
        is_note = note is not None and i == len(rows) - 1
        ax.text(x + 0.18, y + h - 0.68 - 0.3 * i, line, fontsize=size - (0.6 if is_note else 0),
                color=ec if is_note else INK, weight="bold" if is_note else "normal",
                va="center", family=SANS, zorder=3)
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h),
            "b": (x + w / 2, y), "x": x, "y": y, "w": w, "h": h}


def group(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                                fc=tint(color, 0.965), ec=color, lw=1.2,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(x + 0.2, y + h - 0.27, label, fontsize=10.5, weight="bold", style="italic",
            color=color, va="center", family=SANS, zorder=3)


STYLES = {
    "fwd": dict(color=LINE, ls="-"),
    "grad": dict(color=GRAD, ls=(0, (5, 3))),
    "rl": dict(color=MUTED, ls=(0, (1.5, 2.5))),
}


def arrow(ax, pts, kind="fwd", color=None, label=None, label_at=None, lw=1.6):
    style = dict(STYLES[kind])
    if color:
        style["color"] = color
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([x0, x1], [y0, y1], lw=lw, zorder=1.5, **style))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=1.5,
                                 color=style["color"], linestyle=style["ls"]))
    if label:
        if label_at is None:
            segs = list(zip(pts[:-1], pts[1:]))
            (x0, y0), (x1, y1) = max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
            label_at = ((x0 + x1) / 2, (y0 + y1) / 2)
        ax.text(*label_at, label, fontsize=9.6, color=style["color"], ha="center", va="center",
                family=SANS, zorder=4, bbox=dict(fc="white", ec="none", pad=1.6))


def legend(ax, x, y, items, vertical=False, gap=3.6):
    texts = {"fwd": "forward pass / data", "grad": "gradient (backward pass)",
             "rl": "sampled reward (no gradient)", "update": "parameter update"}
    for i, kind in enumerate(items):
        dx, dy = (0, -0.4 * i) if vertical else (gap * i, 0)
        k = "fwd" if kind == "update" else kind
        col = PHASE[2] if kind == "update" else None
        arrow(ax, [(x + dx, y + dy), (x + dx + 0.6, y + dy)], kind=k, color=col)
        ax.text(x + dx + 0.75, y + dy, texts[kind], fontsize=9.5, color=MUTED, va="center", family=SANS)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- overview
def overview():
    fig, ax = new_fig(16, 9.4, 0, "Project architecture: all phases",
                      "Decentralized swarm shape formation via differentiable simulation · Oct 15 – Dec 1, 2026 · ~102 h")
    phases = [
        ("Phase 1", "Setup + foundations", "Oct 15–21 · 12 h", "hands off: diff. sim + loss", [
            ("sim.py", "point-mass step + rollout", True),
            ("shapes.py", "circle, square, line, polyline", True),
            ("losses.py", "Chamfer distance", True),
            ("plan.py", "optimise accelerations", True),
            ("tests/", "gradcheck, shapes, loss", True)]),
        ("Phase 2", "Policy + training", "Oct 22 – Nov 9 · 35 h", "hands off: trained policy", [
            ("observations.py", "k-NN + goal occupancy", True),
            ("policy.py", "shared MLP, tanh output", True),
            ("dataset.py", "drones × goals grid, benchmark", True),
            ("train.py", "BPTT, clipping, curriculum", True)]),
        ("Phase 3", "Baselines + report", "Nov 10–17 · 18 h", "hands off: 3 baselines", [
            ("hungarian_pd.py", "assignment + PD control", False),
            ("potential_field.py", "attract + repel rules", False),
            ("ppo.py", "RL env + PPO", False),
            ("report/progress/", "progress report, Nov 17", False)]),
        ("Phase 4", "Experiments", "Nov 18–24 · 20 h", "hands off: results", [
            ("phase4.ipynb", "all methods, same benchmark", False),
            ("metrics.py", "one set of metrics", False),
            ("5 experiments", "error, scale, heal, unseen, speed", False),
            ("results/", "figures + tables", False)]),
        ("Phase 5", "Demo + write-up", "Nov 25 – Dec 1 · 17 h", "delivers: demo + report", [
            ("phase5.ipynb", "export policy → ONNX", False),
            ("demo/", "JS sim + Canvas + controls", False),
            ("GitHub Pages", "live demo URL", False),
            ("report/final/", "LaTeX final report", False)]),
    ]
    w, step_x = 2.8, 3.1
    for i, (name, sub, dates, handoff, files) in enumerate(phases):
        x = 0.4 + i * step_x
        c = PHASE[i + 1]
        hdr = box(ax, x, 6.4, w, 1.65, name, [sub, dates], color=c, mono=False, note=handoff, title_size=13)
        if i < 4:
            arrow(ax, [hdr["r"], (hdr["r"][0] + step_x - w, hdr["r"][1])], color=c)
        for j, (fname, desc, done) in enumerate(files):
            box(ax, x, 6.05 - 0.85 - j * 0.98, w, 0.85, fname, [desc], mono=True, done=done,
                title_size=10.5, size=9.4)
    group(ax, 0.4, 0.3, 15.2, 0.75, "", INK)
    ax.text(0.65, 0.675, "Shared by every phase:  configs/*.yaml  ·  state tensors [B, N, dim]  ·  "
            "the same sim.step, shapes and metrics for every method", fontsize=10.5, color=INK,
            va="center", family=SANS, zorder=3)
    save(fig, "00_overview")


# --------------------------------------------------------------------------- phase 1
def phase1():
    c = PHASE[1]
    fig, ax = new_fig(16, 9.3, 1, "Phase 1 · Setup and foundations",
                      "Toy optimisation: optimise the acceleration sequence directly (no network) and check that "
                      "gradients flow back through the simulator")
    cfg = box(ax, 0.4, 6.4, 3.4, 1.4, "configs/default.yaml",
              ["dim 2 · N 32 · dt 0.05", "T 100 · c 0.5 · a_max 2"], done=True, title_size=10.8)
    raw = box(ax, 0.4, 3.8, 3.4, 1.6, "parameters θ = raw",
              ["shape [1, T, N, 2], zeros", "requires_grad=True"], color=c)
    act = box(ax, 4.4, 3.8, 2.6, 1.6, "bounded actions", [r"$a = a_{max}\,\tanh(\mathrm{raw})$"],
              color=c, mono=False)
    group(ax, 7.5, 3.3, 4.2, 2.75, "sim.rollout · repeat T = 100 steps", c)
    step = box(ax, 7.8, 3.55, 3.6, 1.95, "sim.step",
               [r"$v \leftarrow (1-c\,\Delta t)\,v + a\,\Delta t$", r"$p \leftarrow p + v\,\Delta t$",
                "no in-place ops, no .item()"], color=c, done=True)
    shp = box(ax, 12.2, 6.4, 3.4, 1.4, "shapes.sample_shape",
              ["'circle', M = 2N = 64 points", "centred, fits the unit box"], color=c, title_size=10.8, done=True)
    loss = box(ax, 12.2, 3.8, 3.4, 1.6, "losses.chamfer",
               [r"$L = \mathrm{CD}(p_T,\ Q)$", "agents→shape + shape→agents"], color=c, done=True)
    adam = box(ax, 0.4, 2.0, 3.4, 1.2, "torch.optim.Adam", ["lr 0.05, cosine decay · 300 iters"], color=c, done=True)
    box(ax, 7.5, 0.35, 4.2, 1.15, "viz.py", ["scatter per step → results/gifs/"], color=c, done=True)
    box(ax, 0.4, 0.35, 6.6, 1.15, "tests/",
        ["chamfer(P, P) = 0 and symmetric · shapes stay in the box",
         "constant velocity when a = c = 0 · gradcheck(step) in float64"], color=c, done=True)

    arrow(ax, [raw["r"], act["l"]])
    arrow(ax, [act["r"], (step["x"], act["r"][1])], label="a", label_at=(7.25, 4.6))
    arrow(ax, [(step["x"] + 3.6, 4.6), loss["l"]], label=r"$p_T$", label_at=(11.95, 4.6))
    arrow(ax, [shp["b"], loss["t"]], label="Q  [1, M, 2]")
    arrow(ax, [cfg["r"], (9.6, cfg["r"][1]), (9.6, 6.05)], label="dt, c, T, a_max")
    arrow(ax, [loss["b"], (loss["b"][0], 2.6), adam["r"]], kind="grad",
          label="loss.backward():  ∂L/∂raw through all T steps (BPTT)", label_at=(6.8, 2.6))
    arrow(ax, [adam["t"], raw["b"]], color=c)
    arrow(ax, [(10.9, 3.3), (10.9, 1.5)], label="trajectory [1, T+1, N, 2]", label_at=(10.9, 2.05))
    legend(ax, 12.2, 1.25, ["fwd", "grad"], vertical=True)
    save(fig, "01_phase1_toy_optimization")


# --------------------------------------------------------------------------- phase 2
def phase2():
    c = PHASE[2]
    fig, ax = new_fig(16, 10.6, 2, "Phase 2 · Policy and training",
                      "One shared MLP replaces the free accelerations; the loss on the final swarm is "
                      "backpropagated through every step into θ")
    group(ax, 4.3, 3.4, 11.3, 5.6, "one rollout · repeat T steps, policy called every step", c)
    batch = box(ax, 0.4, 7.0, 3.5, 1.7, "batch setup",
                ["cell (N, M) from the grid", "random starts p0, v0 = 0", "shapes Q, d_min per scene"], color=c, done=True)
    state = box(ax, 4.7, 4.0, 2.9, 1.5, "state at step t", [r"$p_t,\ v_t$   [B, N, 2]"], title_size=11)
    obs = box(ax, 7.9, 6.5, 3.4, 1.9, "observations.py",
              [r"k nearest neighbours: $p_j - p_i$", r"m nearest target pts: $q_l - p_i$",
               "is each goal taken? + d_min", r"own velocity $v_i$  (39 numbers)"], color=c, done=True)
    pol = box(ax, 12.0, 6.5, 3.3, 1.9, "policy.py",
              [r"shared MLP $\pi_\theta$", "one θ for every agent",
               r"$a_i = a_{max}\tanh(\mathrm{MLP}(o_i))$"], color=c, done=True)
    step = box(ax, 12.0, 3.9, 3.3, 1.6, "sim.step",
               [r"$v \leftarrow (1-c\,\Delta t)\,v + a\,\Delta t$", r"$p \leftarrow p + v\,\Delta t$"],
               color=c, done=True)
    loss = box(ax, 10.4, 1.0, 5.2, 1.8, "losses.py",
               [r"$L$ = Chamfer$(p_T, Q)$ + $w_c$·collision + $w_e$·effort",
                "collision below d_min = 0.4 x spacing, every t", "each scene ÷ spacing², mean over B"], color=c, done=True)
    stab = box(ax, 5.4, 1.0, 4.4, 1.8, "gradient stability",
               ["loss.backward() through T steps", "clip the gradient norm",
                "truncated BPTT · curriculum on T"], color=c, mono=False)
    adam = box(ax, 0.4, 1.0, 4.4, 1.8, "train.py · Adam",
               [r"$\theta \leftarrow \mathrm{Adam}(\theta, \nabla_\theta L)$", "checkpoints + logs → runs/",
                "best → results/models/"], color=c, done=True)

    arrow(ax, [batch["b"], (batch["b"][0], state["l"][1]), state["l"]], label="t = 0", label_at=(2.15, 5.6))
    arrow(ax, [(batch["x"] + 3.5, 8.1), (obs["x"], 8.1)], label="Q", label_at=(5.6, 8.1))
    arrow(ax, [state["t"], (state["t"][0], 7.2), (obs["x"], 7.2)], label="p, v", label_at=(6.9, 7.2))
    arrow(ax, [obs["r"], pol["l"]], label=r"$o_i$", label_at=(11.65, 7.45))
    arrow(ax, [pol["b"], step["t"]], label="a  [B, N, 2]")
    arrow(ax, [step["l"], (state["x"] + 2.9, step["l"][1])], label=r"$p_{t+1},\ v_{t+1}$")
    arrow(ax, [step["b"], (step["b"][0], loss["y"] + 1.8)], label="trajectory", label_at=(13.65, 3.1))
    arrow(ax, [loss["l"], stab["r"]], kind="grad")
    arrow(ax, [stab["l"], adam["r"]], kind="grad")
    arrow(ax, [adam["l"], (0.2, adam["l"][1]), (0.2, 9.2), (13.65, 9.2), pol["t"]], color=c,
          label="updated θ, shared by all agents", label_at=(8.6, 9.2))
    legend(ax, 0.4, 0.45, ["fwd", "grad", "update"])
    save(fig, "02_phase2_policy_training")


# --------------------------------------------------------------------------- phase 3
def phase3():
    c = PHASE[3]
    fig, ax = new_fig(16, 9.6, 3, "Phase 3 · Baselines and progress report",
                      "Three comparison methods, each producing accelerations for the same simulator and "
                      "scored with the same metric")
    lanes = [
        (6.2, 1.7,
         ("global state", ["all positions p_1 … p_N", "all target points Q"]),
         ("hungarian_pd.py", ["assign: scipy linear_sum_assignment",
                              r"PD: $a_i = k_p(q_{\sigma(i)} - p_i) - k_d\,v_i$"],
          "centralized · upper bound on shape quality")),
        (3.9, 1.7,
         ("local view", ["neighbours within radius r", "nearest target points"]),
         ("potential_field.py", ["attract toward nearest target points", "repel neighbours closer than r"],
          "decentralized · hand-designed rules")),
        (1.2, 2.2,
         ("observations.py", ["same o_i as our policy", "(k-NN offsets + own v)"]),
         ("ppo.py", ["gym-style env wrapping sim.step", "reward = −ΔChamfer − penalties",
                     "Stable-Baselines3 or CleanRL"],
          "decentralized · learns from samples")),
    ]
    env = box(ax, 10.1, 1.2, 2.6, 6.7, "shared sim",
              ["sim.step", "same dt, c, a_max", "same shapes", "same seeds"], mono=False)
    for y, h, (in_t, in_l), (m_t, m_l, note) in lanes:
        inp = box(ax, 0.4, y, 3.2, h, in_t, in_l, mono=in_t.endswith(".py"))
        m = box(ax, 4.2, y, 5.2, h, m_t, m_l, color=c, note=note)
        arrow(ax, [inp["r"], m["l"]])
        arrow(ax, [m["r"], (env["x"], m["r"][1])], label="a", label_at=(9.75, m["r"][1]))
    arrow(ax, [(11.4, 1.2), (11.4, 0.7), (6.8, 0.7), (6.8, 1.2)], kind="rl",
          label="reward r_t + next obs (no gradient through the sim)", label_at=(9.1, 0.7))
    met = box(ax, 13.3, 5.4, 2.3, 2.5, "metrics", ["Chamfer error", "per method,", "same shapes"],
              mono=False, color=PHASE[4])
    rep = box(ax, 13.3, 2.0, 2.3, 2.4, "report/progress/", ["due Nov 17", "needs Phase 2 +", "≥ 1 baseline"],
              color=c, title_size=10.5)
    arrow(ax, [(env["x"] + 2.6, 6.65), (met["x"], 6.65)])
    arrow(ax, [met["b"], rep["t"]])
    legend(ax, 12.9, 1.35, ["fwd", "rl"], vertical=True)
    save(fig, "03_phase3_baselines")


# --------------------------------------------------------------------------- phase 4
def phase4():
    c = PHASE[4]
    fig, ax = new_fig(16, 9.6, 4, "Phase 4 · Experiments",
                      "Every method runs through every experiment with the same shapes, seeds, N and T")
    methods = [("policy.py (ours)", "trained through diff. sim", PHASE[2]),
               ("hungarian_pd.py", "centralized", PHASE[3]),
               ("potential_field.py", "hand-designed", PHASE[3]),
               ("ppo.py", "RL baseline", PHASE[3])]
    ev = box(ax, 4.3, 2.15, 3.0, 5.75, "phase4.ipynb",
             ["same shapes", "same seeds", "same N and T", "same sim.step", "",
              "runs every method", "through every", "experiment"], color=c)
    for i, (name, desc, col) in enumerate(methods):
        m = box(ax, 0.4, 6.65 - 1.5 * i, 3.2, 1.25, name, [desc], color=col, title_size=10.8)
        arrow(ax, [m["r"], (ev["x"], m["r"][1])])
    exps = [("shape error", "Chamfer on training shapes, all methods"),
            ("scaling", "train on N agents, test on 2N"),
            ("self-healing", "remove agents mid-rollout, time to recover"),
            ("generalization", "Chamfer on shapes never seen in training"),
            ("training efficiency", "wall-clock + samples to target error vs PPO")]
    for i, (name, desc) in enumerate(exps):
        e = box(ax, 8.0, 6.9 - 1.15 * i, 4.4, 1.0, name, [desc], color=c, mono=False)
        arrow(ax, [(ev["x"] + 3.0, e["l"][1]), e["l"]])
        ax.add_line(Line2D([e["r"][0], 12.75], [e["r"][1]] * 2, color=LINE, lw=1.6, zorder=1.5))
    ax.add_line(Line2D([12.75, 12.75], [6.9 - 1.15 * 4 + 0.5, 7.4], color=LINE, lw=1.6, zorder=1.5))
    met = box(ax, 13.1, 5.2, 2.5, 2.7, "metrics.py",
              ["Chamfer error", "recovery time", "wall-clock time", "env samples"], color=c)
    res = box(ax, 13.1, 2.15, 2.5, 2.45, "results/",
              ["figures/ (plots)", "tables", "→ success", "criteria 2 + 3"], color=c)
    arrow(ax, [(12.75, 6.55), (met["x"], 6.55)])
    arrow(ax, [met["b"], res["t"]])
    box(ax, 0.4, 0.35, 15.2, 1.15, "Fairness rules",
        ["PPO gets the same observations, simulator and tuning effort · every number comes from metrics.py · "
         "fixed seeds so runs are repeatable"], mono=False)
    save(fig, "04_phase4_experiments")


# --------------------------------------------------------------------------- phase 5
def phase5():
    c = PHASE[5]
    fig, ax = new_fig(16, 10.0, 5, "Phase 5 · Browser demo and write-up",
                      "The trained policy runs live in the browser: same observations, same physics, "
                      "inference with onnxruntime-web")
    group(ax, 0.4, 4.6, 4.2, 3.7, "Python (offline, once)", PHASE[2])
    ckpt = box(ax, 0.7, 6.6, 3.6, 1.2, "runs/…/policy.pt", ["trained θ from Phase 2"], color=PHASE[2])
    exp = box(ax, 0.7, 4.9, 3.6, 1.3, "phase5.ipynb", ["torch.onnx.export(MLP)", "input o  [N, obs_dim]"],
              color=PHASE[2])
    onnx = box(ax, 0.7, 2.6, 3.6, 1.2, "demo/models/policy.onnx", ["static file served with the page"],
               color=c, title_size=10.5)
    rep = box(ax, 0.4, 0.4, 4.2, 1.5, "report/final/", ["LaTeX / Overleaf", "figures from results/"], color=c)
    arrow(ax, [ckpt["b"], exp["t"]])
    arrow(ax, [exp["b"], onnx["t"]])

    group(ax, 5.2, 1.6, 10.4, 6.7, "browser · static site on GitHub Pages · loop runs every frame", c)
    state = box(ax, 5.6, 5.9, 2.8, 1.5, "swarm state", ["p, v  [N, 2]", "target points Q"], color=c, mono=False)
    obs = box(ax, 9.0, 5.9, 3.0, 1.5, "js observations", ["k-NN offsets in JS", "same layout as Python"],
              color=c)
    inf = box(ax, 12.5, 5.9, 2.8, 1.5, "onnxruntime-web", [r"$\pi_\theta(o)$ for all N agents", "→ accelerations a"],
              color=c, title_size=10.8)
    step = box(ax, 12.5, 3.6, 2.8, 1.5, "js sim step", ["same Euler update", "→ new p, v"], color=c)
    draw = box(ax, 9.0, 3.6, 3.0, 1.5, "Canvas draw", ["agents + target shape", "requestAnimationFrame"],
               color=c, mono=False)
    ctl = box(ax, 5.6, 1.9, 3.0, 1.5, "controls", ["shape selector", "agent-count slider",
                                                  "click to remove agent"], color=c, mono=False)
    arrow(ax, [state["r"], obs["l"]])
    arrow(ax, [obs["r"], inf["l"]], label="o", label_at=(12.25, 6.65))
    arrow(ax, [inf["b"], step["t"]], label="a")
    arrow(ax, [step["l"], draw["r"]])
    arrow(ax, [draw["l"], (7.6, draw["l"][1]), (7.6, state["y"])], label="next frame", label_at=(8.3, 4.35))
    arrow(ax, [(6.1, ctl["y"] + 1.5), (6.1, state["y"])], label="edit Q, N,\nremove agents", label_at=(6.1, 4.6))
    arrow(ax, [onnx["r"], (4.9, onnx["r"][1]), (4.9, 8.65), (13.9, 8.65), inf["t"]],
          label="model loaded once at page start", label_at=(9.4, 8.65))
    legend(ax, 5.2, 0.8, ["fwd"])
    ax.text(9.0, 0.8, "Demo URL is the deliverable for success criterion 4", fontsize=9.5, color=MUTED,
            va="center", family=SANS)
    save(fig, "05_phase5_demo")


if __name__ == "__main__":
    for make in (overview, phase1, phase2, phase3, phase4, phase5):
        make()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
