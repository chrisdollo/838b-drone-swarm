"""Train the shared policy from the command line (the notebook does the same with live plots).

    uv run python -m scripts.train --name first_run
    uv run python -m scripts.train --name quick --iters 300 --device mps
"""

import argparse

from swarm.config import load_config
from swarm.train import train


def main():
    ap = argparse.ArgumentParser(description="Train the shared swarm policy.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default="run", help="outputs go to runs/<name>/")
    ap.add_argument("--iters", type=int, help="override train.iters")
    ap.add_argument("--device", help="override device (cpu, mps, cuda)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.iters:
        cfg.train.iters = args.iters
    if args.device:
        cfg.device = args.device

    def log(history):
        tr, val = history["train"][-1], history["val"][-1]
        print(f"it {tr['it']:5d}  T {tr['T']:3d}  loss {tr['loss']:.5f}  shape {tr['shape']:.5f}  "
              f"coll {tr['collision']:.2e}  |g| {tr['grad_norm']:.3f}  "
              f"val chamfer {val['chamfer']:.5f}  {tr['time']:.0f}s", flush=True)

    train(cfg, run_dir=f"runs/{args.name}", callback=log, callback_every=50)
    print(f"saved runs/{args.name}/best.pt, last.pt, history.json")


if __name__ == "__main__":
    main()
