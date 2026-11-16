"""Load a YAML config into an object with attribute access (cfg.dt, cfg.steps, ...)."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> SimpleNamespace:
    with open(path) as f:
        return SimpleNamespace(**yaml.safe_load(f))
