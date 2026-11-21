"""Load a YAML config into an object with attribute access (cfg.dt, cfg.train.lr, ...)."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x


def to_dict(x):
    """Inverse of load_config: namespace -> plain dict (for saving with a checkpoint)."""
    if isinstance(x, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(x).items()}
    if isinstance(x, list):
        return [to_dict(v) for v in x]
    return x


def load_config(path: str | Path = "configs/default.yaml") -> SimpleNamespace:
    with open(path) as f:
        return _to_ns(yaml.safe_load(f))


def set_overrides(cfg, overrides: dict):
    """Change settings by dotted name, e.g. {"loss.w_spacing": 1.0}. Used for named experiments."""
    for key, value in overrides.items():
        *path, last = key.split(".")
        node = cfg
        for part in path:
            node = getattr(node, part)
        if not hasattr(node, last):
            raise KeyError(f"unknown setting: {key}")
        setattr(node, last, value)
    return cfg
