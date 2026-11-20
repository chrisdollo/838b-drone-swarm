"""The shared decentralized policy: one small MLP, copied onto every agent.

    a_i = a_max * tanh( MLP_theta( observe_i ) )

The same weights theta are applied to every agent's own observation. Nothing
in the network mixes agents, so each agent could run it on board.
"""

from pathlib import Path

import torch
from torch import nn

from swarm.observations import obs_dim, observe


class SharedMLP(nn.Module):
    def __init__(self, k: int = 6, m: int = 8, dim: int = 2, hidden: int = 64, layers: int = 2,
                 a_max: float = 2.0, pos_scale: float = 10.0):
        super().__init__()
        self.hparams = dict(k=k, m=m, dim=dim, hidden=hidden, layers=layers, a_max=a_max, pos_scale=pos_scale)
        self.k, self.m, self.a_max = k, m, a_max

        body, width = [], obs_dim(k, m, dim)
        for _ in range(layers):
            body += [nn.Linear(width, hidden), nn.SiLU()]   # SiLU: smooth, so gradients through time stay smooth
            width = hidden
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(hidden, dim)
        # Start near "do nothing": tiny output weights give a ~ 0 for every agent.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

        # Distances (offsets, occupancies, d_min) are ~0.01-1 units and velocities
        # ~0-4, so distances are scaled up to give the first layer similar-sized inputs.
        n_dist = obs_dim(k, m, dim) - dim
        scale = torch.cat([torch.full((n_dist,), pos_scale), torch.ones(dim)])
        self.register_buffer("in_scale", scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [..., obs_dim] -> acceleration [..., dim], bounded by a_max."""
        return self.a_max * torch.tanh(self.head(self.body(obs * self.in_scale)))

    def act(self, pos: torch.Tensor, vel: torch.Tensor, Q: torch.Tensor, d_min: torch.Tensor) -> torch.Tensor:
        """Controller interface used by the simulator: (pos, vel, Q, d_min) -> acc [B, N, dim]."""
        return self(observe(pos, vel, Q, d_min, self.k, self.m))

    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "SharedMLP":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        policy = cls(**ckpt["hparams"])
        policy.load_state_dict(ckpt["state_dict"])
        return policy


def policy_from_config(cfg) -> SharedMLP:
    p = cfg.policy
    return SharedMLP(k=p.k_neighbors, m=p.m_targets, dim=cfg.dim, hidden=p.hidden, layers=p.layers,
                     a_max=cfg.a_max, pos_scale=p.pos_scale)
