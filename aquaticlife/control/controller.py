from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch non requis pour lire le repo
    torch = None
    nn = None


class Controller:
    """Interface minimale pour un contrôleur moteur."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class RandomController(Controller):
    """Contrôleur aléatoire utile pour des tests fumée."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)


@dataclass
class MLPConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: Callable[..., object] = np.tanh


class MLPController(Controller):
    """
    MLP léger en NumPy (pour rester indépendant de torch pendant le prototypage).

    Note : pour l'entraînement RL différentiable, remplacer par une implémentation
    PyTorch/JAX avec le même contrat d'interface.
    """

    def __init__(self, cfg: MLPConfig):
        self.cfg = cfg
        self.weights = []
        self.biases = []
        layer_sizes = (cfg.obs_dim,) + cfg.hidden_sizes + (cfg.action_dim,)
        rng = np.random.default_rng()
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = rng.standard_normal((n_in, n_out)).astype(np.float32) * (1 / np.sqrt(n_in))
            b = np.zeros(n_out, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.cfg.activation(x @ w + b)
        x = np.tanh(x @ self.weights[-1] + self.biases[-1])
        return x.astype(np.float32)
