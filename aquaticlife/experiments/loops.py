from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from aquaticlife.control import Controller
from aquaticlife.envs import SwimEnv
from aquaticlife.physics.body import MorphologyParameters
from aquaticlife.utils import log_metrics


@dataclass
class EpisodeStats:
    reward: float
    distance: float
    energy: float
    instability: float
    steps: int


def rollout_episode(env: SwimEnv, controller: Controller, seed: int | None = None) -> EpisodeStats:
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    total_energy = 0.0
    distance = 0.0
    instability = 0.0
    steps = 0
    while not done:
        action = controller(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_energy += info["energy"]
        distance = info["distance"]
        instability += info["instability"]
        steps += 1
    return EpisodeStats(
        reward=total_reward,
        distance=distance,
        energy=total_energy,
        instability=instability / max(1, steps),
        steps=steps,
    )


def evaluate_population(
    population: List[MorphologyParameters],
    controller_fn: Callable[[MorphologyParameters], Controller],
    env_fn: Callable[[MorphologyParameters], SwimEnv],
    log_path,
) -> np.ndarray:
    """
    Évalue une population de morphologies avec un contrôleur fourni.
    Retourne un vecteur fitness et loggue les métriques.
    """
    fitness = np.zeros(len(population), dtype=np.float32)
    for i, morpho in enumerate(population):
        env = env_fn(morpho)
        ctrl = controller_fn(morpho)
        stats = rollout_episode(env, ctrl, seed=i)
        fitness[i] = stats.reward / stats.steps
        log_metrics(
            i,
            {
                "fitness": float(fitness[i]),
                "distance": stats.distance,
                "energy": stats.energy,
                "instability": stats.instability,
            },
            path=log_path,
        )
    return fitness


def ppo_training_placeholder(*args, **kwargs):
    """
    Point d'entrée pour brancher PPO/SAC ultérieurement.
    Cette fonction sert de contrat : signature à garder stable.
    """
    raise NotImplementedError("Brancher un entraîneur PPO ici (PyTorch/JAX).")
