#!/usr/bin/env python
"""
Boucle GA minimale pour tester la co-évolution de la morphologie.

Usage (rapide) :
    python scripts/evolve.py

Ce script est volontairement léger : la fitness utilise un contrôleur aléatoire.
Remplacer `make_fitness_fn` par une version PPO/NE pour des expériences sérieuses.
"""

from pathlib import Path

import numpy as np

from aquaticlife.control import RandomController
from aquaticlife.envs import SwimEnv, SwimEnvConfig
from aquaticlife.evolution import GAConfig, GeneticAlgorithm
from aquaticlife.experiments import rollout_episode
from aquaticlife.physics.fluid import FluidModel
from aquaticlife.utils import log_metrics


def make_fitness_fn(fluid: FluidModel) -> callable:
    def fitness(morpho, idx: int) -> float:
        env = SwimEnv(morpho, fluid, SwimEnvConfig())
        ctrl = RandomController(action_dim=morpho.num_segments - 1)
        stats = rollout_episode(env, ctrl, seed=idx)
        return stats.reward / stats.steps

    return fitness


def main():
    out = Path("runs/evolve_logs.jsonl")
    ga = GeneticAlgorithm(GAConfig(population_size=12, elite_fraction=0.25), make_fitness_fn(FluidModel()))
    pop = ga.init_population()
    for gen in range(5):
        pop, fitness = ga.evolve_one_generation(pop)
        log_metrics(gen, {"fitness_mean": float(fitness.mean()), "fitness_max": float(fitness.max())}, path=out)
        print(f"Generation {gen:02d} | best fitness {fitness.max():.3f}")


if __name__ == "__main__":
    main()
