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
from aquaticlife.physics.fluid import FluidModel
from aquaticlife.utils import log_metrics


def make_fitness_fn(fluid: FluidModel) -> callable:
    def fitness(morpho, idx: int) -> float:
        env = SwimEnv(morpho, fluid, SwimEnvConfig())
        ctrl = RandomController(action_dim=morpho.num_segments - 1)
        obs = env.reset(seed=idx)
        total = 0.0
        done = False
        steps = 0
        while not done:
            action = ctrl(obs)
            obs, reward, done, info = env.step(action)
            total += reward
            steps += 1
        return total / steps

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
