from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from aquaticlife.physics.body import MorphologyParameters


FitnessFn = Callable[[MorphologyParameters, int], float]


@dataclass
class GAConfig:
    population_size: int = 32
    elite_fraction: float = 0.1
    mutation_scale: float = 0.05
    min_segments: int = 2
    max_segments: int = 8
    seed: int = 0


class GeneticAlgorithm:
    """
    GA minimal pour co-évolution morphologie + (optionnel) paramètres de contrôleur.
    Se concentre ici sur la morphologie; l'entraînement du contrôleur est externe.
    """

    def __init__(self, cfg: GAConfig, fitness_fn: FitnessFn):
        self.cfg = cfg
        self.fitness_fn = fitness_fn
        self.rng = np.random.default_rng(cfg.seed)

    def init_population(self) -> List[MorphologyParameters]:
        pop = []
        for _ in range(self.cfg.population_size):
            n = self.rng.integers(self.cfg.min_segments, self.cfg.max_segments + 1)
            lengths = self.rng.uniform(0.05, 0.2, size=n)
            masses = self.rng.uniform(0.05, 0.8, size=n)
            k_joint = self.rng.uniform(1.0, 20.0, size=n - 1)
            c_joint = self.rng.uniform(0.01, 1.0, size=n - 1)
            muscles = self.rng.uniform(0.1, 5.0, size=n - 1)
            pop.append(
                MorphologyParameters(
                    num_segments=n,
                    lengths=lengths,
                    masses=masses,
                    joint_stiffness=k_joint,
                    joint_damping=c_joint,
                    muscle_strength=muscles,
                ).clamp(self.cfg.min_segments, self.cfg.max_segments)
            )
        return pop

    def mutate(self, morpho: MorphologyParameters) -> MorphologyParameters:
        noise = lambda x: x + self.rng.normal(0.0, self.cfg.mutation_scale, size=x.shape)
        lengths = noise(morpho.lengths)
        masses = noise(morpho.masses)
        k_joint = noise(morpho.joint_stiffness)
        c_joint = noise(morpho.joint_damping)
        muscles = noise(morpho.muscle_strength)
        return MorphologyParameters(
            num_segments=morpho.num_segments,
            lengths=lengths,
            masses=masses,
            joint_stiffness=k_joint,
            joint_damping=c_joint,
            muscle_strength=muscles,
        ).clamp(self.cfg.min_segments, self.cfg.max_segments)

    def select_elite(self, population: List[MorphologyParameters], fitness: np.ndarray) -> List[MorphologyParameters]:
        k = max(1, int(self.cfg.elite_fraction * len(population)))
        elite_idx = np.argsort(fitness)[-k:]
        return [population[i] for i in elite_idx]

    def evolve_one_generation(self, population: List[MorphologyParameters]) -> Tuple[List[MorphologyParameters], np.ndarray]:
        fitness = np.array([self.fitness_fn(p, i) for i, p in enumerate(population)], dtype=np.float32)
        elite = self.select_elite(population, fitness)

        new_pop: List[MorphologyParameters] = []
        new_pop.extend(elite)
        while len(new_pop) < self.cfg.population_size:
            parent = elite[self.rng.integers(0, len(elite))]
            child = self.mutate(parent)
            new_pop.append(child)
        return new_pop, fitness
