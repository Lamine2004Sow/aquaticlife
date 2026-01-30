#!/usr/bin/env python
"""
Squelette de boucle RL (PPO/SAC à brancher).

Pour l'instant, exécute des rollouts aléatoires et loggue les métriques,
servant de gabarit pour intégrer un vrai trainer RL.
"""

from pathlib import Path

from aquaticlife.control import RandomController
from aquaticlife.envs import SwimEnv, SwimEnvConfig
from aquaticlife.experiments import rollout_episode
from aquaticlife.physics.body import MorphologyParameters
from aquaticlife.physics.fluid import FluidModel
from aquaticlife.utils import log_metrics


def main():
    out = Path("runs/rl_logs.jsonl")
    morpho = MorphologyParameters(
        num_segments=4,
        lengths=[0.12, 0.1, 0.08, 0.08],
        masses=[0.2, 0.18, 0.15, 0.12],
        joint_stiffness=[8.0, 6.0, 4.0],
        joint_damping=[0.1, 0.1, 0.1],
        muscle_strength=[2.0, 2.0, 2.0],
    )
    env = SwimEnv(morpho, FluidModel(), SwimEnvConfig())
    ctrl = RandomController(action_dim=morpho.num_segments - 1)

    for episode in range(5):
        stats = rollout_episode(env, ctrl, seed=episode)
        log_metrics(
            episode,
            {
                "reward": stats.reward,
                "distance": stats.distance,
                "energy": stats.energy,
                "instability": stats.instability,
            },
            path=out,
        )
        print(f"Episode {episode}: reward={stats.reward:.3f}, distance={stats.distance:.3f}")

    print("TODO: remplacer par un entraîneur PPO (cf. aquaticlife.experiments.ppo_training_placeholder).")


if __name__ == "__main__":
    main()
