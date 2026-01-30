#!/usr/bin/env python
"""
Visualisation temps réel avec pygame.

Usage:
    python scripts/viewer.py          # contrôleur aléatoire
"""

from aquaticlife.control import RandomController
from aquaticlife.envs import SwimEnv, SwimEnvConfig
from aquaticlife.physics.body import MorphologyParameters
from aquaticlife.physics.fluid import FluidModel
from aquaticlife.visualization import PygameViewer


def default_morpho() -> MorphologyParameters:
    return MorphologyParameters(
        num_segments=4,
        lengths=[0.12, 0.1, 0.08, 0.08],
        masses=[0.2, 0.18, 0.15, 0.12],
        joint_stiffness=[8.0, 6.0, 4.0],
        joint_damping=[0.1, 0.1, 0.1],
        muscle_strength=[2.0, 2.0, 2.0],
    )


def main():
    env = SwimEnv(default_morpho(), FluidModel(current_amp=0.3), SwimEnvConfig(episode_duration=10.0, dt=0.02))
    controller = RandomController(action_dim=env.action_dim)
    viewer = PygameViewer(env, controller, window=(1000, 700), scale=700.0)
    viewer.run(episodes=2, fps=60)


if __name__ == "__main__":
    main()
