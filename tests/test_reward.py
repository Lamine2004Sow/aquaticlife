import numpy as np

from aquaticlife.envs import SwimEnv, SwimEnvConfig
from aquaticlife.physics.body import MorphologyParameters
from aquaticlife.physics.fluid import FluidModel


def make_morpho():
    return MorphologyParameters(
        num_segments=3,
        lengths=np.array([0.1, 0.1, 0.1]),
        masses=np.array([0.2, 0.2, 0.2]),
        joint_stiffness=np.array([5.0, 5.0]),
        joint_damping=np.array([0.1, 0.1]),
        muscle_strength=np.array([2.0, 2.0]),
    )


def test_energy_penalty_reduces_reward():
    env = SwimEnv(make_morpho(), FluidModel(drag_coef=0.5), SwimEnvConfig(dt=0.02))
    env.reset(seed=0)
    _, r0, _, info0 = env.step(np.zeros(env.action_dim))
    _, r1, _, info1 = env.step(np.ones(env.action_dim))
    assert r1 < r0
    assert info1["energy"] > info0["energy"]
