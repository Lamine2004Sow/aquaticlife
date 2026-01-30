#!/usr/bin/env python
"""Entraînement PPO léger pour un organisme fixe."""

from pathlib import Path

from aquaticlife.envs import SwimEnv, SwimEnvConfig
from aquaticlife.physics.body import MorphologyParameters
from aquaticlife.physics.fluid import FluidModel
from aquaticlife.rl import PPOAgent, PPOConfig
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
    env = SwimEnv(morpho, FluidModel(current_amp=0.2), SwimEnvConfig(episode_duration=6.0, dt=0.02))
    obs_dim = env.reset().shape[0]
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=env.action_dim,
        cfg=PPOConfig(steps_per_epoch=1024, epochs=5, mini_batch=128, device="cpu"),
    )

    history = agent.train(env)
    for h in history:
        log_metrics(h["epoch"], {"pi_loss": h.get("pi_loss", 0.0), "vf_loss": h.get("vf_loss", 0.0), "kl": h.get("kl", 0.0)}, path=out)
        print(f"Epoch {h['epoch']:02d} | pi_loss={h.get('pi_loss', 0):.4f} vf_loss={h.get('vf_loss', 0):.4f} kl={h.get('kl', 0):.4f}")


if __name__ == "__main__":
    main()
