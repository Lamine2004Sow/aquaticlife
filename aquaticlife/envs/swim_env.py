from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from aquaticlife.physics.body import MorphologyParameters, RigidSegmentChain
from aquaticlife.physics.fluid import FluidModel


@dataclass
class SwimEnvConfig:
    dt: float = 0.02
    episode_duration: float = 5.0  # seconds
    drag_root: float = 1.5
    energy_penalty: float = 0.02
    stability_penalty: float = 0.01


class SwimEnv:
    """
    Environnement gym-like très léger (reset/step) pour RL ou évaluation GA.
    Modèle simplifié mais cohérent : drag visqueux + propulsion par oscillation.
    """

    def __init__(self, morpho: MorphologyParameters, fluid: FluidModel, cfg: SwimEnvConfig | None = None):
        self.cfg = cfg or SwimEnvConfig()
        self.fluid = fluid
        self.body = RigidSegmentChain(morpho)
        self.time = 0.0
        self.max_steps = int(self.cfg.episode_duration / self.cfg.dt)
        self.step_count = 0
        self.root_vel = np.zeros(2, dtype=np.float32)
        self.prev_positions = self.body.forward_kinematics()
        self.action_dim = self.body.params.num_segments - 1

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.body.reset(noise_scale=0.01)
        self.root_vel[:] = 0.0
        self.time = 0.0
        self.step_count = 0
        self.prev_positions = self.body.forward_kinematics()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        pos_prev = self.prev_positions
        # Torques articulaires
        muscle_torques = self.body.apply_muscle_torques(action)
        drag_torques = self.fluid.drag_torque(self.body.angular_vel, self.body.params.lengths)
        self.body.step_dynamics(muscle_torques, drag_torques, self.cfg.dt)

        # Propulsion simplifiée : oscillations angulaires génèrent une poussée dans l'axe du corps.
        osc_amp = float(np.mean(np.abs(self.body.angular_vel)))
        heading = np.array([np.cos(self.body.root_theta), np.sin(self.body.root_theta)], dtype=np.float32)
        propulsion = 0.6 * osc_amp * heading

        current_here = self.fluid.current_at(self.body.root_pos, self.time)
        drag_root = -self.cfg.drag_root * (self.root_vel - current_here)
        self.root_vel += (propulsion + drag_root) * self.cfg.dt
        self.root_vel = np.clip(self.root_vel, -5.0, 5.0)
        self.body.root_pos += self.root_vel * self.cfg.dt

        # Met à jour les positions et calcule vitesse des segments pour informations
        pos_new = self.body.forward_kinematics()
        seg_vel = (pos_new - pos_prev) / self.cfg.dt
        drag_segments = self.fluid.drag_force(seg_vel, pos_new, self.time)
        net_drag_acc = drag_segments.sum(axis=0) / max(float(np.sum(self.body.params.masses)), 1e-3)
        self.root_vel += net_drag_acc * self.cfg.dt
        self.body.root_pos += net_drag_acc * (self.cfg.dt**2)
        self.prev_positions = pos_new

        self.time += self.cfg.dt
        self.step_count += 1

        obs = self._get_obs()
        if not np.isfinite(obs).all():
            # Abort if divergence; return severe penalty
            reward = -10.0
            info = {"distance": float(self.body.root_pos[0]), "energy": 0.0, "instability": 0.0}
            return obs * 0, reward, True, info
        reward, info = self._compute_reward(muscle_torques)
        done = self.step_count >= self.max_steps
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.body.angles,
                self.body.angular_vel,
                self.root_vel,
                self.fluid.current_at(self.body.root_pos, self.time),
            ]
        ).astype(np.float32)

    def _compute_reward(self, torques: np.ndarray) -> Tuple[float, Dict]:
        distance = float(self.body.root_pos[0])  # déplacement vers +x
        energy = float(np.sum(np.abs(torques * self.body.angular_vel)) * self.cfg.dt)
        instability = float(np.mean(np.square(self.body.angles)))
        reward = distance - self.cfg.energy_penalty * energy - self.cfg.stability_penalty * instability
        info = {
            "distance": distance,
            "energy": energy,
            "instability": instability,
        }
        return reward, info
