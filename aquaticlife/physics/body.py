from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class MorphologyParameters:
    """Paramètres évolués décrivant la morphologie d'un organisme segmenté."""

    num_segments: int
    lengths: np.ndarray  # shape (num_segments,)
    masses: np.ndarray  # shape (num_segments,)
    joint_stiffness: np.ndarray  # shape (num_segments - 1,)
    joint_damping: np.ndarray  # shape (num_segments - 1,)
    muscle_strength: np.ndarray  # shape (num_segments - 1,)

    def clamp(self, min_segments: int = 2, max_segments: int = 8) -> "MorphologyParameters":
        """Borne les valeurs pour rester dans l'espace de recherche autorisé."""
        n = int(np.clip(self.num_segments, min_segments, max_segments))
        self.lengths = np.clip(self.lengths[:n], 0.02, 0.25)
        self.masses = np.clip(self.masses[:n], 0.01, 1.0)
        self.joint_stiffness = np.clip(self.joint_stiffness[: n - 1], 0.1, 50.0)
        self.joint_damping = np.clip(self.joint_damping[: n - 1], 0.001, 2.0)
        self.muscle_strength = np.clip(self.muscle_strength[: n - 1], 0.05, 10.0)
        self.num_segments = n
        return self


class RigidSegmentChain:
    """
    Chaîne 2D de segments rigides articulés.

    L'état est décrit en coordonnées articulaires (angles relatifs) et vitesses.
    Les positions XY globales du centre des segments sont dérivées à la volée
    à partir d'une cinématique directe simple (plan).
    """

    def __init__(self, params: MorphologyParameters):
        self.params = params.clamp()
        self.angles = np.zeros(self.params.num_segments - 1, dtype=np.float32)  # rad
        self.angular_vel = np.zeros_like(self.angles)
        # Position absolue du premier segment (COM) et orientation globale
        self.root_pos = np.zeros(2, dtype=np.float32)
        self.root_theta = 0.0

    def reset(self, noise_scale: float = 0.01) -> None:
        self.angles = noise_scale * np.random.randn(self.params.num_segments - 1).astype(np.float32)
        self.angular_vel = np.zeros_like(self.angles)
        self.root_pos[:] = 0.0
        self.root_theta = 0.0

    def forward_kinematics(self) -> np.ndarray:
        """
        Retourne les positions XY des centres de chaque segment.
        shape: (num_segments, 2)
        """
        n = self.params.num_segments
        pos = np.zeros((n, 2), dtype=np.float32)
        theta = self.root_theta
        x, y = self.root_pos
        for i in range(n):
            length = self.params.lengths[i]
            if i > 0:
                theta += self.angles[i - 1]
            # centre du segment dans son repère local au milieu de la barre
            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)
            x += dx
            y += dy
            pos[i] = (x, y)
            # avance jusqu'à l'extrémité pour la prochaine articulation
            x += dx
            y += dy
        return pos

    def apply_muscle_torques(self, activations: np.ndarray) -> np.ndarray:
        """
        Convertit des activations musculaires [-1, 1] en couples aux articulations.
        """
        act = np.clip(activations, -1.0, 1.0)
        return act * self.params.muscle_strength

    def step_dynamics(
        self,
        muscle_torques: np.ndarray,
        drag_torques: np.ndarray,
        dt: float,
    ) -> None:
        """
        Intégration angulaire simple (semi-implicite) sous torques articulaires.
        """
        stiffness = self.params.joint_stiffness
        damping = self.params.joint_damping
        # Couples ressort et amortisseur
        spring_torque = -stiffness * self.angles
        damper_torque = -damping * self.angular_vel
        total = muscle_torques + drag_torques + spring_torque + damper_torque
        inertia = self.params.masses[:-1] * (self.params.lengths[:-1] ** 2)
        inertia = np.maximum(inertia, 1e-3)
        self.angular_vel += (total / inertia) * dt
        self.angles += self.angular_vel * dt

    def copy(self) -> "RigidSegmentChain":
        clone = RigidSegmentChain(params=self.params)
        clone.angles = self.angles.copy()
        clone.angular_vel = self.angular_vel.copy()
        clone.root_pos = self.root_pos.copy()
        clone.root_theta = float(self.root_theta)
        return clone
