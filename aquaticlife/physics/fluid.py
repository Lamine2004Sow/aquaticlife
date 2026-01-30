from __future__ import annotations

import numpy as np


class FluidModel:
    """
    Fluide 2D visqueux avec champ de courant spatial/temps et traînée linéaire.

    Hypothèses :
    - Régime laminaire, faible Reynolds (drag proportionnelle à la vitesse relative).
    - Courant pouvant varier dans l'espace et le temps.
    """

    def __init__(
        self,
        drag_coef: float = 2.5,
        rot_drag_coef: float = 1.5,
        base_current: float = 0.0,
        current_amp: float = 0.5,
        spatial_freq: float = 0.5,
        temporal_freq: float = 0.3,
    ):
        self.drag_coef = drag_coef
        self.rot_drag_coef = rot_drag_coef
        self.base_current = np.array([base_current, 0.0], dtype=np.float32)
        self.current_amp = current_amp
        self.spatial_freq = spatial_freq
        self.temporal_freq = temporal_freq

    def current_at(self, pos: np.ndarray, t: float) -> np.ndarray:
        """
        Champ de courant sinusoïdal doux en fonction de la position (x,y) et du temps t.
        """
        k = self.spatial_freq
        w = 2 * np.pi * self.temporal_freq
        return self.base_current + self.current_amp * np.array(
            [np.sin(k * pos[1] + w * t), np.cos(k * pos[0] - w * t)], dtype=np.float32
        )

    def drag_force(self, velocities: np.ndarray, positions: np.ndarray, t: float) -> np.ndarray:
        """
        Retourne la force de traînée par segment (shape (n,2)), relative au courant local.
        """
        currents = np.vstack([self.current_at(p, t) for p in positions])
        return -self.drag_coef * (velocities - currents)

    def drag_torque(self, angular_vel: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Couple de traînée approximatif autour des articulations (rotation dans le fluide).
        """
        arm = lengths[:-1] * 0.5
        return -self.rot_drag_coef * arm * angular_vel
