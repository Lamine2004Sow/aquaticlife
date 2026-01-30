from __future__ import annotations

import numpy as np


class FluidModel:
    """
    Fluide 2D visqueux simplifié.

    Hypothèses :
    - Régime laminaire, faible Reynolds.
    - Force de traînée linéaire F = -k_drag * v par segment.
    - Courant optionnel : champ vectoriel stationnaire.
    """

    def __init__(self, drag_coef: float = 2.5, current: float = 0.0):
        self.drag_coef = drag_coef
        self.current = np.array([current, 0.0], dtype=np.float32)

    def drag_force(self, velocities: np.ndarray) -> np.ndarray:
        """Retourne la force de traînée par segment (shape (n,2))."""
        return -self.drag_coef * (velocities - self.current)

    def drag_torque(self, angular_vel: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Couple de traînée approximatif autour des articulations.
        """
        arm = lengths[:-1] * 0.5
        return -self.drag_coef * arm * angular_vel
