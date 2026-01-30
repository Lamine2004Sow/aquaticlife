from __future__ import annotations

import sys
from typing import Tuple

import numpy as np

try:
    import pygame
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Pygame is required for live visualisation. Install with `pip install '.[viz]'`.") from exc


def world_to_screen(pos: np.ndarray, scale: float, offset: Tuple[int, int]) -> Tuple[int, int]:
    x = int(pos[0] * scale + offset[0])
    y = int(offset[1] - pos[1] * scale)  # y up in world, down on screen
    return x, y


class PygameViewer:
    """Affiche en temps réel les segments et la trajectoire d'un organisme."""

    def __init__(self, env, controller, window=(900, 600), scale: float = 500.0):
        self.env = env
        self.controller = controller
        self.window = window
        self.scale = scale
        pygame.init()
        self.screen = pygame.display.set_mode(window)
        pygame.display.set_caption("AquaticLife - visualisation temps réel")
        self.clock = pygame.time.Clock()
        self.offset = (window[0] // 4, window[1] // 2)

    def draw_body(self):
        positions = self.env.body.forward_kinematics()
        pts = [world_to_screen(p, self.scale, self.offset) for p in positions if np.isfinite(p).all()]
        self.screen.fill((15, 18, 30))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, (70, 200, 255), False, pts, width=4)
        for p in pts:
            pygame.draw.circle(self.screen, (255, 255, 255), p, 5)

    def run(self, episodes: int = 1, fps: int = 60):
        for ep in range(episodes):
            obs = self.env.reset(seed=ep)
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                action = self.controller(obs)
                obs, reward, done, info = self.env.step(action)
                self.draw_body()
                pygame.display.flip()
                self.clock.tick(fps)
        pygame.quit()
